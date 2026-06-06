import tempfile
import unittest
from datetime import date
from pathlib import Path

import openpyxl
import pandas as pd

from analyzer import analyze_official_docs, analyze_social_comments
from fund_parser import build_fund_summary, extract_funds, group_underlying_funds
from mention_pipeline import (
    build_audit_summary,
    build_content_hash,
    build_mentions_from_social_results,
    build_review_queue,
    redact_pii,
)
from report_writer import write_report
from schemas import FundCompanyMention, SourceRef, SourceType
from search_docs import extract_document_date, score_document
from social_search import build_company_search_units, filter_recent_social, read_social_comments_csv
from window import default_window, subtract_months


class FundPipelineTests(unittest.TestCase):
    def test_extracts_and_groups_share_classes(self):
        raw = pd.DataFrame({
            '基金代码': ['968000.OF', '968001.OF', '968002.OF'],
            '基金名称': ['测试亚洲债券人民币份额', '测试亚洲债券美元份额', '测试高股息股票基金'],
            'Product name': [
                'ABC Asian Total Return Bond Fund RMB',
                'ABC Asian Total Return Bond Fund USD',
                'XYZ Asian Dividend Equity Fund Acc',
            ],
            'ISIN': ['LU1234567890', 'LU1234567891', 'LU1234567892'],
        })
        funds = extract_funds(raw)
        share_df, base_df = group_underlying_funds(funds)
        summary = build_fund_summary(share_df, base_df)

        self.assertEqual(len(share_df), 3)
        self.assertEqual(len(base_df), 2)
        self.assertEqual(int(summary.iloc[0]['share_product_count']), 3)
        self.assertIn('测试亚洲债券人民币份额', summary.iloc[0]['fund_names'])
        self.assertIn('未知基金公司', summary.iloc[0]['fund_companies'])

    def test_company_search_units_group_by_fund_company(self):
        raw = pd.DataFrame({
            '基金代码': ['968000.OF', '968001.OF', '968040.OF'],
            '基金名称': ['摩根亚洲总收益债券基金 累积', '摩根亚洲总收益债券基金 派息', '惠理价值基金'],
            'Product name': [
                'JPMorgan Asian Total Return Bond Fund RMB ACC',
                'JPMorgan Asian Total Return Bond Fund RMB DIST',
                'Value Partners Classic Fund P-CNY',
            ],
            'ISIN': ['HK0000259686', 'HK0000259694', 'HK0000264959'],
        })
        share_df, _ = group_underlying_funds(extract_funds(raw))
        units = build_company_search_units(share_df)

        self.assertEqual(set(units['fund_company']), {'摩根资产管理', '惠理集团'})
        jpm = units[units['fund_company'].eq('摩根资产管理')].iloc[0]
        self.assertIn('968000.OF', jpm['product_code'])
        self.assertIn('JPMorgan', jpm['company_aliases'])

    def test_document_date_and_scoring_prefers_latest_official_monthly(self):
        doc_date, source = extract_document_date('Fund Monthly Report March 2026', 'https://manager.com/report.pdf')
        self.assertEqual(doc_date, '2026-03-01')
        self.assertEqual(source, 'month_name')

        fresh, relevance = score_document({
            'document_date': doc_date,
            'doc_type_guess': 'monthly_report',
            'source_quality': 'official_or_manager',
        }, today=date(2026, 5, 15))
        self.assertGreaterEqual(fresh, 70)
        self.assertGreaterEqual(relevance, 100)

    def test_official_analysis_uses_docs_and_peers(self):
        base_df = pd.DataFrame([
            {'base_fund_id': 'FUND_0001', 'base_fund_name': 'ABC Asian Total Return Bond Fund'},
            {'base_fund_id': 'FUND_0002', 'base_fund_name': 'DEF Asian Bond Fund'},
        ])
        docs_df = pd.DataFrame([
            {
                'base_fund_id': 'FUND_0001',
                'title': 'ABC Asian Total Return Bond Fund Factsheet March 2026',
                'snippet': 'duration credit hedged monthly report',
                'text_excerpt': 'The fund uses duration management and hedging.',
                'is_latest_candidate': True,
                'relevance_score': 110,
                'document_date': '2026-03-01',
                'link': 'https://example.com/factsheet.pdf',
                'doc_type_guess': 'factsheet',
                'source_domain': 'example.com',
            }
        ])
        analysis = analyze_official_docs(base_df, docs_df)
        first = analysis[analysis['base_fund_id'].eq('FUND_0001')].iloc[0]
        self.assertEqual(first['category'], '亚洲债券')
        self.assertIn('同类基金', first['peer_comparison'])
        self.assertIn('久期主动管理', first['rare_or_differentiated_strategies'])

    def test_social_csv_filters_old_comments_and_analyzes_pain_points(self):
        path = '/tmp/fund-social-test.csv'
        pd.DataFrame([
            {'fund_company': '摩根资产管理', 'platform': '小红书', 'user_text': '净值跌了但派息还行', 'publish_time': '2026-03-01'},
            {'fund_company': '摩根资产管理', 'platform': '微博', 'user_text': '老评论', 'publish_time': '2024-01-01'},
        ]).to_csv(path, index=False)
        social = filter_recent_social(read_social_comments_csv(path))
        mentions = build_mentions_from_social_results(social, window=(date(2025, 12, 6), date(2026, 6, 6)))
        analysis = analyze_social_comments(mentions)

        self.assertEqual(len(social), 1)
        self.assertEqual(analysis.iloc[0]['fund_company'], '摩根资产管理')
        self.assertEqual(int(analysis.iloc[0]['valid_comment_count']), 1)
        self.assertIn('亏损/回撤', analysis.iloc[0]['top_pain_aspects'])

    def test_window_uses_natural_months(self):
        self.assertEqual(subtract_months(date(2026, 6, 6), 6), date(2025, 12, 6))
        self.assertEqual(default_window(today=date(2026, 6, 6)), (date(2025, 12, 6), date(2026, 6, 6)))
        self.assertEqual(subtract_months(date(2026, 3, 31), 1), date(2026, 2, 28))

    def test_fund_company_mention_schema_validates_required_and_enum_fields(self):
        with self.assertRaises(ValueError):
            FundCompanyMention(id_hash='', source=SourceRef(platform='微博', source_type=SourceType.SOCIAL))
        with self.assertRaises(ValueError):
            SourceRef(platform='微博', source_type='invalid_source')

    def test_mentions_redact_hash_dedupe_and_flag_review(self):
        redacted, changed = redact_pii('我买了摩根基金，电话13800138000，净值跌了')
        self.assertTrue(changed)
        self.assertNotIn('13800138000', redacted)

        h1 = build_content_hash('小红书', 'https://xhslink.com/a', redacted)
        h2 = build_content_hash('小红书', 'https://xhslink.com/a', redacted)
        self.assertEqual(h1, h2)

        social = pd.DataFrame([
            {
                'fund_company': '摩根资产管理',
                'platform': '小红书',
                'source_type': 'user_exported_comment',
                'link': 'https://xhslink.com/a',
                'user_text': '我买了摩根基金，电话13800138000，净值跌了',
                'publish_time': '2026-03-01',
                'product_code': '968000.OF',
                'base_fund_name': '摩根亚洲总收益债券基金',
            },
            {
                'fund_company': '摩根资产管理',
                'platform': '小红书',
                'source_type': 'user_exported_comment',
                'link': 'https://xhslink.com/a',
                'user_text': '我买了摩根基金，电话13800138000，净值跌了',
                'publish_time': '2026-03-01',
                'product_code': '968000.OF',
                'base_fund_name': '摩根亚洲总收益债券基金',
            },
            {
                'fund_company': '摩根资产管理',
                'platform': '抖音',
                'source_type': 'public_search_result',
                'link': 'https://www.douyin.com/search/a',
                'user_text': '摩根基金怎么样',
                'publish_time': '',
            },
        ])
        mentions = build_mentions_from_social_results(social, window=(date(2025, 12, 6), date(2026, 6, 6)))
        self.assertEqual(len(mentions), 3)
        self.assertEqual(mentions.iloc[0]['duplicate_of'], '')
        self.assertEqual(mentions.iloc[1]['duplicate_of'], mentions.iloc[0]['id_hash'])
        self.assertIn('pii', mentions.iloc[0]['risk_flags'])
        self.assertIn('search_result_only', mentions.iloc[2]['risk_flags'])
        self.assertIn('low_confidence_date', mentions.iloc[2]['risk_flags'])

        review = build_review_queue(mentions)
        self.assertEqual(len(review), 1)
        audit = build_audit_summary(social, mentions, window=(date(2025, 12, 6), date(2026, 6, 6)))
        self.assertIn('deduped_comment_count', set(audit['metric']))

    def test_stale_comments_are_flagged_not_silently_dropped(self):
        social = pd.DataFrame([
            {
                'fund_company': '惠理集团',
                'platform': '微博',
                'source_type': 'social',
                'link': 'https://weibo.com/old',
                'user_text': '以前买过，回撤比较大',
                'publish_time': '2024-01-01',
            },
        ])
        mentions = build_mentions_from_social_results(social, window=(date(2025, 12, 6), date(2026, 6, 6)))
        self.assertEqual(len(mentions), 1)
        self.assertIn('stale', mentions.iloc[0]['risk_flags'])
        analysis = analyze_social_comments(mentions)
        self.assertEqual(float(analysis.iloc[0]['recent_hit_rate']), 0)

    def test_report_contains_audit_and_standardized_sheets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_name = Path(tmpdir, 'report.xlsx').name
            output_path = write_report(
                summary_df=pd.DataFrame([{'a': 1}]),
                share_df=pd.DataFrame([{'a': 1}]),
                base_df=pd.DataFrame([{'a': 1}]),
                docs_df=pd.DataFrame([{'a': 1}]),
                official_analysis_df=pd.DataFrame([{'a': 1}]),
                social_df=pd.DataFrame([{'a': 1}]),
                mention_df=pd.DataFrame([{'id_hash': 'abc'}]),
                social_analysis_df=pd.DataFrame([{'fund_company': '摩根资产管理'}]),
                audit_df=pd.DataFrame([{'metric': 'window', 'value': '2025-12-06 至 2026-06-06'}]),
                review_queue_df=pd.DataFrame([{'id_hash': 'abc'}]),
                output_name=output_name,
            )
            wb = openpyxl.load_workbook(output_path)
            self.assertIn('基金公司评论标准记录', wb.sheetnames)
            self.assertIn('采集审计摘要', wb.sheetnames)
            self.assertIn('人工复核队列', wb.sheetnames)


if __name__ == '__main__':
    unittest.main()
