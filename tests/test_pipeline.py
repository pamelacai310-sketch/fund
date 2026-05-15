import unittest
from datetime import date

import pandas as pd

from analyzer import analyze_official_docs, analyze_social_comments
from fund_parser import build_fund_summary, extract_funds, group_underlying_funds
from search_docs import extract_document_date, score_document
from social_search import filter_recent_social, read_social_comments_csv


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
            {'product_code': '968000.OF', 'platform': '小红书', 'user_text': '净值跌了但派息还行', 'publish_time': '2026-03-01'},
            {'product_code': '968000.OF', 'platform': '微博', 'user_text': '老评论', 'publish_time': '2024-01-01'},
        ]).to_csv(path, index=False)
        social = filter_recent_social(read_social_comments_csv(path))
        analysis = analyze_social_comments(social)

        self.assertEqual(len(social), 1)
        self.assertEqual(int(analysis.iloc[0]['recent_result_count']), 1)
        self.assertIn('回撤/亏损焦虑', analysis.iloc[0]['top_pain_points'])


if __name__ == '__main__':
    unittest.main()
