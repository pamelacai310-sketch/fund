from __future__ import annotations

import argparse
import pandas as pd
from input_reader import read_input_file
from fund_parser import build_fund_summary, extract_funds, group_underlying_funds
from search_docs import search_latest_documents
from social_search import search_social_discussions, read_social_comments_csv
from analyzer import analyze_official_docs, analyze_social_comments
from master_skills import build_master_skill_outputs
from mention_pipeline import build_audit_summary, build_mentions_from_social_results, build_review_queue
from product_master import build_product_master_data
from report_writer import write_report
from structured_products import build_structured_product_outputs
from classification_engine import build_classification_outputs
from market_data import build_market_data_outputs
from peer_scoring import build_peer_score_outputs
from window import default_window


def run_pipeline(
    input_files: list[str],
    enable_doc_search: bool = True,
    enable_social_search: bool = True,
    enable_market_data: bool = True,
    social_comments_csv: str | None = None,
    market_cache_dir: str = '.cache/fund_market_data',
    as_of_date: str | None = None,
    risk_free_rate: float = 0.0,
    output_name: str = 'fund_research_report.xlsx',
):
    all_raw = []
    print('步骤 1/8：读取输入文件...')
    for file_path in input_files:
        df = read_input_file(file_path)
        df['source_file'] = file_path
        all_raw.append(df)
    if not all_raw:
        raise ValueError('没有可读取的输入文件。')
    raw_df = pd.concat(all_raw, ignore_index=True)

    print('步骤 2/8：抽取基金产品...')
    funds_df = extract_funds(raw_df)

    print('步骤 3/8：归并底层基金和份额...')
    share_df, base_df = group_underlying_funds(funds_df)
    summary_df = build_fund_summary(share_df, base_df)
    print(f'识别到份额数量：{len(share_df)}')
    print(f'识别到底层基金数量：{len(base_df)}')

    print('步骤 4/8：搜索官方资料/月报...')
    docs_df = search_latest_documents(base_df) if enable_doc_search else pd.DataFrame()
    analysis_docs_df = (
        docs_df[docs_df['analysis_eligible'].fillna(False).astype(bool)].copy()
        if not docs_df.empty and 'analysis_eligible' in docs_df else docs_df
    )

    print('步骤 5/8：构建确定性分类、策略指纹和同类关系...')
    provisional_master_df = build_product_master_data(base_df, share_df, analysis_docs_df, pd.DataFrame())
    classification_outputs = build_classification_outputs(base_df, analysis_docs_df, provisional_master_df)
    official_analysis_df = analyze_official_docs(
        base_df,
        analysis_docs_df,
        classification_df=classification_outputs['classification_df'],
        peer_edges_df=classification_outputs['peer_edges_df'],
    )
    product_master_df = build_product_master_data(
        base_df,
        share_df,
        analysis_docs_df,
        official_analysis_df,
        classification_df=classification_outputs['classification_df'],
        fingerprint_df=classification_outputs['fingerprint_df'],
    )

    print('步骤 6/8：加载免费行情并重建总回报序列...')
    market_outputs = build_market_data_outputs(
        share_df if enable_market_data else pd.DataFrame(),
        as_of_date=as_of_date,
        cache_dir=market_cache_dir,
    )
    print('步骤 7/8：计算同类基金双轴评分和份额体验...')
    score_outputs = build_peer_score_outputs(
        base_df=base_df,
        series_master_df=market_outputs['series_master_df'],
        observations_df=market_outputs['observations_df'],
        classification_df=classification_outputs['classification_df'],
        fingerprint_df=classification_outputs['fingerprint_df'],
        product_master_df=product_master_df,
        risk_free_rate=risk_free_rate,
    )

    print('步骤 8/8：搜索、分析社媒评论并写入报告...')
    social_frames = []
    if enable_social_search:
        social_frames.append(search_social_discussions(share_df))
    if social_comments_csv:
        social_frames.append(read_social_comments_csv(social_comments_csv))
    social_df = pd.concat(social_frames, ignore_index=True) if social_frames else pd.DataFrame()
    social_window = default_window()
    mention_df = build_mentions_from_social_results(social_df, social_window)
    social_analysis_df = analyze_social_comments(mention_df)
    review_queue_df = build_review_queue(mention_df)
    audit_df = build_audit_summary(social_df, mention_df, social_window)
    master_outputs = build_master_skill_outputs(
        base_df=base_df,
        official_analysis_df=official_analysis_df,
        docs_df=analysis_docs_df,
        social_analysis_df=social_analysis_df,
    )
    structured_outputs = build_structured_product_outputs(product_master_df)

    output_path = write_report(
        summary_df=summary_df,
        share_df=share_df,
        base_df=base_df,
        docs_df=docs_df,
        official_analysis_df=official_analysis_df,
        product_master_df=product_master_df,
        social_df=social_df,
        mention_df=mention_df,
        social_analysis_df=social_analysis_df,
        audit_df=audit_df,
        review_queue_df=review_queue_df,
        **master_outputs,
        **structured_outputs,
        **classification_outputs,
        **market_outputs,
        **score_outputs,
        output_name=output_name,
    )
    print(f'完成，报告已输出：{output_path}')
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description='基金产品识别、归并、官方资料搜索和社媒分析工具')
    parser.add_argument('input_files', nargs='+', help='输入文件路径，可为图片、Excel 或 CSV')
    parser.add_argument('--no-doc-search', action='store_true', help='不搜索官方资料/月报')
    parser.add_argument('--no-social-search', action='store_true', help='不搜索小红书/微博/抖音公开网页结果')
    parser.add_argument('--no-market-data', action='store_true', help='不加载免费历史净值和计算基金评分')
    parser.add_argument('--social-comments-csv', default=None, help='用户自行导出的社媒评论 CSV 路径')
    parser.add_argument('--market-cache-dir', default='.cache/fund_market_data', help='免费行情原始响应缓存目录')
    parser.add_argument('--as-of-date', default=None, help='评分截止日，格式 YYYY-MM-DD；默认运行日')
    parser.add_argument('--risk-free-rate', type=float, default=0.0, help='年化无风险利率小数，例如 0.02')
    parser.add_argument('--output-name', default='fund_research_report.xlsx', help='输出工作簿文件名')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_pipeline(
        input_files=args.input_files,
        enable_doc_search=not args.no_doc_search,
        enable_social_search=not args.no_social_search,
        enable_market_data=not args.no_market_data,
        social_comments_csv=args.social_comments_csv,
        market_cache_dir=args.market_cache_dir,
        as_of_date=args.as_of_date,
        risk_free_rate=args.risk_free_rate,
        output_name=args.output_name,
    )
