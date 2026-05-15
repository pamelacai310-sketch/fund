import argparse
import pandas as pd
from input_reader import read_input_file
from fund_parser import extract_funds, group_underlying_funds
from search_docs import search_latest_documents
from social_search import search_social_discussions, read_social_comments_csv
from analyzer import analyze_official_docs, analyze_social_comments
from report_writer import write_report


def run_pipeline(
    input_files: list[str],
    enable_doc_search: bool = True,
    enable_social_search: bool = True,
    social_comments_csv: str | None = None,
):
    all_raw = []
    print('步骤 1/6：读取输入文件...')
    for file_path in input_files:
        df = read_input_file(file_path)
        df['source_file'] = file_path
        all_raw.append(df)
    if not all_raw:
        raise ValueError('没有可读取的输入文件。')
    raw_df = pd.concat(all_raw, ignore_index=True)

    print('步骤 2/6：抽取基金产品...')
    funds_df = extract_funds(raw_df)

    print('步骤 3/6：归并底层基金和份额...')
    share_df, base_df = group_underlying_funds(funds_df)
    print(f'识别到份额数量：{len(share_df)}')
    print(f'识别到底层基金数量：{len(base_df)}')

    print('步骤 4/6：搜索官方资料/月报...')
    docs_df = search_latest_documents(base_df) if enable_doc_search else pd.DataFrame()

    print('步骤 5/6：分析官方资料和同类产品...')
    official_analysis_df = analyze_official_docs(base_df, docs_df)

    print('步骤 6/6：搜索和分析社媒评论...')
    social_frames = []
    if enable_social_search:
        social_frames.append(search_social_discussions(share_df))
    if social_comments_csv:
        social_frames.append(read_social_comments_csv(social_comments_csv))
    social_df = pd.concat(social_frames, ignore_index=True) if social_frames else pd.DataFrame()
    social_analysis_df = analyze_social_comments(social_df)

    output_path = write_report(
        share_df=share_df,
        base_df=base_df,
        docs_df=docs_df,
        official_analysis_df=official_analysis_df,
        social_df=social_df,
        social_analysis_df=social_analysis_df,
    )
    print(f'完成，报告已输出：{output_path}')
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description='基金产品识别、归并、官方资料搜索和社媒分析工具')
    parser.add_argument('input_files', nargs='+', help='输入文件路径，可为图片、Excel 或 CSV')
    parser.add_argument('--no-doc-search', action='store_true', help='不搜索官方资料/月报')
    parser.add_argument('--no-social-search', action='store_true', help='不搜索小红书/微博/抖音公开网页结果')
    parser.add_argument('--social-comments-csv', default=None, help='用户自行导出的社媒评论 CSV 路径')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_pipeline(
        input_files=args.input_files,
        enable_doc_search=not args.no_doc_search,
        enable_social_search=not args.no_social_search,
        social_comments_csv=args.social_comments_csv,
    )
