from pathlib import Path
import pandas as pd
from config import OUTPUT_DIR


def write_report(
    summary_df: pd.DataFrame,
    share_df: pd.DataFrame,
    base_df: pd.DataFrame,
    docs_df: pd.DataFrame,
    official_analysis_df: pd.DataFrame,
    social_df: pd.DataFrame,
    mention_df: pd.DataFrame,
    social_analysis_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    review_queue_df: pd.DataFrame,
    output_name: str = 'fund_research_report.xlsx',
) -> Path:
    output_path = OUTPUT_DIR / output_name
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='基金清单摘要', index=False)
        share_df.to_excel(writer, sheet_name='全部份额', index=False)
        base_df.to_excel(writer, sheet_name='底层基金统计', index=False)
        docs_df.to_excel(writer, sheet_name='官方资料搜索结果', index=False)
        official_analysis_df.to_excel(writer, sheet_name='官方资料分析', index=False)
        social_df.to_excel(writer, sheet_name='基金公司社媒搜索结果', index=False)
        mention_df.to_excel(writer, sheet_name='基金公司评论标准记录', index=False)
        social_analysis_df.to_excel(writer, sheet_name='基金公司痛点卖点', index=False)
        audit_df.to_excel(writer, sheet_name='采集审计摘要', index=False)
        review_queue_df.to_excel(writer, sheet_name='人工复核队列', index=False)
    return output_path
