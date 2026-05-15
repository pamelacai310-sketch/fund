import time
from datetime import datetime, timedelta
import pandas as pd
import requests
from config import SERPAPI_KEY, SOCIAL_PLATFORMS, SOCIAL_RECENT_DAYS


def build_social_keywords(product_code: str, fund_name_cn: str, fund_name_en: str, base_name: str) -> list[str]:
    values = [product_code, fund_name_cn, base_name, fund_name_en]
    out, seen = [], set()
    for value in values:
        value = str(value or '').strip()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out[:3]


def build_social_query(keyword: str, platform: str, since_date: datetime) -> str:
    year, month = since_date.year, since_date.month
    pain_words = '亏 OR 跌 OR 分红 OR 派息 OR 净值 OR 对冲 OR 收益 OR 买了 OR 怎么样'
    if platform == '小红书':
        site = 'site:xiaohongshu.com'
    elif platform == '微博':
        site = 'site:weibo.com OR site:m.weibo.cn'
    elif platform == '抖音':
        site = 'site:douyin.com'
    else:
        site = ''
    return f'"{keyword}" ({pain_words}) ({platform}) ({year} OR {year}-{month:02d}) {site}'


def serpapi_search(query: str, max_results: int = 5) -> list[dict]:
    if not SERPAPI_KEY:
        print('SERPAPI_KEY is not set; skipping social web search.')
        return []
    params = {'engine': 'google', 'q': query, 'api_key': SERPAPI_KEY, 'num': max_results}
    resp = requests.get('https://serpapi.com/search.json', params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get('organic_results', [])[:max_results]


def search_social_discussions(share_df: pd.DataFrame, max_results_per_code: int = 5) -> pd.DataFrame:
    rows = []
    since_date = datetime.today() - timedelta(days=SOCIAL_RECENT_DAYS)
    for _, row in share_df.iterrows():
        product_code = row.get('product_code', '')
        keywords = build_social_keywords(product_code, row.get('fund_name_cn', ''), row.get('fund_name_en', ''), row.get('base_fund_name', ''))
        for platform in SOCIAL_PLATFORMS:
            for keyword in keywords:
                query = build_social_query(keyword, platform, since_date)
                for r in serpapi_search(query, max_results=max_results_per_code):
                    rows.append({
                        'product_code': product_code,
                        'base_fund_name': row.get('base_fund_name', ''),
                        'platform': platform,
                        'query': query,
                        'title': r.get('title', ''),
                        'link': r.get('link', ''),
                        'snippet': r.get('snippet', ''),
                        'date_guess': r.get('date', ''),
                    })
                time.sleep(0.3)
    return pd.DataFrame(rows)


def read_social_comments_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    required = ['product_code', 'platform', 'user_text']
    for col in required:
        if col not in df.columns:
            raise ValueError(f'评论 CSV 缺少必要列: {col}')
    return df
