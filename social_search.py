from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from urllib.parse import urlparse

import pandas as pd
import requests
from dateutil import parser as date_parser

from config import MAX_SOCIAL_RESULTS_PER_QUERY, SERPAPI_KEY, SOCIAL_PLATFORMS, SOCIAL_RECENT_DAYS


SOCIAL_COLUMNS = [
    'product_code', 'base_fund_id', 'base_fund_name', 'platform', 'source_type',
    'query', 'title', 'link', 'user_text', 'publish_time', 'is_recent',
    'like_count', 'date_source',
]


def recent_window(today: date | None = None) -> tuple[date, date]:
    end = today or date.today()
    return end - timedelta(days=SOCIAL_RECENT_DAYS), end


def build_social_keywords(product_code: str, fund_name_cn: str, fund_name_en: str, base_name: str) -> list[str]:
    values = [product_code, fund_name_cn, base_name, fund_name_en]
    out, seen = [], set()
    for value in values:
        value = str(value or '').strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            out.append(value)
    return out[:4]


def platform_site_filter(platform: str) -> str:
    if platform == '小红书':
        return 'site:xiaohongshu.com OR site:xhslink.com'
    if platform == '微博':
        return 'site:weibo.com OR site:m.weibo.cn'
    if platform == '抖音':
        return 'site:douyin.com OR site:iesdouyin.com'
    return ''


def build_social_query(keyword: str, platform: str, since_date: date, until_date: date) -> str:
    pain_words = '亏 OR 跌 OR 分红 OR 派息 OR 净值 OR 对冲 OR 收益 OR 买了 OR 怎么样 OR 体验'
    site = platform_site_filter(platform)
    return f'"{keyword}" ({pain_words}) ({site}) after:{since_date.isoformat()} before:{until_date.isoformat()}'


def serpapi_search(query: str, max_results: int = 5) -> list[dict]:
    if not SERPAPI_KEY:
        return []
    params = {'engine': 'google', 'q': query, 'api_key': SERPAPI_KEY, 'num': max_results}
    resp = requests.get('https://serpapi.com/search.json', params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get('organic_results', [])[:max_results]


def infer_platform(url: str, fallback: str = '') -> str:
    domain = urlparse(str(url)).netloc.lower()
    if 'xiaohongshu' in domain or 'xhslink' in domain:
        return '小红书'
    if 'weibo' in domain:
        return '微博'
    if 'douyin' in domain or 'iesdouyin' in domain:
        return '抖音'
    return fallback


def parse_date(value: str) -> str:
    if not value:
        return ''
    try:
        return date_parser.parse(str(value), fuzzy=True).date().isoformat()
    except (ValueError, OverflowError, TypeError):
        return ''


def is_relevant_social_result(base_name: str, title: str, snippet: str) -> bool:
    text = f'{title} {snippet}'.lower()
    tokens = [
        token.lower()
        for token in str(base_name).replace('-', ' ').replace('(', ' ').replace(')', ' ').split()
        if len(token) >= 4 and token.lower() not in {'fund', 'class', 'income', 'equity', 'bond'}
    ]
    if not tokens:
        return True
    hits = sum(1 for token in tokens[:5] if token in text)
    return hits >= min(2, len(tokens))


def is_recent_date(value: str, since: date, until: date) -> bool:
    parsed = parse_date(value)
    if not parsed:
        return False
    d = date_parser.parse(parsed).date()
    return since <= d <= until


def search_social_discussions(share_df: pd.DataFrame, max_results_per_code: int = MAX_SOCIAL_RESULTS_PER_QUERY) -> pd.DataFrame:
    if not SERPAPI_KEY:
        return pd.DataFrame(columns=SOCIAL_COLUMNS)
    rows = []
    search_df = build_social_search_units(share_df)
    since_date, until_date = recent_window()
    for _, row in search_df.iterrows():
        product_code = row.get('product_code', '')
        keywords = build_social_keywords(product_code, row.get('fund_name_cn', ''), row.get('fund_name_en', ''), row.get('base_fund_name', ''))
        for platform in SOCIAL_PLATFORMS:
            for keyword in keywords:
                query = build_social_query(keyword, platform, since_date, until_date)
                for result in serpapi_search(query, max_results=max_results_per_code):
                    link = result.get('link', '')
                    if not is_relevant_social_result(row.get('base_fund_name', ''), result.get('title', ''), result.get('snippet', '')):
                        continue
                    publish_time = parse_date(result.get('date', ''))
                    rows.append({
                        'product_code': product_code,
                        'base_fund_id': row.get('base_fund_id', ''),
                        'base_fund_name': row.get('base_fund_name', ''),
                        'platform': infer_platform(link, platform),
                        'source_type': 'public_search_result',
                        'query': query,
                        'title': result.get('title', ''),
                        'link': link,
                        'user_text': result.get('snippet', ''),
                        'publish_time': publish_time,
                        'is_recent': is_recent_date(publish_time, since_date, until_date) if publish_time else '',
                        'like_count': '',
                        'date_source': 'search_result_date' if publish_time else 'query_window_only',
                    })
                time.sleep(0.2)
    if not rows:
        return pd.DataFrame(columns=SOCIAL_COLUMNS)
    return pd.DataFrame(rows)[SOCIAL_COLUMNS].drop_duplicates(subset=['product_code', 'platform', 'link', 'user_text']).reset_index(drop=True)


def build_social_search_units(share_df: pd.DataFrame) -> pd.DataFrame:
    if share_df.empty or 'base_fund_id' not in share_df.columns:
        return share_df
    rows = []
    for base_id, group in share_df.groupby('base_fund_id', dropna=False):
        first = group.iloc[0]
        rows.append({
            'product_code': ', '.join(sorted(set(x for x in group.get('product_code', pd.Series(dtype=str)).astype(str) if x))),
            'base_fund_id': base_id,
            'base_fund_name': first.get('base_fund_name', ''),
            'fund_name_cn': first.get('fund_name_cn', ''),
            'fund_name_en': first.get('fund_name_en', ''),
        })
    return pd.DataFrame(rows)


def read_social_comments_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    required = ['product_code', 'platform', 'user_text']
    for col in required:
        if col not in df.columns:
            raise ValueError(f'评论 CSV 缺少必要列: {col}')
    if 'url' in df.columns and 'link' not in df.columns:
        df['link'] = df['url']
    since_date, until_date = recent_window()
    for col in SOCIAL_COLUMNS:
        if col not in df.columns:
            df[col] = ''
    df['source_type'] = df['source_type'].fillna('').replace('', 'user_exported_comment')
    df['publish_time'] = df['publish_time'].apply(parse_date)
    df['is_recent'] = df['publish_time'].apply(lambda value: is_recent_date(value, since_date, until_date) if value else '')
    df['date_source'] = df['publish_time'].apply(lambda value: 'comment_publish_time' if value else '')
    return df[SOCIAL_COLUMNS]


def filter_recent_social(social_df: pd.DataFrame) -> pd.DataFrame:
    if social_df.empty or 'is_recent' not in social_df.columns:
        return social_df
    unknown = social_df['is_recent'].astype(str).eq('')
    recent = social_df['is_recent'].astype(str).str.lower().isin(['true', '1'])
    return social_df[unknown | recent].reset_index(drop=True)
