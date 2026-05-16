from __future__ import annotations

import io
import re
import time
from datetime import date, datetime
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from config import (
    DOC_KEYWORDS,
    FETCH_DOCUMENT_TEXT,
    MAX_DOC_RESULTS_PER_FUND,
    OFFICIAL_DOMAINS,
    SERPAPI_KEY,
    TAVILY_API_KEY,
)


DOC_COLUMNS = [
    'base_fund_id', 'base_fund_name', 'query', 'title', 'link', 'snippet',
    'source_domain', 'source_quality', 'doc_type_guess', 'document_date',
    'date_source', 'freshness_score', 'relevance_score', 'latest_rank',
    'is_latest_candidate', 'fetch_status', 'text_excerpt',
]

DOC_TYPE_WEIGHTS = {
    'monthly_report': 40,
    'factsheet': 35,
    'kfs': 30,
    'prospectus': 20,
    'pdf': 12,
    'webpage': 5,
}

MONTH_NAME_RE = re.compile(
    r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
    r'jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
    r'[\s.-]+20\d{2}\b',
    re.I,
)
YEAR_MONTH_RE = re.compile(r'\b(20\d{2})[-年./](0?[1-9]|1[0-2])\b')
MONTH_YEAR_CN_RE = re.compile(r'\b(20\d{2})年(0?[1-9]|1[0-2])月\b')


def build_doc_query(fund_name: str, identifiers: list[str] | None = None) -> str:
    keywords = ' OR '.join([f'"{k}"' for k in DOC_KEYWORDS])
    aliases = [fund_name] + [x for x in (identifiers or []) if x]
    alias_query = ' OR '.join([f'"{alias}"' for alias in aliases[:4]])
    return f'({alias_query}) ({keywords}) latest PDF'


def serpapi_search(query: str, max_results: int = 10) -> list[dict]:
    if not SERPAPI_KEY:
        return []
    params = {'engine': 'google', 'q': query, 'api_key': SERPAPI_KEY, 'num': max_results}
    resp = requests.get('https://serpapi.com/search.json', params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get('organic_results', [])[:max_results]


def tavily_search(query: str, max_results: int = 10) -> list[dict]:
    if not TAVILY_API_KEY:
        return []
    payload = {
        'api_key': TAVILY_API_KEY,
        'query': query,
        'search_depth': 'advanced',
        'max_results': max_results,
        'include_raw_content': False,
    }
    resp = requests.post('https://api.tavily.com/search', json=payload, timeout=30)
    resp.raise_for_status()
    results = []
    for item in resp.json().get('results', [])[:max_results]:
        results.append({
            'title': item.get('title', ''),
            'link': item.get('url', ''),
            'snippet': item.get('content', ''),
        })
    return results


def merged_search(query: str, max_results: int) -> list[dict]:
    seen, out = set(), []
    for source_name, search_fn in [('serpapi', serpapi_search), ('tavily', tavily_search)]:
        for item in search_fn(query, max_results=max_results):
            link = item.get('link') or item.get('url') or ''
            if not link or link in seen:
                continue
            seen.add(link)
            row = dict(item)
            row['search_provider'] = source_name
            out.append(row)
    return out[:max_results]


def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace('www.', '')
    except Exception:
        return ''


def source_quality(domain: str) -> str:
    if any(domain.endswith(official) for official in OFFICIAL_DOMAINS):
        return 'official_or_manager'
    if any(key in domain for key in ['morningstar', 'fundsupermart', 'ifast', 'aastocks', 'hk.morningstar']):
        return 'trusted_fund_database'
    if domain:
        return 'third_party_or_media'
    return 'unknown'


def guess_doc_type(text: str) -> str:
    t = text.lower()
    if 'monthly' in t or '月报' in t or '月報' in t:
        return 'monthly_report'
    if 'factsheet' in t or 'fact-sheet' in t or '基金资料' in t or '基金資料' in t:
        return 'factsheet'
    if 'kfs' in t or 'product key facts' in t or '产品资料概要' in t or '產品資料概要' in t:
        return 'kfs'
    if 'prospectus' in t or '招募说明书' in t or '說明書' in t:
        return 'prospectus'
    if '.pdf' in t:
        return 'pdf'
    return 'webpage'


def extract_document_date(*texts: str) -> tuple[str, str]:
    combined = ' '.join(str(t or '') for t in texts)
    for match in MONTH_YEAR_CN_RE.finditer(combined):
        return f'{int(match.group(1)):04d}-{int(match.group(2)):02d}-01', 'cn_year_month'
    for match in YEAR_MONTH_RE.finditer(combined):
        return f'{int(match.group(1)):04d}-{int(match.group(2)):02d}-01', 'year_month'
    match = MONTH_NAME_RE.search(combined)
    if match:
        try:
            parsed = date_parser.parse(match.group(0), default=datetime(1900, 1, 1))
            return parsed.date().replace(day=1).isoformat(), 'month_name'
        except (ValueError, OverflowError):
            pass
    for token in re.findall(r'\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b', combined):
        try:
            return date_parser.parse(token).date().isoformat(), 'full_date'
        except (ValueError, OverflowError):
            continue
    return '', ''


def fetch_document_text(url: str, max_chars: int = 15000) -> tuple[str, str]:
    if not url or not FETCH_DOCUMENT_TEXT:
        return '', 'skipped'
    try:
        resp = requests.get(url, timeout=40, headers={'User-Agent': 'Mozilla/5.0'})
        resp.raise_for_status()
    except Exception as exc:
        return '', f'fetch_error: {exc}'
    content_type = resp.headers.get('content-type', '').lower()
    if 'pdf' in content_type or url.lower().split('?')[0].endswith('.pdf'):
        return extract_pdf_text(resp.content, max_chars=max_chars), 'fetched_pdf'
    return extract_html_text(resp.text, max_chars=max_chars), 'fetched_html'


def extract_html_text(html: str, max_chars: int = 15000) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    lines = [line.strip() for line in soup.get_text('\n').splitlines() if line.strip()]
    return '\n'.join(lines)[:max_chars]


def extract_pdf_text(pdf_bytes: bytes, max_chars: int = 15000) -> str:
    import pdfplumber
    texts = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages[:10]:
                texts.append(page.extract_text() or '')
    except Exception as exc:
        return f'[PDF_EXTRACT_ERROR] {exc}'
    return '\n'.join(texts)[:max_chars]


def score_document(row: dict, today: date | None = None) -> tuple[int, int]:
    today = today or date.today()
    doc_type_score = DOC_TYPE_WEIGHTS.get(row.get('doc_type_guess', 'webpage'), 0)
    quality_score = {
        'official_or_manager': 35,
        'trusted_fund_database': 20,
        'third_party_or_media': 8,
        'unknown': 0,
    }.get(row.get('source_quality'), 0)
    date_score = 0
    if row.get('document_date'):
        try:
            doc_date = date_parser.parse(row['document_date']).date()
            months_old = max(0, (today.year - doc_date.year) * 12 + today.month - doc_date.month)
            date_score = max(0, 40 - months_old * 3)
        except (ValueError, OverflowError):
            date_score = 0
    relevance = doc_type_score + quality_score + date_score
    freshness = date_score + doc_type_score
    return freshness, relevance


def identifiers_for_fund(row: pd.Series) -> list[str]:
    values = []
    for col in ['product_codes', 'isins', 'morningstar_codes', 'fund_names_cn', 'fund_names_en']:
        for part in str(row.get(col, '') or '').replace('|', ',').split(','):
            part = part.strip()
            if part:
                values.append(part)
    return values


def search_latest_documents(base_df: pd.DataFrame, max_results_per_fund: int = MAX_DOC_RESULTS_PER_FUND) -> pd.DataFrame:
    if not SERPAPI_KEY and not TAVILY_API_KEY:
        return pd.DataFrame(columns=DOC_COLUMNS)
    rows = []
    for _, fund in base_df.iterrows():
        query = build_doc_query(fund['base_fund_name'], identifiers_for_fund(fund))
        for result in merged_search(query, max_results=max_results_per_fund):
            title = result.get('title', '')
            link = result.get('link') or result.get('url') or ''
            snippet = result.get('snippet', '')
            domain = get_domain(link)
            doc_text, fetch_status = fetch_document_text(link)
            doc_date, date_source = extract_document_date(title, link, snippet, doc_text[:3000])
            doc_type = guess_doc_type(' '.join([title, link, snippet, doc_text[:1000]]))
            row = {
                'base_fund_id': fund['base_fund_id'],
                'base_fund_name': fund['base_fund_name'],
                'query': query,
                'title': title,
                'link': link,
                'snippet': snippet,
                'source_domain': domain,
                'source_quality': source_quality(domain),
                'doc_type_guess': doc_type,
                'document_date': doc_date,
                'date_source': date_source,
                'fetch_status': fetch_status,
                'text_excerpt': doc_text[:1200],
            }
            row['freshness_score'], row['relevance_score'] = score_document(row)
            rows.append(row)
        time.sleep(0.25)
    if not rows:
        return pd.DataFrame(columns=DOC_COLUMNS)
    df = pd.DataFrame(rows)
    df = df.sort_values(['base_fund_id', 'relevance_score', 'freshness_score'], ascending=[True, False, False])
    df['latest_rank'] = df.groupby('base_fund_id').cumcount() + 1
    df['is_latest_candidate'] = df['latest_rank'].eq(1)
    return df[DOC_COLUMNS].reset_index(drop=True)
