from __future__ import annotations

import io
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from fund_parser import normalize_base_name

from config import (
    DOC_KEYWORDS,
    FETCH_DOCUMENT_TEXT,
    MAX_DOC_FETCH_PER_FUND,
    MAX_DOC_RESULTS_PER_FUND,
    OFFICIAL_DOMAINS,
    SERPAPI_KEY,
    TAVILY_API_KEY,
)


DOC_COLUMNS = [
    'base_fund_id', 'base_fund_name', 'query', 'title', 'link', 'snippet',
    'source_domain', 'source_quality', 'doc_type_guess', 'document_date',
    'date_source', 'freshness_score', 'relevance_score', 'latest_rank',
    'is_latest_candidate', 'identity_match_score', 'identity_match_method',
    'analysis_eligible',
    'fetch_status', 'text_excerpt',
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

IDENTITY_STOPWORDS = {
    'fund', 'funds', 'class', 'share', 'shares', 'prc', 'cny', 'rmb', 'usd',
    'hkd', 'acc', 'dist', 'mdist', 'hedged', 'hdg', 'the', 'and', 'of',
    '基金', '份额', '人民幣', '人民币', '累积', '累計', '累计', '派息', '对冲', '對沖',
}

IDENTITY_FACETS = {
    'bond': [r'\bbond\b', r'债券', r'債券'],
    'equity': [r'\bequit(?:y|ies)\b', r'\bstocks?\b', r'股票'],
    'multi_asset': [r'multi[- ]?asset', r'balanced', r'portfolio', r'多资产', r'多元资产', r'平衡', r'灵活配置'],
    'asia': [r'\basia(?:n)?\b', r'亚洲', r'亞洲'],
    'hong_kong': [r'hong kong', r'香港'],
    'global': [r'\bglobal\b', r'环球', r'全球'],
    'china': [r'\bchina\b', r'中国', r'中國'],
    'pacific': [r'\bpacific\b', r'太平洋', r'亚太', r'亞太'],
    'high_yield': [r'high[- ]?yield', r'高收益'],
    'high_income': [r'high[- ]?income', r'高入息', r'高息'],
    'income': [r'\bincome\b', r'入息', r'收益'],
    'dividend': [r'dividend', r'股息', r'高息股票'],
    'growth': [r'\bgrowth\b', r'增长', r'增長'],
    'disruptive': [r'disruptive', r'创新', r'創新'],
    'volatility': [r'volatility', r'波幅', r'低波动'],
    'defensive': [r'defensive', r'防守'],
    'strategic_income': [r'strategic income', r'策略收益'],
    'classic_value': [r'\bclassic\b', r'\bvalue\b', r'价值', r'價值'],
}

ASSET_FACETS = {'bond', 'equity', 'multi_asset'}
GEOGRAPHY_FACETS = {'asia', 'hong_kong', 'global', 'china', 'pacific'}
STYLE_FACETS = {
    'high_yield', 'high_income', 'income', 'dividend', 'growth', 'disruptive',
    'volatility', 'defensive', 'strategic_income', 'classic_value',
}


def _identity_normalize(value: str) -> str:
    text = normalize_base_name(str(value or '')).lower()
    text = re.sub(r'\basian\b', 'asia', text)
    text = re.sub(r'\bequities\b', 'equity', text)
    text = re.sub(r'\bstocks\b', 'stock', text)
    return re.sub(r'[^a-z0-9\u4e00-\u9fff]+', ' ', text).strip()


def _identity_tokens(value: str) -> set[str]:
    return {token for token in _identity_normalize(value).split() if token not in IDENTITY_STOPWORDS and len(token) > 1}


def _identity_facets(value: str) -> set[str]:
    text = str(value or '').lower()
    return {
        facet for facet, patterns in IDENTITY_FACETS.items()
        if any(re.search(pattern, text, flags=re.I) for pattern in patterns)
    }


def _split_identifiers(row: pd.Series) -> list[str]:
    values = []
    for col in ['isins', 'product_codes', 'morningstar_codes']:
        for part in re.split(r'[,|；]', str(row.get(col, '') or '')):
            clean = part.strip().lower().replace('.of', '')
            if clean and clean not in values:
                values.append(clean)
    return values


def score_identity_match(fund: pd.Series, *candidate_texts: str) -> tuple[float, str]:
    candidate = ' '.join(str(value or '') for value in candidate_texts)
    candidate_lower = candidate.lower().replace('.of', '')
    for identifier in _split_identifiers(fund):
        if len(identifier) >= 6 and identifier in candidate_lower:
            return 1.0, f'identifier:{identifier}'

    aliases = [str(fund.get('base_fund_name', '') or '')]
    for col in ['fund_names_en', 'fund_names_cn']:
        aliases.extend(part.strip() for part in str(fund.get(col, '') or '').split('|') if part.strip())
    title_and_link = ' '.join(str(value or '') for value in candidate_texts[:2])
    link_text = str(candidate_texts[1] if len(candidate_texts) > 1 else '').lower()
    title_tokens = _identity_tokens(str(candidate_texts[0] if candidate_texts else ''))
    candidate_tokens = title_tokens if len(title_tokens) >= 3 else _identity_tokens(title_and_link)
    title_facets = _identity_facets(str(candidate_texts[0] if candidate_texts else ''))
    normalized_candidate = _identity_normalize(title_and_link)
    best_score = 0.0
    best_method = 'no_identity_match'
    for alias in aliases:
        alias_tokens = _identity_tokens(alias)
        if len(alias_tokens) < 2:
            continue
        normalized_alias = _identity_normalize(alias)
        alias_facets = _identity_facets(alias)
        if normalized_alias and normalized_alias in normalized_candidate:
            return 0.98, f'exact_name:{normalized_alias[:80]}'
        asset_target = alias_facets & ASSET_FACETS
        asset_candidate = title_facets & ASSET_FACETS
        geography_target = alias_facets & GEOGRAPHY_FACETS
        geography_candidate = title_facets & GEOGRAPHY_FACETS
        style_target = alias_facets & STYLE_FACETS
        style_candidate = title_facets & STYLE_FACETS
        if asset_target and asset_candidate and asset_target.isdisjoint(asset_candidate):
            continue
        if geography_target and geography_candidate and geography_target.isdisjoint(geography_candidate):
            continue
        if style_candidate and not style_candidate.issubset(style_target):
            continue
        overlap = alias_tokens & candidate_tokens
        recall = len(overlap) / len(alias_tokens)
        union = alias_tokens | candidate_tokens
        jaccard = len(overlap) / len(union) if union else 0.0
        manager_tokens = {'jpmorgan', 'amundi', 'hsbc', 'pictet', 'valuepartners', 'value', 'bea'}
        manager_in_link = any(token in link_text for token in manager_tokens & alias_tokens)
        if len(overlap) >= 3 and manager_in_link:
            score = min(0.95, 0.70 + 0.05 * len(overlap))
            if score > best_score:
                best_score = score
                best_method = f'manager_domain_plus_identity_tokens:{_identity_normalize(alias)[:80]}'
            continue
        if len(alias_tokens) < 3 or len(overlap) < 3 or recall < 0.75 or jaccard < 0.50:
            continue
        facet_bonus = 0.05 if alias_facets and alias_facets.issubset(_identity_facets(title_and_link)) else 0.0
        score = min(1.0, 0.80 * recall + 0.20 * jaccard + facet_bonus)
        if score > best_score:
            best_score = score
            best_method = f'name_tokens:{_identity_normalize(alias)[:80]}'
    return round(best_score, 4), best_method


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
        resp = requests.get(
            url,
            timeout=(8, 20),
            headers={'User-Agent': 'Mozilla/5.0'},
            stream=True,
        )
        resp.raise_for_status()
    except Exception as exc:
        return '', f'fetch_error: {exc}'
    chunks = []
    total_bytes = 0
    try:
        for chunk in resp.iter_content(chunk_size=65536):
            if not chunk:
                continue
            chunks.append(chunk)
            total_bytes += len(chunk)
            if total_bytes >= 15_000_000:
                break
    except Exception as exc:
        return '', f'fetch_error: {exc}'
    content = b''.join(chunks)
    content_type = resp.headers.get('content-type', '').lower()
    if 'pdf' in content_type or url.lower().split('?')[0].endswith('.pdf'):
        return extract_pdf_text(content, max_chars=max_chars), 'fetched_pdf'
    encoding = resp.encoding or 'utf-8'
    return extract_html_text(content.decode(encoding, errors='replace'), max_chars=max_chars), 'fetched_html'


def extract_html_text(html: str, max_chars: int = 15000) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    containers = soup.select('main, article, [role="main"]')
    root = max(containers, key=lambda node: len(node.get_text(' ', strip=True))) if containers else soup
    lines = [line.strip() for line in root.get_text('\n').splitlines() if line.strip()]
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
    for col in ['fund_names_cn', 'fund_names_en', 'isins', 'product_codes', 'morningstar_codes']:
        parts = str(row.get(col, '') or '').replace(' | ', '|').replace(',', '|').split('|')
        for part in parts:
            part = part.strip()
            if part and part not in values:
                values.append(part)
                break
    return values


def search_latest_documents(base_df: pd.DataFrame, max_results_per_fund: int = MAX_DOC_RESULTS_PER_FUND) -> pd.DataFrame:
    if not SERPAPI_KEY and not TAVILY_API_KEY:
        return pd.DataFrame(columns=DOC_COLUMNS)
    candidates = []
    for _, fund in base_df.iterrows():
        query = build_doc_query(fund['base_fund_name'], identifiers_for_fund(fund))
        for result in merged_search(query, max_results=max_results_per_fund):
            title = result.get('title', '')
            link = result.get('link') or result.get('url') or ''
            snippet = result.get('snippet', '')
            domain = get_domain(link)
            doc_type = guess_doc_type(' '.join([title, link, snippet]))
            identity_score, identity_method = score_identity_match(fund, title, link, snippet)
            if identity_score < 0.68:
                continue
            candidates.append({
                'base_fund_id': fund['base_fund_id'],
                'base_fund_name': fund['base_fund_name'],
                'query': query,
                'title': title,
                'link': link,
                'snippet': snippet,
                'source_domain': domain,
                'source_quality': source_quality(domain),
                'doc_type_guess': doc_type,
                'search_provider': result.get('search_provider', ''),
                'identity_match_score': identity_score,
                'identity_match_method': identity_method,
            })
        time.sleep(0.10)
    if not candidates:
        return pd.DataFrame(columns=DOC_COLUMNS)
    candidate_df = pd.DataFrame(candidates)
    quality_order = {'official_or_manager': 3, 'trusted_fund_database': 2, 'third_party_or_media': 1, 'unknown': 0}
    type_order = {key: value for key, value in DOC_TYPE_WEIGHTS.items()}
    candidate_df['_quality_rank'] = candidate_df['source_quality'].map(quality_order).fillna(0)
    candidate_df['_type_rank'] = candidate_df['doc_type_guess'].map(type_order).fillna(0)
    candidate_df = candidate_df.sort_values(
        ['base_fund_id', '_quality_rank', '_type_rank'],
        ascending=[True, False, False],
    )
    fetch_indices = set(
        candidate_df.groupby('base_fund_id', sort=False).head(MAX_DOC_FETCH_PER_FUND).index
    ) if FETCH_DOCUMENT_TEXT else set()
    fetched: dict[int, tuple[str, str]] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {
            executor.submit(fetch_document_text, row['link']): index
            for index, row in candidate_df.loc[list(fetch_indices)].iterrows()
        }
        for future in as_completed(future_map):
            index = future_map[future]
            try:
                fetched[index] = future.result()
            except Exception as exc:
                fetched[index] = ('', f'fetch_error: {exc}')
    rows = []
    for index, candidate in candidate_df.iterrows():
        doc_text, fetch_status = fetched.get(index, ('', 'not_fetched_low_priority'))
        doc_date, date_source = extract_document_date(
            candidate['title'], candidate['link'], candidate['snippet'], doc_text[:3000],
        )
        doc_type = guess_doc_type(' '.join([
            candidate['title'], candidate['link'], candidate['snippet'], doc_text[:1000],
        ]))
        row = {
            **{key: candidate.get(key, '') for key in [
                'base_fund_id', 'base_fund_name', 'query', 'title', 'link',
                'snippet', 'source_domain', 'source_quality',
            ]},
            'doc_type_guess': doc_type,
            'document_date': doc_date,
            'date_source': date_source,
            'fetch_status': fetch_status,
            'text_excerpt': doc_text[:15000],
            'identity_match_score': candidate.get('identity_match_score', 0),
            'identity_match_method': candidate.get('identity_match_method', ''),
        }
        row['freshness_score'], row['relevance_score'] = score_document(row)
        row['relevance_score'] += round(float(row['identity_match_score']) * 50)
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sort_values(['base_fund_id', 'relevance_score', 'freshness_score'], ascending=[True, False, False])
    df['latest_rank'] = df.groupby('base_fund_id').cumcount() + 1
    df['is_latest_candidate'] = df['latest_rank'].eq(1)
    df['analysis_eligible'] = False
    for _, group in df.groupby('base_fund_id', sort=False):
        fetched = group['fetch_status'].astype(str).str.startswith('fetched')
        preferred = group[
            fetched & group['source_quality'].isin(['official_or_manager', 'trusted_fund_database'])
        ]
        if preferred.empty:
            preferred = group[fetched & group['identity_match_score'].ge(0.90)].head(2)
        df.loc[preferred.index, 'analysis_eligible'] = True
    return df[DOC_COLUMNS].reset_index(drop=True)
