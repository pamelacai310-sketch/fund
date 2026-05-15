import time
from urllib.parse import urlparse
import requests
import pandas as pd
from bs4 import BeautifulSoup
from config import SERPAPI_KEY, DOC_KEYWORDS


def build_doc_query(fund_name: str) -> str:
    keywords = ' OR '.join([f'"{k}"' for k in DOC_KEYWORDS])
    return f'"{fund_name}" ({keywords}) latest PDF'


def serpapi_search(query: str, max_results: int = 10) -> list[dict]:
    if not SERPAPI_KEY:
        print('SERPAPI_KEY is not set; skipping online document search.')
        return []
    params = {'engine': 'google', 'q': query, 'api_key': SERPAPI_KEY, 'num': max_results}
    resp = requests.get('https://serpapi.com/search.json', params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get('organic_results', [])[:max_results]


def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc
    except Exception:
        return ''


def guess_doc_type(text: str) -> str:
    t = text.lower()
    if 'factsheet' in t or 'fact-sheet' in t:
        return 'factsheet'
    if 'monthly' in t or '月报' in t:
        return 'monthly_report'
    if 'kfs' in t or 'product key facts' in t or '产品资料概要' in t:
        return 'kfs'
    if '.pdf' in t:
        return 'pdf'
    return 'webpage'


def search_latest_documents(base_df: pd.DataFrame, max_results_per_fund: int = 8) -> pd.DataFrame:
    rows = []
    for _, fund in base_df.iterrows():
        query = build_doc_query(fund['base_fund_name'])
        for r in serpapi_search(query, max_results=max_results_per_fund):
            rows.append({
                'base_fund_id': fund['base_fund_id'],
                'base_fund_name': fund['base_fund_name'],
                'query': query,
                'title': r.get('title', ''),
                'link': r.get('link', ''),
                'snippet': r.get('snippet', ''),
                'source_domain': get_domain(r.get('link', '')),
                'doc_type_guess': guess_doc_type(r.get('title', '') + ' ' + r.get('link', '')),
            })
        time.sleep(0.5)
    return pd.DataFrame(rows)


def fetch_document_text(url: str, max_chars: int = 15000) -> str:
    if not url:
        return ''
    try:
        resp = requests.get(url, timeout=40, headers={'User-Agent': 'Mozilla/5.0'})
        resp.raise_for_status()
    except Exception as exc:
        return f'[FETCH_ERROR] {exc}'
    content_type = resp.headers.get('content-type', '').lower()
    if 'pdf' in content_type or url.lower().endswith('.pdf'):
        return extract_pdf_text(resp.content, max_chars=max_chars)
    return extract_html_text(resp.text, max_chars=max_chars)


def extract_html_text(html: str, max_chars: int = 15000) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    lines = [line.strip() for line in soup.get_text('\n').splitlines() if line.strip()]
    return '\n'.join(lines)[:max_chars]


def extract_pdf_text(pdf_bytes: bytes, max_chars: int = 15000) -> str:
    import io
    import pdfplumber
    texts = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages[:8]:
                texts.append(page.extract_text() or '')
    except Exception as exc:
        return f'[PDF_EXTRACT_ERROR] {exc}'
    return '\n'.join(texts)[:max_chars]
