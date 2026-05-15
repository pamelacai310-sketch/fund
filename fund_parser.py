from __future__ import annotations

import re
import pandas as pd
from rapidfuzz import fuzz

PRODUCT_CODE_RE = re.compile(r"\b(?:\d{6}\.OF|[A-Z]{0,3}\d{4,6}(?:\.[A-Z]{2})?)\b", re.I)
ISIN_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9,10}\b")
MORNINGSTAR_RE = re.compile(r"\b(?:FS|0P)[A-Z0-9]{8,10}\b", re.I)

COMMON_COLUMN_ALIASES = {
    '产品代码': 'product_code', 'Product code': 'product_code', 'product code': 'product_code',
    '基金代码': 'product_code', '代码': 'product_code', 'Code': 'product_code', 'Fund code': 'product_code',
    '晨星代码': 'morningstar_code', 'Morning Star Code': 'morningstar_code',
    'Morningstar Code': 'morningstar_code', 'morningstar': 'morningstar_code',
    'ISIN': 'isin', 'isin code': 'isin', '基金名称': 'fund_name_cn', '中文名称': 'fund_name_cn',
    '基金名称（中）': 'fund_name_cn', '产品名称': 'fund_name_cn',
    'Product name': 'fund_name_en', 'Fund name': 'fund_name_en', '英文名称': 'fund_name_en',
    '基金名称（英）': 'fund_name_en', 'English name': 'fund_name_en',
}

SHARE_CLASS_PATTERNS = [
    r"\bPRC[-\s]?(RMB|CNY)\b", r"\bRMB\b", r"\bCNY\b", r"\bUSD\b", r"\bHKD\b", r"\bAUD\b", r"\bSGD\b",
    r"\bHDG\b", r"\bHEDGED\b",
    r"\bACC\b", r"\bMDIST\b", r"\bDIST\b", r"\bDIS\.\b", r"\bACC\.\b",
    r"\bBC\b", r"\bBCH\b", r"\bBM2\b", r"\bBM3H\b", r"\bBM30\b", r"\bBCO\b",
    r"\bP[-\s]?CNY\b", r"\bM[-\s]?CNY\b", r"\bCLASS\s+[A-Z0-9]+\b",
]
CN_SHARE_CLASS_WORDS = ['人民币对冲', '人民币', '对冲', '累计', '累积', '派息', '每月派息', '月派息', '分派', '美元', '港元', '份额']
PLACEHOLDER_VALUES = {'', 'nan', 'none', 'null', '-', '--', 'n/a', 'na'}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: COMMON_COLUMN_ALIASES.get(str(c).strip(), c) for c in df.columns})


def clean_cell(value) -> str:
    text = str(value or '').strip()
    return '' if text.lower() in PLACEHOLDER_VALUES else text


def find_product_code(text: str) -> str:
    m = PRODUCT_CODE_RE.search(str(text).replace(' ', ''))
    return m.group(0).upper() if m else ''


def find_isin(text: str) -> str:
    m = ISIN_RE.search(str(text).replace(' ', ''))
    return m.group(0).upper() if m else ''


def find_morningstar_code(text: str) -> str:
    m = MORNINGSTAR_RE.search(str(text).replace(' ', ''))
    return m.group(0).upper() if m else ''


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text)))


def looks_like_english_fund_name(text: str) -> bool:
    t = str(text)
    words = ['fund', 'bond', 'equity', 'income', 'growth', 'dividend', 'portfolio', 'strategy', 'opportunities', 'asset']
    return len(t) >= 10 and any(w in t.lower() for w in words)


def extract_funds(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df.copy())
    if {'product_code', 'morningstar_code', 'isin', 'fund_name_cn', 'fund_name_en'}.intersection(df.columns):
        for col in ['product_code', 'morningstar_code', 'isin', 'fund_name_cn', 'fund_name_en']:
            if col not in df.columns:
                df[col] = ''
        out = df[['product_code', 'morningstar_code', 'isin', 'fund_name_cn', 'fund_name_en']].copy()
        out['product_code'] = out['product_code'].astype(str).apply(find_product_code)
        out['morningstar_code'] = out['morningstar_code'].astype(str).apply(find_morningstar_code)
        out['isin'] = out['isin'].astype(str).apply(find_isin)
        for col in ['fund_name_cn', 'fund_name_en']:
            out[col] = out[col].apply(clean_cell)
        return finalize_funds(out)
    return extract_from_unstructured(df)


def extract_from_unstructured(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        cells = [clean_cell(x) for x in row.tolist()]
        cells = [x for x in cells if x]
        line = ' '.join(cells)
        item = {
            'product_code': find_product_code(line),
            'morningstar_code': find_morningstar_code(line),
            'isin': find_isin(line),
            'fund_name_cn': '',
            'fund_name_en': '',
            'raw_line': line,
        }
        for cell in cells:
            if contains_chinese(cell) and len(cell) >= 4:
                item['fund_name_cn'] = cell
            elif looks_like_english_fund_name(cell):
                item['fund_name_en'] = cell
        if any(item.get(k) for k in ['product_code', 'morningstar_code', 'isin', 'fund_name_cn', 'fund_name_en']):
            rows.append(item)
    if not rows:
        return pd.DataFrame(columns=['product_code','morningstar_code','isin','fund_name_cn','fund_name_en','raw_line'])
    return finalize_funds(pd.DataFrame(rows))


def finalize_funds(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['product_code', 'morningstar_code', 'isin', 'fund_name_cn', 'fund_name_en']:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].apply(clean_cell)
    out = df[df['product_code'].ne('') | df['isin'].ne('') | df['morningstar_code'].ne('') | df['fund_name_cn'].ne('') | df['fund_name_en'].ne('')].copy()
    subset = ['product_code', 'morningstar_code', 'isin', 'fund_name_cn', 'fund_name_en']
    return out.drop_duplicates(subset=subset).reset_index(drop=True)


def normalize_base_name(name: str) -> str:
    s = re.sub(r"\s+", ' ', str(name).strip())
    s = re.sub(r"\s*[-–—]\s*", ' - ', s)
    for pat in SHARE_CLASS_PATTERNS:
        s = re.sub(pat, '', s, flags=re.I)
    for word in CN_SHARE_CLASS_WORDS:
        s = s.replace(word, '')
    return re.sub(r"\s+", ' ', re.sub(r"[-–—_/]+$", '', s).strip()).strip()


def derive_base_fund_name(row: pd.Series) -> str:
    name = str(row.get('fund_name_en') or row.get('fund_name_cn') or row.get('morningstar_code') or row.get('product_code') or '').strip()
    return normalize_base_name(name)


def build_fund_summary(share_df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    names = []
    for _, row in share_df.iterrows():
        label = row.get('fund_name_cn') or row.get('fund_name_en') or row.get('product_code') or row.get('isin')
        if label:
            names.append(str(label))
    return pd.DataFrame([{
        'share_product_count': len(share_df),
        'underlying_fund_count': len(base_df),
        'fund_names': ' | '.join(sorted(set(names))),
        'product_codes': ', '.join(sorted(set(x for x in share_df.get('product_code', pd.Series(dtype=str)).astype(str) if x))),
        'isins': ', '.join(sorted(set(x for x in share_df.get('isin', pd.Series(dtype=str)).astype(str) if x))),
    }])


def assign_base_ids(names: list[str], threshold: int = 92) -> list[str]:
    clusters, assigned = [], []
    for name in names:
        matched = None
        for idx, center in enumerate(clusters):
            if fuzz.token_sort_ratio(str(name).lower(), str(center).lower()) >= threshold:
                matched = idx
                break
        if matched is None:
            clusters.append(name)
            matched = len(clusters) - 1
        assigned.append(f'FUND_{matched + 1:04d}')
    return assigned


def group_underlying_funds(funds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = funds.copy()
    for col in ['product_code', 'morningstar_code', 'isin', 'fund_name_cn', 'fund_name_en']:
        if col not in df.columns:
            df[col] = ''
    df['base_fund_name'] = df.apply(derive_base_fund_name, axis=1)
    df['base_fund_id'] = assign_base_ids(df['base_fund_name'].tolist())
    grouped = []
    for base_id, g in df.groupby('base_fund_id'):
        grouped.append({
            'base_fund_id': base_id,
            'base_fund_name': sorted(set(g['base_fund_name'].astype(str)))[0],
            'share_count': len(g),
            'product_codes': ', '.join(sorted(set(x for x in g['product_code'].astype(str) if x))),
            'isins': ', '.join(sorted(set(x for x in g['isin'].astype(str) if x))),
            'morningstar_codes': ', '.join(sorted(set(x for x in g['morningstar_code'].astype(str) if x))),
            'fund_names_cn': ' | '.join(sorted(set(x for x in g['fund_name_cn'].astype(str) if x and x != 'nan'))),
            'fund_names_en': ' | '.join(sorted(set(x for x in g['fund_name_en'].astype(str) if x and x != 'nan'))),
        })
    return df.reset_index(drop=True), pd.DataFrame(grouped).sort_values(['base_fund_name']).reset_index(drop=True)
