from __future__ import annotations

import re

import pandas as pd


MASTER_DATA_COLUMNS = [
    'base_fund_id', 'base_fund_name', 'fund_company', 'share_count',
    'product_codes', 'isins', 'morningstar_codes', 'fund_names_cn', 'fund_names_en',
    'latest_factsheet_title', 'latest_factsheet_link', 'latest_factsheet_date',
    'latest_kid_title', 'latest_kid_link', 'latest_kid_date',
    'latest_monthly_report_title', 'latest_monthly_report_link', 'latest_monthly_report_date',
    'investment_objective', 'asset_class', 'fund_category', 'base_currency',
    'share_class_features', 'distribution_policy', 'currency_hedging',
    'benchmark', 'management_fee', 'ongoing_charges', 'subscription_fee',
    'redemption_fee', 'switching_fee', 'official_fee_summary',
    'risk_level', 'risk_level_source', 'risk_summary',
    'top_holdings', 'asset_allocation', 'sector_allocation', 'geographic_allocation',
    'duration', 'average_credit_rating', 'yield_to_maturity', 'distribution_yield',
    'max_drawdown', 'drawdown_source', 'liquidity_terms', 'dealing_frequency',
    'settlement_period', 'redemption_liquidity', 'master_data_quality_score',
    'missing_master_fields', 'evidence_links',
]

REQUIRED_MASTER_FIELDS = [
    'official_fee_summary', 'risk_level', 'top_holdings', 'asset_allocation',
    'max_drawdown', 'liquidity_terms', 'latest_kid_link', 'latest_factsheet_link',
]

CATEGORY_ASSET_CLASS = {
    '亚洲债券': '债券',
    '亚洲高收益债': '高收益债',
    '环球债券': '债券',
    '亚洲股息/高息股票': '股票',
    '亚洲/亚太股票': '股票',
    '香港股票': '股票',
    '大中华价值股票': '股票',
    '主题成长股票': '股票',
    '多资产/平衡': '多资产',
    '环球股票': '股票',
    '策略收益': '多资产收益',
}


def _clean(value: str) -> str:
    return re.sub(r'\s+', ' ', str(value or '')).strip()


def _docs_for(base_fund_id: str, docs_df: pd.DataFrame) -> pd.DataFrame:
    if docs_df.empty or 'base_fund_id' not in docs_df.columns:
        return pd.DataFrame()
    return docs_df[docs_df['base_fund_id'].astype(str).eq(str(base_fund_id))].copy()


def _analysis_row(base_fund_id: str, official_analysis_df: pd.DataFrame) -> pd.Series | None:
    if official_analysis_df.empty or 'base_fund_id' not in official_analysis_df.columns:
        return None
    matched = official_analysis_df[official_analysis_df['base_fund_id'].astype(str).eq(str(base_fund_id))]
    if matched.empty:
        return None
    return matched.iloc[0]


def _best_doc(docs: pd.DataFrame, doc_type: str) -> pd.Series | None:
    if docs.empty or 'doc_type_guess' not in docs.columns:
        return None
    matched = docs[docs['doc_type_guess'].astype(str).eq(doc_type)]
    if matched.empty:
        return None
    return matched.sort_values(['is_latest_candidate', 'relevance_score', 'freshness_score'], ascending=[False, False, False]).iloc[0]


def _combine_doc_text(docs: pd.DataFrame) -> str:
    if docs.empty:
        return ''
    parts = []
    for _, row in docs.head(5).iterrows():
        parts.append(' '.join(str(row.get(col, '') or '') for col in ['title', 'snippet', 'text_excerpt']))
    return '\n'.join(parts)


def _first_regex(text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            return _clean(match.group(1))
    return ''


def _percent_after_label(text: str, labels: list[str]) -> str:
    label_expr = '|'.join(re.escape(label) for label in labels)
    for line in str(text or '').splitlines():
        if not re.search(label_expr, line, flags=re.I):
            continue
        patterns = [
            rf'(?:{label_expr})\s*[:：]?\s*(?:up to|最高|不超过)?\s*(-?[0-9]+(?:\.[0-9]+)?\s*%)',
            rf'(-?[0-9]+(?:\.[0-9]+)?\s*%)\s*(?:per annum|每年|p\.a\.)?\s*(?:{label_expr})',
        ]
        value = _first_regex(line, patterns)
        if value:
            return value
    return ''


def _extract_objective(text: str) -> str:
    patterns = [
        r'(?:Investment objective|投资目标|投資目標)\s*[:：]?\s*(.{30,500}?)(?:\n[A-Z][A-Za-z /]{2,40}\s*[:：]|\n\s*\n|Risk|风险|風險)',
        r'(?:Objective|目标|目標)\s*[:：]?\s*(.{30,400}?)(?:\n[A-Z][A-Za-z /]{2,40}\s*[:：]|\n\s*\n)',
    ]
    return _first_regex(text, patterns)[:500]


def _extract_benchmark(text: str) -> str:
    return _first_regex(text, [
        r'(?:Benchmark|基准|基準)\s*[:：]?\s*([^\n]{5,180})',
        r'(?:reference index|参考指数|參考指數)\s*[:：]?\s*([^\n]{5,180})',
    ])


def _extract_base_currency(text: str) -> str:
    return _first_regex(text, [
        r'(?:Base currency|Fund currency|基础货币|基礎貨幣)\s*[:：]?\s*([A-Z]{3}|人民币|美元|港元)',
    ])


def _extract_distribution_policy(text: str, fund_name: str) -> str:
    explicit = _first_regex(text, [
        r'(?:Distribution policy|Dividend policy|派息政策|分派政策)\s*[:：]?\s*([^\n]{3,180})',
    ])
    if explicit:
        return explicit
    name = fund_name.lower()
    if any(token in name for token in ['dist', 'mdist', '派息', '每月派息']):
        return '派息份额'
    if any(token in name for token in ['acc', '累计', '累积']):
        return '累积份额'
    return ''


def _extract_currency_hedging(text: str, names: str) -> str:
    combined = f'{text} {names}'.lower()
    if any(token in combined for token in ['hedged', 'hdg', '对冲', '對沖']):
        return '含人民币/币种对冲份额或对冲机制'
    return ''


def _extract_risk_level(text: str) -> tuple[str, str]:
    patterns = [
        (r'(?:SRI|SRRI|Synthetic Risk.*?Indicator|Risk indicator)\s*[:：]?\s*([1-7])\s*(?:/|out of)?\s*7?', 'SRI/SRRI'),
        (r'(?:risk rating|risk class|风险等级|風險等級|风险级别|風險級別)\s*[:：]?\s*([1-7]|低|中低|中|中高|高|低风险|中风险|高风险)', 'official_risk_rating'),
        (r'(?:风险水平|風險水平)\s*[:：]?\s*([^\n]{1,40})', 'official_risk_level'),
    ]
    for pattern, source in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            return _clean(match.group(1)), source
    return '', ''


def _extract_summary_after(text: str, labels: list[str], max_len: int = 260) -> str:
    label_expr = '|'.join(re.escape(label) for label in labels)
    match = re.search(rf'(?:{label_expr})\s*[:：]?\s*([^\n]{{3,{max_len}}})', text, flags=re.I)
    return _clean(match.group(1)) if match else ''


def _lines_with_percent(text: str, include_words: list[str], limit: int = 8) -> str:
    rows = []
    for line in text.splitlines():
        clean = _clean(line)
        if '%' not in clean:
            continue
        lowered = clean.lower()
        if include_words and not any(word.lower() in lowered for word in include_words):
            continue
        if len(clean) <= 180:
            rows.append(clean)
        if len(rows) >= limit:
            break
    return '；'.join(dict.fromkeys(rows))


def _extract_top_holdings(text: str) -> str:
    holding_block = _first_regex(text, [
        r'(?:Top holdings|Top 10 holdings|十大持仓|十大投資|主要持仓|主要投資)\s*[:：]?\s*(.{20,1000}?)(?:Asset allocation|Sector|Geographic|Country|Credit|Duration|\n\s*\n)',
    ])
    source = holding_block or text
    rows = []
    for line in source.splitlines():
        clean = _clean(line)
        if not clean or '%' not in clean:
            continue
        if re.search(r'(cash|equity|bond|sector|country|region|股票|债券|債券|现金|現金|地区|地區|行业|行業)', clean, flags=re.I):
            continue
        if 4 <= len(clean) <= 160:
            rows.append(clean)
        if len(rows) >= 10:
            break
    return '；'.join(dict.fromkeys(rows))


def _extract_asset_allocation(text: str) -> str:
    return _lines_with_percent(text, ['equity', 'bond', 'cash', 'fixed income', '股票', '债券', '債券', '现金', '現金', '基金'], limit=10)


def _extract_sector_allocation(text: str) -> str:
    return _lines_with_percent(text, ['sector', 'financial', 'technology', 'consumer', 'industrial', '行业', '行業', '金融', '科技', '消费'], limit=10)


def _extract_geographic_allocation(text: str) -> str:
    return _lines_with_percent(text, ['country', 'geographic', 'region', 'china', 'hong kong', 'asia', '地区', '地區', '国家', '國家', '中国', '香港', '亚洲'], limit=10)


def _extract_duration(text: str) -> str:
    return _first_regex(text, [
        r'(?:duration|存续期|久期)\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?\s*(?:years|yrs|年))',
        r'([0-9]+(?:\.[0-9]+)?\s*(?:years|yrs|年))\s*(?:duration|久期)',
    ])


def _extract_credit_rating(text: str) -> str:
    return _first_regex(text, [
        r'(?:average credit rating|平均信用评级|平均信貸評級)\s*[:：]?\s*([A-Z+\-]{1,5})',
        r'(?:credit rating|信用评级|信貸評級)\s*[:：]?\s*([A-Z+\-]{1,5})',
    ])


def _extract_yields(text: str) -> tuple[str, str]:
    ytm = _percent_after_label(text, ['yield to maturity', 'YTM', '到期收益率'])
    distribution = _percent_after_label(text, ['distribution yield', 'dividend yield', '派息率', '分派收益率'])
    return ytm, distribution


def _extract_drawdown(text: str) -> tuple[str, str]:
    value = _percent_after_label(text, ['maximum drawdown', 'max drawdown', '最大回撤'])
    return (value, 'official_doc_text') if value else ('', '')


def _extract_liquidity(text: str) -> tuple[str, str, str, str]:
    dealing = _extract_summary_after(text, ['Dealing frequency', 'Dealing day', '交易频率', '交易日'], max_len=160)
    settlement = _extract_summary_after(text, ['Settlement', 'Settlement period', '结算', '交收'], max_len=160)
    redemption = _extract_summary_after(text, ['Redemption', '赎回', '贖回'], max_len=220)
    terms = '；'.join(x for x in [dealing, settlement, redemption] if x)
    return terms, dealing, settlement, redemption


def _fee_summary(management: str, ongoing: str, subscription: str, redemption: str, switching: str) -> str:
    pieces = []
    for label, value in [
        ('管理费', management), ('经常性开支/ongoing charges', ongoing),
        ('申购费', subscription), ('赎回费', redemption), ('转换费', switching),
    ]:
        if value:
            pieces.append(f'{label}: {value}')
    return '；'.join(pieces)


def _quality_score(row: dict) -> int:
    present = sum(1 for field in REQUIRED_MASTER_FIELDS if str(row.get(field, '') or '').strip())
    return round(present / len(REQUIRED_MASTER_FIELDS) * 100)


def build_product_master_data(
    base_df: pd.DataFrame,
    share_df: pd.DataFrame,
    docs_df: pd.DataFrame,
    official_analysis_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, fund in base_df.iterrows():
        base_id = str(fund.get('base_fund_id', ''))
        docs = _docs_for(base_id, docs_df)
        combined_text = _combine_doc_text(docs)
        analysis = _analysis_row(base_id, official_analysis_df)
        category = str(analysis.get('category') if analysis is not None else '') or '其他'
        factsheet = _best_doc(docs, 'factsheet')
        kid = _best_doc(docs, 'kfs')
        monthly = _best_doc(docs, 'monthly_report')
        names = ' '.join(str(fund.get(col, '') or '') for col in ['base_fund_name', 'fund_names_cn', 'fund_names_en'])

        management_fee = _percent_after_label(combined_text, ['management fee', 'annual management fee', '管理费', '管理費'])
        ongoing_charges = _percent_after_label(combined_text, ['ongoing charges', 'ongoing charges figure', '经常性开支', '經常性開支'])
        subscription_fee = _percent_after_label(combined_text, ['subscription fee', 'initial charge', '认购费', '申购费', '認購費', '申購費'])
        redemption_fee = _percent_after_label(combined_text, ['redemption fee', '赎回费', '贖回費'])
        switching_fee = _percent_after_label(combined_text, ['switching fee', 'conversion fee', '转换费', '轉換費'])
        risk_level, risk_source = _extract_risk_level(combined_text)
        ytm, distribution_yield = _extract_yields(combined_text)
        max_drawdown, drawdown_source = _extract_drawdown(combined_text)
        liquidity_terms, dealing_frequency, settlement_period, redemption_liquidity = _extract_liquidity(combined_text)

        row = {
            'base_fund_id': base_id,
            'base_fund_name': fund.get('base_fund_name', ''),
            'fund_company': fund.get('fund_company', ''),
            'share_count': fund.get('share_count', ''),
            'product_codes': fund.get('product_codes', ''),
            'isins': fund.get('isins', ''),
            'morningstar_codes': fund.get('morningstar_codes', ''),
            'fund_names_cn': fund.get('fund_names_cn', ''),
            'fund_names_en': fund.get('fund_names_en', ''),
            'latest_factsheet_title': factsheet.get('title', '') if factsheet is not None else '',
            'latest_factsheet_link': factsheet.get('link', '') if factsheet is not None else '',
            'latest_factsheet_date': factsheet.get('document_date', '') if factsheet is not None else '',
            'latest_kid_title': kid.get('title', '') if kid is not None else '',
            'latest_kid_link': kid.get('link', '') if kid is not None else '',
            'latest_kid_date': kid.get('document_date', '') if kid is not None else '',
            'latest_monthly_report_title': monthly.get('title', '') if monthly is not None else '',
            'latest_monthly_report_link': monthly.get('link', '') if monthly is not None else '',
            'latest_monthly_report_date': monthly.get('document_date', '') if monthly is not None else '',
            'investment_objective': _extract_objective(combined_text),
            'asset_class': CATEGORY_ASSET_CLASS.get(category, ''),
            'fund_category': category,
            'base_currency': _extract_base_currency(combined_text),
            'share_class_features': _clean('；'.join(dict.fromkeys(re.findall(r'\b(?:CNY|RMB|USD|HKD|HDG|Hedged|ACC|DIST|MDIST|BM2|BM30|BCO)\b|人民币对冲|人民币|派息|累积|累计', names, flags=re.I)))),
            'distribution_policy': _extract_distribution_policy(combined_text, names),
            'currency_hedging': _extract_currency_hedging(combined_text, names),
            'benchmark': _extract_benchmark(combined_text),
            'management_fee': management_fee,
            'ongoing_charges': ongoing_charges,
            'subscription_fee': subscription_fee,
            'redemption_fee': redemption_fee,
            'switching_fee': switching_fee,
            'official_fee_summary': _fee_summary(management_fee, ongoing_charges, subscription_fee, redemption_fee, switching_fee),
            'risk_level': risk_level,
            'risk_level_source': risk_source,
            'risk_summary': _extract_summary_after(combined_text, ['Risk', '风险', '風險'], max_len=260),
            'top_holdings': _extract_top_holdings(combined_text),
            'asset_allocation': _extract_asset_allocation(combined_text),
            'sector_allocation': _extract_sector_allocation(combined_text),
            'geographic_allocation': _extract_geographic_allocation(combined_text),
            'duration': _extract_duration(combined_text),
            'average_credit_rating': _extract_credit_rating(combined_text),
            'yield_to_maturity': ytm,
            'distribution_yield': distribution_yield,
            'max_drawdown': max_drawdown,
            'drawdown_source': drawdown_source,
            'liquidity_terms': liquidity_terms,
            'dealing_frequency': dealing_frequency,
            'settlement_period': settlement_period,
            'redemption_liquidity': redemption_liquidity,
            'evidence_links': '；'.join(dict.fromkeys(str(x) for x in docs.get('link', pd.Series(dtype=str)).dropna().head(5))),
        }
        row['master_data_quality_score'] = _quality_score(row)
        row['missing_master_fields'] = '；'.join(field for field in REQUIRED_MASTER_FIELDS if not str(row.get(field, '') or '').strip())
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=MASTER_DATA_COLUMNS)
    return pd.DataFrame(rows)[MASTER_DATA_COLUMNS]
