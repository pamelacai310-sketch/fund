from __future__ import annotations

import re

import pandas as pd


MASTER_DATA_COLUMNS = [
    'base_fund_id', 'base_fund_name', 'fund_company', 'share_count',
    'product_codes', 'isins', 'morningstar_codes', 'fund_names_cn', 'fund_names_en',
    'latest_factsheet_title', 'latest_factsheet_link', 'latest_factsheet_date',
    'latest_kid_title', 'latest_kid_link', 'latest_kid_date',
    'latest_monthly_report_title', 'latest_monthly_report_link', 'latest_monthly_report_date',
    'investment_objective', 'investment_strategy', 'return_mechanism',
    'hard_investment_policy', 'target_exposure_range', 'asset_class', 'fund_category',
    'classification_l1', 'classification_l2', 'classification_l3',
    'classification_confidence', 'classification_route', 'policy_fingerprint_hash',
    'base_currency',
    'share_class_features', 'distribution_policy', 'currency_hedging',
    'benchmark', 'management_fee', 'ongoing_charges', 'subscription_fee',
    'redemption_fee', 'switching_fee', 'official_fee_summary',
    'risk_level', 'risk_level_source', 'risk_summary',
    'top_holdings', 'asset_allocation', 'sector_allocation', 'geographic_allocation',
    'duration', 'average_credit_rating', 'yield_to_maturity', 'distribution_yield',
    'derivative_types', 'derivative_purpose', 'short_selling_policy',
    'leverage_role', 'gross_exposure', 'net_exposure', 'market_exposure',
    'max_drawdown', 'drawdown_source', 'liquidity_terms', 'dealing_frequency',
    'settlement_period', 'redemption_liquidity',
    'is_structured_product', 'tranche_structure', 'senior_tranche_terms',
    'mezzanine_tranche_terms', 'subordinated_tranche_terms', 'waterfall_order',
    'loss_absorption_order', 'junior_cushion_ratio', 'warning_line',
    'stop_loss_line', 'underlying_liquidity_assessment',
    'rights_obligations_consistency', 'structured_product_risk_flags',
    'master_data_quality_score', 'missing_master_fields', 'evidence_links',
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
    '债券': '债券',
    '股票': '股票',
    '高息股票': '股票',
    '另类/CTA': '另类投资',
    '另类/MarketNeutral': '另类投资',
    '另类/RelativeValue': '另类投资',
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


def _extract_investment_strategy(text: str) -> str:
    return _first_regex(text, [
        r'(?:Principal investment strateg(?:y|ies)|Investment strategy|投资策略|投資策略)\s*[:：]?\s*(.{30,900}?)(?:\n[A-Z][A-Za-z /]{2,50}\s*[:：]|\n\s*\n|Risk|风险|風險)',
        r'(?:How does the fund invest|基金如何投资|基金如何投資)\s*[:：]?\s*(.{30,700}?)(?:\n\s*\n|Risk|风险|風險)',
    ])[:900]


def _extract_policy_lines(text: str) -> tuple[str, str]:
    hard_policy = []
    target_ranges = []
    for line in str(text or '').splitlines():
        clean = _clean(line)
        if not clean or len(clean) > 260:
            continue
        if re.search(r'(?:at least|minimum|not less than|不少于|至少).{0,80}[0-9]+(?:\.[0-9]+)?\s*%|[0-9]+(?:\.[0-9]+)?\s*%.{0,80}(?:minimum|at least|不少于|至少)', clean, flags=re.I):
            hard_policy.append(clean)
        if re.search(r'[0-9]+(?:\.[0-9]+)?\s*%\s*(?:-|–|—|to|至)\s*[0-9]+(?:\.[0-9]+)?\s*%', clean, flags=re.I):
            target_ranges.append(clean)
    return '；'.join(dict.fromkeys(hard_policy[:6])), '；'.join(dict.fromkeys(target_ranges[:6]))


def _extract_derivative_fields(text: str) -> tuple[str, str, str, str, str, str, str]:
    derivative_types = []
    for label, pattern in [
        ('futures', r'futures?|期货|期貨'),
        ('options', r'options?|期权|期權'),
        ('swaps', r'swaps?|掉期'),
        ('forwards', r'forwards?|远期|遠期'),
    ]:
        if re.search(pattern, text, flags=re.I):
            derivative_types.append(label)
    if re.search(r'(?:solely|only|primarily).{0,30}hedg|hedging purposes? only|仅用于.{0,15}对冲|只用于.{0,15}套期', text, flags=re.I | re.S):
        purpose = 'hedging_only'
    elif re.search(r'derivatives?.{0,50}(?:core|principal|return generation)|(?:core|principal).{0,50}derivatives?|衍生品.{0,30}(?:核心|收益来源)', text, flags=re.I | re.S):
        purpose = 'return_generation_core'
    else:
        purpose = 'unspecified' if derivative_types else ''

    if re.search(r'short.{0,30}hedg|卖空.{0,20}对冲', text, flags=re.I | re.S):
        shorting = 'hedging_only'
    elif re.search(r'short sell|short positions?|long\s*/?\s*short|卖空|做空|多空', text, flags=re.I):
        shorting = 'return_generation_or_unspecified'
    else:
        shorting = ''

    if re.search(r'leverag(?:e|ed).{0,30}(?:strategy|return)|杠杆.{0,20}(?:策略|收益)', text, flags=re.I | re.S):
        leverage = 'strategy_core'
    elif re.search(r'leverag|杠杆', text, flags=re.I):
        leverage = 'permitted_or_unspecified'
    else:
        leverage = ''

    gross = _percent_after_label(text, ['gross exposure', '总敞口', '總敞口'])
    net = _percent_after_label(text, ['net exposure', '净敞口', '淨敞口'])
    if re.search(r'market[- ]neutral|beta[- ]neutral|dollar[- ]neutral|市场中性|貝塔中性', text, flags=re.I):
        market_exposure = 'neutral'
    elif re.search(r'long\s*/?\s*short|多空', text, flags=re.I):
        market_exposure = 'long_short_unspecified_net'
    elif re.search(r'long[- ]only|只做多', text, flags=re.I):
        market_exposure = 'long_only'
    else:
        market_exposure = ''
    return '；'.join(derivative_types), purpose, shorting, leverage, gross, net, market_exposure


def _extract_return_mechanism(text: str) -> str:
    mechanisms = []
    for label, pattern in [
        ('股票Beta/资本增值', r'equity returns?|capital growth.{0,40}equit|股票增值'),
        ('票息/信用利差', r'credit spread|bond income|interest income|票息|信用利差'),
        ('股息收入', r'dividend income|high dividend|股息收入|高股息'),
        ('资产配置', r'asset allocation|资产配置|資產配置'),
        ('系统化趋势', r'systematic trend|trend following|time[- ]series momentum|趋势跟踪'),
        ('相对价值/价差', r'relative value|calendar spread|statistical arbitrage|相对价值|套利|价差'),
    ]:
        if re.search(pattern, text, flags=re.I | re.S):
            mechanisms.append(label)
    return '；'.join(mechanisms)


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


def _has_structured_terms(text: str) -> bool:
    return bool(re.search(
        r'结构化|分级|优先级?|劣后级?|夹层|平层|senior|subordinated|junior|mezzanine|tranche|waterfall',
        str(text or ''),
        flags=re.I,
    ))


def _lines_containing(text: str, words: list[str], limit: int = 4) -> str:
    rows = []
    for line in str(text or '').splitlines():
        clean = _clean(line)
        if not clean:
            continue
        if any(re.search(word, clean, flags=re.I) for word in words):
            rows.append(clean[:220])
        if len(rows) >= limit:
            break
    return '；'.join(dict.fromkeys(rows))


def _extract_tranche_structure(text: str) -> str:
    if not _has_structured_terms(text):
        return ''
    levels = []
    if re.search(r'优先级?|senior', text, flags=re.I):
        levels.append('优先')
    if re.search(r'夹层|平层|mezzanine', text, flags=re.I):
        levels.append('夹层/平层')
    if re.search(r'劣后级?|subordinated|junior', text, flags=re.I):
        levels.append('劣后')
    return ' / '.join(levels) if levels else '结构化分层已提及但层级未完整披露'


def _ratio_after_label(text: str, labels: list[str]) -> str:
    label_expr = '|'.join(re.escape(label) for label in labels)
    for line in str(text or '').splitlines():
        if not re.search(label_expr, line, flags=re.I):
            continue
        value = _first_regex(line, [
            rf'(?:{label_expr})\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?\s*%)',
            rf'([0-9]+(?:\.[0-9]+)?\s*%)\s*(?:{label_expr})',
            rf'(?:{label_expr})\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?)',
        ])
        if value:
            return value
    return ''


def _extract_structured_fields(text: str, liquidity_terms: str, asset_class: str) -> dict[str, str]:
    is_structured = _has_structured_terms(text)
    structure = _extract_tranche_structure(text)
    senior_terms = _lines_containing(text, [r'优先级?', r'senior'])
    mezzanine_terms = _lines_containing(text, [r'夹层', r'平层', r'mezzanine'])
    junior_terms = _lines_containing(text, [r'劣后级?', r'subordinated', r'junior'])
    cushion = _ratio_after_label(text, ['劣后安全垫', '劣后比例', '劣后级比例', 'junior cushion', 'subordination ratio'])
    warning_line = _ratio_after_label(text, ['预警线', '警戒线', 'warning line'])
    stop_loss_line = _ratio_after_label(text, ['止损线', '止損線', 'stop loss line', 'liquidation line'])

    combined_liquidity = f'{text} {liquidity_terms} {asset_class}'
    if re.search(r'非标|未上市|私募股权|不动产|real estate|unlisted|private equity|illiquid|低流动性', combined_liquidity, flags=re.I):
        liquidity_assessment = '较低：底层资产可能缺少可执行止损所需流动性'
    elif re.search(r'每日|daily|标准化|standardi[sz]ed|上市|交易所|债券|bond|股票|equity', combined_liquidity, flags=re.I):
        liquidity_assessment = '较高：资料显示底层资产或交易安排具备标准化/日常交易特征'
    else:
        liquidity_assessment = '待确认：资料未充分披露底层资产流动性'

    rights_consistency = ''
    if re.search(r'权利义务.{0,20}(?:相同|没有任何区别|無任何區別)|same rights and obligations', text, flags=re.I):
        rights_consistency = '劣后/优先权利义务可能无实质差异，需核验是否仅为跟投或心理安慰'
    elif is_structured:
        rights_consistency = '需核验各层级权利义务、收益分配和亏损吸收条款是否实质区分'

    flags = []
    if is_structured and not structure:
        flags.append('missing_tranche_terms')
    if is_structured and not cushion:
        flags.append('missing_junior_cushion')
    if is_structured and not warning_line:
        flags.append('missing_warning_line')
    if is_structured and not stop_loss_line:
        flags.append('missing_stop_loss_line')
    if is_structured and '较低' in liquidity_assessment:
        flags.append('low_liquidity_stop_loss')
    if rights_consistency.startswith('劣后/优先'):
        flags.append('fake_subordination')

    return {
        'is_structured_product': '是' if is_structured else '未识别/未披露',
        'tranche_structure': structure,
        'senior_tranche_terms': senior_terms,
        'mezzanine_tranche_terms': mezzanine_terms,
        'subordinated_tranche_terms': junior_terms,
        'waterfall_order': '优先 > 夹层/平层 > 劣后' if is_structured else '',
        'loss_absorption_order': '劣后 > 夹层/平层 > 优先' if is_structured else '',
        'junior_cushion_ratio': cushion,
        'warning_line': warning_line,
        'stop_loss_line': stop_loss_line,
        'underlying_liquidity_assessment': liquidity_assessment if is_structured else '',
        'rights_obligations_consistency': rights_consistency,
        'structured_product_risk_flags': '；'.join(flags),
    }


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
    classification_df: pd.DataFrame | None = None,
    fingerprint_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    for _, fund in base_df.iterrows():
        base_id = str(fund.get('base_fund_id', ''))
        docs = _docs_for(base_id, docs_df)
        combined_text = _combine_doc_text(docs)
        analysis = _analysis_row(base_id, official_analysis_df)
        classification_match = (
            classification_df[classification_df['base_fund_id'].astype(str).eq(base_id)]
            if classification_df is not None and not classification_df.empty and 'base_fund_id' in classification_df
            else pd.DataFrame()
        )
        classification = classification_match.iloc[0] if not classification_match.empty else None
        fingerprint_match = (
            fingerprint_df[fingerprint_df['base_fund_id'].astype(str).eq(base_id)]
            if fingerprint_df is not None and not fingerprint_df.empty and 'base_fund_id' in fingerprint_df
            else pd.DataFrame()
        )
        fingerprint = fingerprint_match.iloc[0] if not fingerprint_match.empty else None
        category = str(
            classification.get('category') if classification is not None
            else (analysis.get('category') if analysis is not None else '')
        ) or '其他'
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
        investment_strategy = _extract_investment_strategy(combined_text)
        hard_policy, target_ranges = _extract_policy_lines(combined_text)
        derivative_types, derivative_purpose, shorting, leverage, gross, net, market_exposure = _extract_derivative_fields(combined_text)
        asset_class = CATEGORY_ASSET_CLASS.get(category, '')
        structured_fields = _extract_structured_fields(combined_text, liquidity_terms, asset_class)

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
            'investment_strategy': investment_strategy,
            'return_mechanism': _extract_return_mechanism(combined_text),
            'hard_investment_policy': hard_policy,
            'target_exposure_range': target_ranges,
            'asset_class': asset_class,
            'fund_category': category,
            'classification_l1': classification.get('classification_l1', '') if classification is not None else '',
            'classification_l2': classification.get('classification_l2', '') if classification is not None else '',
            'classification_l3': classification.get('classification_l3', '') if classification is not None else '',
            'classification_confidence': classification.get('classification_confidence', '') if classification is not None else '',
            'classification_route': classification.get('routing_decision', '') if classification is not None else '',
            'policy_fingerprint_hash': fingerprint.get('fingerprint_hash', '') if fingerprint is not None else '',
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
            'derivative_types': derivative_types,
            'derivative_purpose': derivative_purpose,
            'short_selling_policy': shorting,
            'leverage_role': leverage,
            'gross_exposure': gross,
            'net_exposure': net,
            'market_exposure': market_exposure,
            'max_drawdown': max_drawdown,
            'drawdown_source': drawdown_source,
            'liquidity_terms': liquidity_terms,
            'dealing_frequency': dealing_frequency,
            'settlement_period': settlement_period,
            'redemption_liquidity': redemption_liquidity,
            **structured_fields,
            'evidence_links': '；'.join(dict.fromkeys(str(x) for x in docs.get('link', pd.Series(dtype=str)).dropna().head(5))),
        }
        row['master_data_quality_score'] = _quality_score(row)
        row['missing_master_fields'] = '；'.join(field for field in REQUIRED_MASTER_FIELDS if not str(row.get(field, '') or '').strip())
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=MASTER_DATA_COLUMNS)
    return pd.DataFrame(rows)[MASTER_DATA_COLUMNS]
