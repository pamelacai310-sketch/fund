from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
import re

import pandas as pd


TAXONOMY_VERSION = 'fund_taxonomy_v1'
RULES_VERSION = 'classifier_rules_v1'
FACT_PROFILE_VERSION = 'classification_v1'
FINGERPRINT_VERSION = 'policy_fp_v1'
PEER_WEIGHTS_VERSION = 'peer_weights_v1'


@dataclass
class ClassificationFactPacket:
    underlying_fund_id: str
    as_of_date: str = ''
    primary_assets: set[str] = field(default_factory=set)
    policy_min_exposure: dict[str, float] = field(default_factory=dict)
    target_exposure: dict[str, tuple[float, float]] = field(default_factory=dict)
    actual_exposure: dict[str, float] = field(default_factory=dict)
    return_objective: str = ''
    strategy_codes: set[str] = field(default_factory=set)
    return_drivers: set[str] = field(default_factory=set)
    systematic: bool | None = None
    discretionary: bool | None = None
    derivative_types: set[str] = field(default_factory=set)
    derivative_purpose: str = ''
    short_selling: str = ''
    leverage_role: str = ''
    gross_exposure: float | None = None
    net_exposure: float | None = None
    beta_target: tuple[float, float] | None = None
    trend_following: bool | None = None
    carry_strategy: bool | None = None
    market_neutral: bool | None = None
    arbitrage_type: str = ''
    spread_structure: str = ''
    geography: set[str] = field(default_factory=set)
    sector_theme: set[str] = field(default_factory=set)
    equity_style: set[str] = field(default_factory=set)
    market_cap: set[str] = field(default_factory=set)
    credit_quality: set[str] = field(default_factory=set)
    duration_years: float | None = None
    currency_exposure: set[str] = field(default_factory=set)
    benchmark_role: str = ''
    active_passive: str = ''
    evidence_ids: dict[str, list[str]] = field(default_factory=dict)
    field_confidence: dict[str, float] = field(default_factory=dict)
    source_strength: dict[str, str] = field(default_factory=dict)
    completeness: float = 0.0
    raw_name_text: str = ''
    raw_evidence_text: str = ''


@dataclass
class ClassificationResult:
    l1: str
    l2: str
    l3: str
    scores: dict[str, float]
    confidence: float
    margin: float
    positive_hits: list[str]
    negative_hits: list[str]
    override_hits: list[str]
    conflicts: list[str]
    routing_decision: str


@dataclass(frozen=True)
class PolicyFingerprint:
    taxonomy_path: tuple[str, str, str]
    return_drivers: frozenset[str]
    primary_assets: frozenset[str]
    strategy_mechanics: frozenset[str]
    market_exposure: str
    geography: frozenset[str]
    sector_theme: frozenset[str]
    equity_style: frozenset[str]
    market_cap: frozenset[str]
    credit_quality: frozenset[str]
    duration_band: str
    implementation: str
    derivative_role: str
    short_selling_role: str
    leverage_role: str
    active_passive: str
    benchmark_role: str
    fingerprint_version: str = FINGERPRINT_VERSION


def _text_has(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def _join_docs(docs: pd.DataFrame) -> str:
    if docs.empty:
        return ''
    parts = []
    for _, row in docs.head(6).iterrows():
        parts.append(' '.join(str(row.get(col, '') or '') for col in ['title', 'snippet', 'text_excerpt']))
    return '\n'.join(parts)


def _parse_percent(value: str) -> float | None:
    match = re.search(r'(-?[0-9]+(?:\.[0-9]+)?)\s*%', str(value or ''))
    if not match:
        return None
    return float(match.group(1)) / 100


def _parse_number(value: str) -> float | None:
    match = re.search(r'-?[0-9]+(?:\.[0-9]+)?', str(value or ''))
    return float(match.group(0)) if match else None


def _extract_exposure(text: str, asset_patterns: dict[str, list[str]]) -> tuple[dict[str, float], dict[str, float]]:
    minimums: dict[str, float] = {}
    actual: dict[str, float] = {}
    for asset, words in asset_patterns.items():
        labels = '|'.join(words)
        policy_patterns = [
            rf'(?:at least|minimum(?: of)?|不少于|至少)\s*([0-9]+(?:\.[0-9]+)?)\s*%.{{0,60}}(?:{labels})',
            rf'(?:{labels}).{{0,60}}(?:at least|minimum(?: of)?|不少于|至少)\s*([0-9]+(?:\.[0-9]+)?)\s*%',
        ]
        for pattern in policy_patterns:
            match = re.search(pattern, text, flags=re.I | re.S)
            if match:
                minimums[asset] = float(match.group(1)) / 100
                break
        actual_patterns = [
            rf'(?:{labels})\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?)\s*%',
            rf'([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:{labels})',
        ]
        for pattern in actual_patterns:
            match = re.search(pattern, text, flags=re.I)
            if match:
                actual[asset] = float(match.group(1)) / 100
                break
    return minimums, actual


def _extract_band(text: str, label: str) -> tuple[float, float] | None:
    match = re.search(
        rf'(?:{label}).{{0,50}}?([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:-|–|—|to|至)\s*([0-9]+(?:\.[0-9]+)?)\s*%',
        text,
        flags=re.I | re.S,
    )
    return (float(match.group(1)) / 100, float(match.group(2)) / 100) if match else None


def _extract_named_number(text: str, labels: list[str]) -> float | None:
    label = '|'.join(labels)
    match = re.search(rf'(?:{label})\s*[:：]?\s*(-?[0-9]+(?:\.[0-9]+)?)\s*%?', text, flags=re.I)
    return float(match.group(1)) / 100 if match else None


def _set_tags(text: str, rules: dict[str, list[str]]) -> set[str]:
    return {tag for tag, patterns in rules.items() if _text_has(text, patterns)}


ASSET_PATTERNS = {
    'equity': [r'equit(?:y|ies)', r'stocks?', r'股票', r'证券基金'],
    'fixed_income': [r'fixed income', r'bonds?', r'debt securities', r'债券', r'債券'],
    'commodity': [r'commodit(?:y|ies)', r'商品'],
    'property': [r'real estate', r'property', r'REITs?', r'房地产', r'不动产'],
    'cash': [r'money market', r'cash', r'货币市场', r'现金'],
}

GEOGRAPHY_RULES = {
    'Asia': [r'\basia(?:n)?\b', r'亚洲', r'亞洲'],
    'AsiaPacific': [r'asia pacific', r'pacific securities', r'亚太', r'太平洋'],
    'HongKong': [r'hong kong', r'香港'],
    'GreaterChina': [r'greater china', r'大中华', r'大中華'],
    'China': [r'\bchina\b', r'中国', r'中國'],
    'Global': [r'\bglobal\b', r'worldwide', r'环球', r'全球'],
    'EmergingMarkets': [r'emerging markets?', r'新兴市场', r'新興市場'],
}

THEME_RULES = {
    'DividendIncome': [r'dividend', r'high[- ]?income equit', r'high[- ]?dividend', r'股息', r'高息股票', r'高股息'],
    'DisruptiveInnovation': [r'disruptive', r'innovation', r'创新', r'創新', r'颠覆'],
    'Technology': [r'technolog', r'科技', r'人工智能', r'\bAI\b'],
    'LowVolatility': [r'minimum volatility', r'low volatility', r'波幅基金', r'低波动'],
}

STYLE_RULES = {
    'Value': [r'\bvalue\b', r'价值', r'價值', r'classic fund'],
    'Growth': [r'\bgrowth\b', r'成长', r'增長'],
    'Income': [r'dividend', r'high income', r'收益', r'股息', r'高息'],
    'LowVolatility': [r'low volatility', r'minimum volatility', r'波幅', r'低波动'],
}


def build_classification_fact_packet(
    fund: pd.Series,
    docs: pd.DataFrame,
    master_row: pd.Series | None = None,
) -> ClassificationFactPacket:
    names = ' '.join(str(fund.get(col, '') or '') for col in ['base_fund_name', 'fund_names_cn', 'fund_names_en'])
    doc_text = _join_docs(docs)
    master_text = ''
    if master_row is not None:
        master_text = ' '.join(str(master_row.get(col, '') or '') for col in [
            'investment_objective', 'investment_strategy', 'hard_investment_policy',
            'target_exposure_range', 'asset_allocation', 'geographic_allocation',
            'sector_allocation', 'benchmark', 'duration', 'average_credit_rating',
            'derivative_purpose', 'derivative_types', 'short_selling_policy',
            'leverage_role', 'market_exposure',
        ])
    evidence = f'{doc_text}\n{master_text}'.strip()
    combined = f'{names}\n{evidence}'
    minimums, actual = _extract_exposure(evidence, ASSET_PATTERNS)
    primary_assets = set(minimums)
    primary_assets.update(asset for asset, value in actual.items() if value >= 0.5)
    if not primary_assets and evidence:
        primary_assets.update(asset for asset, patterns in ASSET_PATTERNS.items() if _text_has(evidence, patterns))

    target_exposure = {}
    for asset, patterns in ASSET_PATTERNS.items():
        band = _extract_band(evidence, '|'.join(patterns))
        if band:
            target_exposure[asset] = band

    derivative_types = _set_tags(evidence, {
        'futures': [r'futures?', r'期货', r'期貨'],
        'options': [r'options?', r'期权', r'期權'],
        'swaps': [r'swaps?', r'掉期'],
        'forwards': [r'forwards?', r'远期', r'遠期'],
    })
    hedging_only = _text_has(evidence, [r'(?:solely|only|primarily).{0,30}hedg', r'hedging purposes? only', r'仅用于.{0,12}对冲', r'只用于.{0,12}套期'])
    return_core = _text_has(evidence, [r'derivatives?.{0,50}(?:core|principal|return generation)', r'(?:core|principal).{0,50}derivatives?', r'衍生品.{0,30}(?:核心|收益来源)'])
    derivative_purpose = 'hedging_only' if hedging_only else ('return_generation_core' if return_core else ('unspecified' if derivative_types else ''))

    explicit_short = _text_has(evidence, [r'short sell', r'short positions?', r'long\s*/?\s*short', r'卖空', r'做空', r'多空'])
    short_hedge = _text_has(evidence, [r'short.{0,30}hedg', r'卖空.{0,20}对冲'])
    short_selling = 'hedging_only' if short_hedge else ('return_generation' if explicit_short else '')
    market_neutral = True if _text_has(evidence, [r'market[- ]neutral', r'beta[- ]neutral', r'dollar[- ]neutral', r'市场中性', r'貝塔中性']) else None
    trend_following = True if _text_has(evidence, [r'trend following', r'time[- ]series momentum', r'systematic trend', r'趋势跟踪', r'趨勢跟蹤']) else None
    systematic = True if _text_has(evidence, [r'systematic', r'rules?[- ]based', r'quantitative', r'系统化', r'量化']) else None
    discretionary = True if _text_has(evidence, [r'discretionary', r'主观策略', r'主動判斷']) else None
    carry_strategy = True if _text_has(evidence, [r'carry strategy', r'carry return', r'套息', r'持有收益']) else None

    calendar_arb = _text_has(evidence, [r'calendar spread', r'inter[- ]month spread', r'calendar arbitrage', r'跨期套利', r'月间价差'])
    relative_value = _text_has(evidence, [r'relative value', r'statistical arbitrage', r'fixed income arbitrage', r'相对价值', r'套利'])
    arbitrage_type = 'calendar_spread' if calendar_arb else ('relative_value' if relative_value else '')
    spread_structure = 'same_underlying_different_maturities' if calendar_arb and _text_has(evidence, [r'same underlying', r'near.{0,20}far', r'不同到期', r'近月.{0,20}远月']) else ''

    beta_low = re.search(r'(?:beta target|target beta|贝塔目标)\s*[:：]?\s*(-?0(?:\.\d+)?)', evidence, flags=re.I)
    beta_target = (-0.1, 0.1) if beta_low else None
    gross_exposure = _extract_named_number(evidence, ['gross exposure', '总敞口', '總敞口'])
    net_exposure = _extract_named_number(evidence, ['net exposure', '净敞口', '淨敞口'])

    strategy_codes = set()
    for code, patterns in {
        'strategic_allocation': [r'strategic asset allocation', r'战略资产配置', r'策略性资产配置'],
        'tactical_allocation': [r'tactical asset allocation', r'战术资产配置', r'灵活配置'],
        'risk_managed': [r'risk managed', r'risk budget', r'volatility target', r'风险预算', r'目标波动'],
        'long_short': [r'long\s*/?\s*short', r'多空策略'],
        'multi_strategy': [r'multi[- ]strategy', r'多策略'],
        'index': [r'track(?:s|ing)?\s+(?:the performance of|an? index)', r'passive(?:ly)? managed', r'追踪.{0,20}指数', r'被动管理'],
        'enhanced_index': [r'enhanced index', r'指数增强'],
    }.items():
        if _text_has(evidence, patterns):
            strategy_codes.add(code)

    return_drivers = set()
    for driver, patterns in {
        'equity_beta': [r'equity returns?', r'capital growth.{0,40}equit', r'股票增值'],
        'credit_income': [r'credit spread', r'bond income', r'interest income', r'票息', r'信用利差'],
        'asset_allocation': [r'asset allocation', r'资产配置', r'資產配置'],
        'trend': [r'trend following', r'systematic trend', r'趋势跟踪'],
        'relative_value_spread': [r'relative value', r'calendar spread', r'套利', r'价差'],
        'dividend_income': [r'dividend income', r'high dividend', r'股息', r'高息股票'],
    }.items():
        if _text_has(evidence, patterns):
            return_drivers.add(driver)

    duration = _parse_number(master_row.get('duration', '')) if master_row is not None else None
    credit = _set_tags(combined, {
        'HighYield': [r'high yield', r'below investment grade', r'高收益债', r'非投资级'],
        'InvestmentGrade': [r'investment grade', r'投资级'],
    })
    currencies = _set_tags(combined, {
        'CNY': [r'\bCNY\b', r'\bRMB\b', r'人民币'],
        'USD': [r'\bUSD\b', r'美元'],
        'HKD': [r'\bHKD\b', r'港元'],
    })
    benchmark_text = str(master_row.get('benchmark', '') or '') if master_row is not None else ''
    active_passive = 'passive' if 'index' in strategy_codes else ('active' if _text_has(evidence, [r'actively managed', r'主动管理', r'主動管理']) else '')
    benchmark_role = 'target_or_tracking' if active_passive == 'passive' else ('comparison' if benchmark_text else '')
    leverage_role = 'strategy_core' if _text_has(evidence, [r'leverag(?:e|ed).{0,30}(?:strategy|return)', r'杠杆.{0,20}(?:策略|收益)']) else ('permitted_or_unspecified' if _text_has(evidence, [r'leverag', r'杠杆']) else '')

    latest_dates = []
    if not docs.empty and 'document_date' in docs:
        latest_dates = sorted([str(x) for x in docs['document_date'].dropna() if str(x)], reverse=True)
    evidence_links = list(dict.fromkeys(str(x) for x in docs.get('link', pd.Series(dtype=str)).dropna())) if not docs.empty else []
    evidence_quality = 0.85 if evidence else 0.20
    source = 'official_disclosure' if evidence else 'fund_name'
    critical_values = [primary_assets, minimums or actual, return_drivers, strategy_codes, derivative_purpose, _set_tags(combined, GEOGRAPHY_RULES), benchmark_role, duration, credit]
    completeness = sum(bool(value) for value in critical_values) / len(critical_values)

    return ClassificationFactPacket(
        underlying_fund_id=str(fund.get('base_fund_id', '')),
        as_of_date=latest_dates[0] if latest_dates else '',
        primary_assets=primary_assets,
        policy_min_exposure=minimums,
        target_exposure=target_exposure,
        actual_exposure=actual,
        return_objective=str(master_row.get('investment_objective', '') or '') if master_row is not None else '',
        strategy_codes=strategy_codes,
        return_drivers=return_drivers,
        systematic=systematic,
        discretionary=discretionary,
        derivative_types=derivative_types,
        derivative_purpose=derivative_purpose,
        short_selling=short_selling,
        leverage_role=leverage_role,
        gross_exposure=gross_exposure,
        net_exposure=net_exposure,
        beta_target=beta_target,
        trend_following=trend_following,
        carry_strategy=carry_strategy,
        market_neutral=market_neutral,
        arbitrage_type=arbitrage_type,
        spread_structure=spread_structure,
        geography=_set_tags(combined, GEOGRAPHY_RULES),
        sector_theme=_set_tags(combined, THEME_RULES),
        equity_style=_set_tags(combined, STYLE_RULES),
        market_cap=_set_tags(combined, {'LargeCap': [r'large[- ]cap', r'大盘'], 'SmallMidCap': [r'small.{0,5}mid[- ]cap', r'中小盘']}),
        credit_quality=credit,
        duration_years=duration,
        currency_exposure=currencies,
        benchmark_role=benchmark_role,
        active_passive=active_passive,
        evidence_ids={'official_documents': evidence_links},
        field_confidence={'classification_evidence': evidence_quality},
        source_strength={'classification_evidence': source},
        completeness=round(completeness, 4),
        raw_name_text=names,
        raw_evidence_text=evidence,
    )


def _path(l1: str, l2: str, l3: str) -> str:
    return '/'.join([l1, l2, l3])


def _classify(packet: ClassificationFactPacket) -> ClassificationResult:
    scores: dict[str, float] = {}
    positives: list[str] = []
    negatives: list[str] = []
    overrides: list[str] = []
    conflicts: list[str] = []
    evidence_quality = packet.field_confidence.get('classification_evidence', 0.2)
    doc = packet.raw_evidence_text
    name = packet.raw_name_text

    def add(path: str, weight: float, rule_id: str, quality: float = evidence_quality, negative: bool = False) -> None:
        scores[path] = scores.get(path, 0.0) + weight * quality
        (negatives if negative else positives).append(rule_id)

    explicit_cta = packet.trend_following is True and 'futures' in packet.derivative_types and packet.derivative_purpose == 'return_generation_core'
    explicit_neutral = packet.market_neutral is True and (packet.beta_target is not None or packet.net_exposure is not None or packet.short_selling == 'return_generation')
    explicit_calendar = packet.arbitrage_type == 'calendar_spread' and packet.spread_structure == 'same_underlying_different_maturities'

    cta_path = _path('Alternative', 'CTA', 'SystematicTrend')
    neutral_path = _path('Alternative', 'MarketNeutral', 'EquityMarketNeutral')
    calendar_path = _path('Alternative', 'RelativeValue', 'CalendarSpread')
    if packet.trend_following:
        add(cta_path, 6, 'cta_trend_following')
    if 'futures' in packet.derivative_types and packet.derivative_purpose == 'return_generation_core':
        add(cta_path, 5, 'cta_futures_core')
    if packet.systematic and packet.trend_following:
        add(cta_path, 4, 'cta_systematic')
    if packet.derivative_purpose == 'hedging_only':
        add(cta_path, -9, 'cta_derivatives_hedging_only', negative=True)
    elif packet.derivative_types and not packet.trend_following:
        negatives.append('cta_derivatives_without_trend_insufficient')
    if explicit_cta:
        add(cta_path, 10, 'cta_return_driver_override', quality=1.0)
        overrides.append('override_cta_core_mechanism')

    if packet.market_neutral:
        add(neutral_path, 9, 'market_neutral_explicit')
    if packet.beta_target:
        add(neutral_path, 7, 'market_neutral_low_beta')
    if packet.short_selling == 'return_generation':
        add(neutral_path, 3, 'market_neutral_short_book')
    if packet.short_selling == 'hedging_only':
        add(neutral_path, -6, 'market_neutral_shorting_hedge_only', negative=True)
    if explicit_neutral:
        add(neutral_path, 10, 'market_neutral_override', quality=1.0)
        overrides.append('override_market_neutral_mechanism')

    if packet.arbitrage_type == 'calendar_spread':
        add(calendar_path, 10, 'calendar_spread_explicit')
    if packet.spread_structure:
        add(calendar_path, 7, 'calendar_same_underlying_maturities')
    if explicit_calendar:
        add(calendar_path, 10, 'calendar_spread_override', quality=1.0)
        overrides.append('override_calendar_spread_mechanism')

    high_yield = _text_has(name, [r'high yield bond', r'高收益债'])
    fixed_path = _path('FixedIncome', 'HighYield' if high_yield else 'Aggregate', 'BelowInvestmentGrade' if high_yield else 'CoreBond')
    equity_subtype = 'Index' if packet.active_passive == 'passive' else ('NontraditionalEquity' if 'long_short' in packet.strategy_codes else 'LongOnly')
    name_themes = _set_tags(name, THEME_RULES)
    equity_l3 = 'DividendIncome' if 'DividendIncome' in name_themes else ('LowVolatility' if 'LowVolatility' in name_themes else ('LongShortNetLong' if 'long_short' in packet.strategy_codes else 'BroadMarket'))
    equity_path = _path('Equity', equity_subtype, equity_l3)
    multi_subtype = 'RiskManaged' if 'risk_managed' in packet.strategy_codes else ('FlexibleAllocation' if 'tactical_allocation' in packet.strategy_codes else ('Income' if _text_has(f'{name} {doc}', [r'strategic income', r'multi.asset high income', r'策略收益', r'多元资产入息']) else 'StrategicAllocation'))
    multi_path = _path('MultiAsset', multi_subtype, 'CrossAsset')

    if packet.policy_min_exposure.get('fixed_income', 0) >= 0.8:
        add(fixed_path, 8, 'bond_hard_policy_80')
    if packet.actual_exposure.get('fixed_income', 0) >= 0.8:
        add(fixed_path, 7, 'bond_actual_exposure_80')
    if 'fixed_income' in packet.primary_assets:
        add(fixed_path, 5, 'bond_principal_asset')
    if packet.duration_years is not None or packet.credit_quality:
        add(fixed_path, 4, 'bond_duration_or_credit')
    if _text_has(name, [r'bond', r'债券', r'債券']):
        add(fixed_path, 8, 'explicit_bond_product_name', quality=1.0)

    if packet.policy_min_exposure.get('equity', 0) >= 0.8:
        add(equity_path, 8, 'equity_hard_policy_80')
    if packet.actual_exposure.get('equity', 0) >= 0.85:
        add(equity_path, 7, 'equity_actual_exposure_85')
    if 'equity' in packet.primary_assets:
        add(equity_path, 5, 'equity_principal_asset')
    if _text_has(name, [r'equit', r'stock', r'securities fund', r'股票', r'证券基金']):
        add(equity_path, 8, 'explicit_equity_product_name', quality=1.0)
    elif _text_has(name, [r'growth fund', r'hong kong fund', r'dividend', r'classic fund', r'增長基金', r'香港基金', r'股息基金', r'价值基金']):
        add(equity_path, 5, 'equity_product_family_name', quality=1.0)
    if 'long_short' in packet.strategy_codes and not explicit_neutral:
        add(equity_path, 5, 'equity_long_short_net_long')
        negatives.append('market_neutral_not_implied_by_long_short')

    major_assets = {x for x in packet.primary_assets if x in {'equity', 'fixed_income', 'commodity', 'property', 'cash'}}
    targeted_assets = {x for x, band in packet.target_exposure.items() if band[1] > 0}
    if len(major_assets) >= 2:
        add(multi_path, 8, 'multi_asset_principal_assets')
    if len(targeted_assets) >= 2:
        add(multi_path, 6, 'multi_asset_target_ranges')
    if packet.strategy_codes.intersection({'strategic_allocation', 'tactical_allocation'}):
        add(multi_path, 5, 'multi_asset_allocation_strategy')
    if _text_has(name, [r'balanced', r'multi[- ]asset', r'portfolio', r'strategic income', r'平衡基金', r'灵活配置', r'多元资产', r'策略收益']):
        add(multi_path, 8, 'explicit_multi_asset_product_name', quality=1.0)
    if 'multi_strategy' in packet.strategy_codes and len(packet.return_drivers) < 2:
        negatives.append('multi_strategy_label_without_multiple_return_mechanisms')
    if len(major_assets) >= 2 and 'multi_strategy' not in packet.strategy_codes:
        negatives.append('multi_asset_not_alternative_multi_strategy')

    money_path = _path('MoneyMarket', 'ShortTerm', 'Liquidity')
    if 'cash' in packet.primary_assets and _text_has(f'{name} {doc}', [r'money market', r'货币市场']):
        add(money_path, 8, 'money_market_principal_strategy')
    commodity_path = _path('Commodity', 'LongOnly', 'BroadCommodity')
    if 'commodity' in packet.primary_assets and not explicit_cta:
        add(commodity_path, 5, 'commodity_principal_asset')

    if explicit_cta or explicit_neutral or explicit_calendar:
        for path in list(scores):
            if path.startswith(('Equity/', 'FixedIncome/', 'MultiAsset/', 'Commodity/')):
                scores[path] -= 10
                negatives.append(f'alternative_override_penalty:{path}')
    if packet.market_neutral and not explicit_neutral:
        conflicts.append('market_neutral_claim_missing_beta_net_or_short_mechanism')
    if packet.trend_following and packet.derivative_purpose != 'return_generation_core':
        conflicts.append('trend_signal_missing_core_derivative_purpose')

    if not scores:
        scores[_path('Other', 'Unclassified', 'InsufficientEvidence')] = 0.0
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    winner, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = max(0.0, min(1.0, (top_score - second_score) / max(abs(top_score), 1.0)))
    score_strength = max(0.0, min(1.0, top_score / 10))
    critical_coverage = _critical_coverage(packet, winner)
    contradiction_penalty = min(1.0, len(conflicts) / 3)
    confidence = 0.35 * score_strength + 0.25 * margin + 0.20 * critical_coverage + 0.15 * evidence_quality - 0.05 * contradiction_penalty
    confidence = round(max(0.0, min(1.0, confidence)), 4)
    margin = round(margin, 4)
    routing = _route(confidence, margin, critical_coverage, conflicts)
    l1, l2, l3 = winner.split('/')
    return ClassificationResult(l1, l2, l3, dict(ranked), confidence, margin, positives, negatives, overrides, conflicts, routing)


def _critical_coverage(packet: ClassificationFactPacket, path: str) -> float:
    if path.startswith('Alternative/CTA'):
        values = [packet.trend_following, packet.systematic, packet.derivative_types, packet.derivative_purpose, packet.return_drivers]
    elif path.startswith('Alternative/MarketNeutral'):
        values = [packet.market_neutral, packet.beta_target or packet.net_exposure, packet.short_selling, packet.return_drivers]
    elif path.startswith('FixedIncome'):
        values = [packet.primary_assets, packet.policy_min_exposure or packet.actual_exposure, packet.credit_quality, packet.duration_years]
    elif path.startswith('Equity'):
        values = [packet.primary_assets, packet.policy_min_exposure or packet.actual_exposure, packet.geography, packet.equity_style or packet.sector_theme]
    elif path.startswith('MultiAsset'):
        values = [packet.primary_assets, packet.target_exposure, packet.strategy_codes, packet.return_drivers]
    else:
        values = [packet.primary_assets, packet.return_drivers, packet.strategy_codes]
    return sum(bool(value) for value in values) / len(values)


def _route(confidence: float, margin: float, completeness: float, conflicts: list[str]) -> str:
    if confidence >= 0.90 and margin >= 0.25 and completeness >= 0.50 and not conflicts:
        return 'auto_accept'
    if confidence >= 0.60:
        return 'fetch_more_evidence'
    return 'manual_review'


def _duration_band(value: float | None) -> str:
    if value is None:
        return ''
    if value < 3:
        return 'Short'
    if value <= 7:
        return 'Intermediate'
    return 'Long'


def build_policy_fingerprint(packet: ClassificationFactPacket, result: ClassificationResult) -> PolicyFingerprint:
    market_exposure = 'neutral' if result.l2 == 'MarketNeutral' else ('net_long' if 'long_short' in packet.strategy_codes else ('long_only' if result.l1 in {'Equity', 'FixedIncome', 'Commodity'} else ''))
    implementation = 'systematic' if packet.systematic else ('discretionary' if packet.discretionary else '')
    return PolicyFingerprint(
        taxonomy_path=(result.l1, result.l2, result.l3),
        return_drivers=frozenset(packet.return_drivers),
        primary_assets=frozenset(packet.primary_assets),
        strategy_mechanics=frozenset(packet.strategy_codes),
        market_exposure=market_exposure,
        geography=frozenset(packet.geography),
        sector_theme=frozenset(packet.sector_theme),
        equity_style=frozenset(packet.equity_style),
        market_cap=frozenset(packet.market_cap),
        credit_quality=frozenset(packet.credit_quality),
        duration_band=_duration_band(packet.duration_years),
        implementation=implementation,
        derivative_role=packet.derivative_purpose,
        short_selling_role=packet.short_selling,
        leverage_role=packet.leverage_role,
        active_passive=packet.active_passive,
        benchmark_role=packet.benchmark_role,
    )


PEER_WEIGHTS = {
    'Equity': {'l2': .20, 'l3': .10, 'geography': .20, 'equity_style': .18, 'sector_theme': .10, 'market_cap': .10, 'active_passive': .06, 'market_exposure': .06},
    'FixedIncome': {'l2': .15, 'l3': .10, 'geography': .15, 'credit_quality': .22, 'duration_band': .16, 'primary_assets': .07, 'derivative_role': .05, 'market_exposure': .10},
    'MultiAsset': {'l2': .15, 'l3': .10, 'geography': .10, 'primary_assets': .20, 'strategy_mechanics': .20, 'return_drivers': .15, 'implementation': .05, 'derivative_role': .05},
    'Alternative': {'l2': .25, 'l3': .25, 'return_drivers': .15, 'strategy_mechanics': .10, 'market_exposure': .15, 'implementation': .06, 'derivative_role': .04},
    'default': {'l2': .30, 'l3': .20, 'primary_assets': .20, 'geography': .15, 'return_drivers': .15},
}


def _hard_incompatibility(left: PolicyFingerprint, right: PolicyFingerprint) -> str:
    if left.taxonomy_path[0] != right.taxonomy_path[0]:
        return 'different_l1_asset_or_return_driver'
    pairs = {left.taxonomy_path[:2], right.taxonomy_path[:2]}
    incompatible = [
        {('Equity', 'LongOnly'), ('Equity', 'NontraditionalEquity')},
        {('Alternative', 'CTA'), ('Alternative', 'MarketNeutral')},
        {('Alternative', 'CTA'), ('Alternative', 'RelativeValue')},
    ]
    if any(pairs == pair for pair in incompatible):
        return 'incompatible_strategy_mechanism'
    return ''


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _field_value(fp: PolicyFingerprint, field_name: str):
    if field_name == 'l2':
        return fp.taxonomy_path[1]
    if field_name == 'l3':
        return fp.taxonomy_path[2]
    return getattr(fp, field_name)


def _score_peer(left: PolicyFingerprint, right: PolicyFingerprint) -> dict:
    incompatibility = _hard_incompatibility(left, right)
    if incompatibility:
        return {'similarity': 0.0, 'coverage': 1.0, 'tier': 'not_peer', 'hard_incompatibility': incompatibility, 'component_scores': {}}
    weights = PEER_WEIGHTS.get(left.taxonomy_path[0], PEER_WEIGHTS['default'])
    weighted_score = 0.0
    observed_weight = 0.0
    components = {}
    for field_name, weight in weights.items():
        a = _field_value(left, field_name)
        b = _field_value(right, field_name)
        if not a or not b:
            continue
        score = _jaccard(a, b) if isinstance(a, frozenset) else (1.0 if a == b else 0.0)
        components[field_name] = round(score, 4)
        weighted_score += weight * score
        observed_weight += weight
    if observed_weight == 0:
        return {'similarity': 0.0, 'coverage': 0.0, 'tier': 'insufficient_data', 'hard_incompatibility': '', 'component_scores': {}}
    raw = weighted_score / observed_weight
    coverage = observed_weight / sum(weights.values())
    final = raw * math.sqrt(coverage)
    if final >= .85 and coverage >= .70:
        tier = 'core'
    elif final >= .70:
        tier = 'near'
    elif final >= .55:
        tier = 'extended'
    else:
        tier = 'not_peer'
    return {'similarity': round(final, 4), 'coverage': round(coverage, 4), 'tier': tier, 'hard_incompatibility': '', 'component_scores': components}


def _jsonable(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, tuple):
        return [_jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _json(value) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _fingerprint_hash(fp: PolicyFingerprint) -> str:
    canonical = _json(asdict(fp))
    return sha256(canonical.encode('utf-8')).hexdigest()


def _legacy_category(packet: ClassificationFactPacket, result: ClassificationResult) -> str:
    name_geography = _set_tags(packet.raw_name_text, GEOGRAPHY_RULES)
    geography = name_geography or packet.geography
    name_themes = _set_tags(packet.raw_name_text, THEME_RULES)
    if result.l1 == 'FixedIncome':
        if result.l2 == 'HighYield' and 'Asia' in geography:
            return '亚洲高收益债'
        if 'Asia' in geography:
            return '亚洲债券'
        if 'Global' in geography:
            return '环球债券'
        return '债券'
    if result.l1 == 'Equity':
        if result.l3 == 'DividendIncome':
            return '亚洲股息/高息股票' if geography.intersection({'Asia', 'AsiaPacific'}) else '高息股票'
        if name_themes.intersection({'DisruptiveInnovation', 'Technology'}):
            return '主题成长股票'
        if 'Value' in packet.equity_style and geography.intersection({'China', 'GreaterChina', 'HongKong'}):
            return '大中华价值股票'
        if 'HongKong' in geography:
            return '香港股票'
        if geography.intersection({'Asia', 'AsiaPacific'}):
            return '亚洲/亚太股票'
        if 'Global' in geography:
            return '环球股票'
        return '股票'
    if result.l1 == 'MultiAsset':
        return '策略收益' if result.l2 == 'Income' else '多资产/平衡'
    if result.l1 == 'Alternative':
        return f'另类/{result.l2}'
    return '其他'


def build_classification_outputs(
    base_df: pd.DataFrame,
    docs_df: pd.DataFrame,
    product_master_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    packets: dict[str, ClassificationFactPacket] = {}
    results: dict[str, ClassificationResult] = {}
    fingerprints: dict[str, PolicyFingerprint] = {}
    names: dict[str, str] = {}
    fact_rows = []
    classification_rows = []
    fingerprint_rows = []

    for _, fund in base_df.iterrows():
        base_id = str(fund.get('base_fund_id', ''))
        names[base_id] = str(fund.get('base_fund_name', ''))
        docs = docs_df[docs_df['base_fund_id'].astype(str).eq(base_id)] if not docs_df.empty and 'base_fund_id' in docs_df else pd.DataFrame()
        master_match = product_master_df[product_master_df['base_fund_id'].astype(str).eq(base_id)] if not product_master_df.empty and 'base_fund_id' in product_master_df else pd.DataFrame()
        master_row = master_match.iloc[0] if not master_match.empty else None
        packet = build_classification_fact_packet(fund, docs, master_row)
        result = _classify(packet)
        fp = build_policy_fingerprint(packet, result)
        packets[base_id] = packet
        results[base_id] = result
        fingerprints[base_id] = fp

        fact_dict = asdict(packet)
        fact_rows.append({
            'base_fund_id': base_id,
            'base_fund_name': names[base_id],
            'as_of_date': packet.as_of_date,
            **{key: _json(value) if isinstance(value, (set, dict, tuple)) else value for key, value in fact_dict.items() if key not in {'underlying_fund_id', 'raw_name_text', 'raw_evidence_text'}},
            'fact_profile_version': FACT_PROFILE_VERSION,
        })
        classification_rows.append({
            'base_fund_id': base_id,
            'base_fund_name': names[base_id],
            'classification_l1': result.l1,
            'classification_l2': result.l2,
            'classification_l3': result.l3,
            'classification_path': _path(result.l1, result.l2, result.l3),
            'category': _legacy_category(packet, result),
            'classification_confidence': result.confidence,
            'classification_margin': result.margin,
            'routing_decision': result.routing_decision,
            'positive_rule_hits': '；'.join(result.positive_hits),
            'negative_rule_hits': '；'.join(result.negative_hits),
            'override_rule_hits': '；'.join(result.override_hits),
            'classification_conflicts': '；'.join(result.conflicts),
            'score_vector': _json(result.scores),
            'fact_completeness': packet.completeness,
            'classification_critical_coverage': round(_critical_coverage(packet, _path(result.l1, result.l2, result.l3)), 4),
            'taxonomy_version': TAXONOMY_VERSION,
            'rules_version': RULES_VERSION,
            'llm_tokens_used': 0,
        })
        fp_dict = asdict(fp)
        fingerprint_rows.append({
            'base_fund_id': base_id,
            'base_fund_name': names[base_id],
            **{key: _json(value) if isinstance(value, (set, frozenset, tuple)) else value for key, value in fp_dict.items()},
            'fingerprint_hash': _fingerprint_hash(fp),
        })

    peer_rows = []
    ids = list(fingerprints)
    for index, left_id in enumerate(ids):
        for right_id in ids[index + 1:]:
            score = _score_peer(fingerprints[left_id], fingerprints[right_id])
            confidence_gate = 'verified'
            if score['tier'] in {'core', 'near'} and (
                results[left_id].routing_decision != 'auto_accept'
                or results[right_id].routing_decision != 'auto_accept'
            ):
                score['tier'] = {'core': 'near', 'near': 'extended'}[score['tier']]
                confidence_gate = 'provisional_downgrade'
            peer_rows.append({
                'fund_a_id': left_id,
                'fund_a_name': names[left_id],
                'fund_b_id': right_id,
                'fund_b_name': names[right_id],
                'fund_a_classification': '/'.join(fingerprints[left_id].taxonomy_path),
                'fund_b_classification': '/'.join(fingerprints[right_id].taxonomy_path),
                'peer_similarity': score['similarity'],
                'peer_coverage': score['coverage'],
                'peer_tier': score['tier'],
                'peer_confidence_gate': confidence_gate,
                'hard_incompatibility': score['hard_incompatibility'],
                'component_scores': _json(score['component_scores']),
                'peer_weights_version': PEER_WEIGHTS_VERSION,
            })

    classification_df = pd.DataFrame(classification_rows)
    review_df = classification_df[classification_df['routing_decision'].ne('auto_accept')].copy() if not classification_df.empty else pd.DataFrame()
    if not review_df.empty:
        review_df['review_reason'] = review_df.apply(
            lambda row: '；'.join(filter(None, [
                '分类置信度未达到自动接受门槛' if float(row['classification_confidence']) < .90 else '',
                '第一、第二候选分离度不足' if float(row['classification_margin']) < .25 else '',
                '分类关键事实不完整' if float(row['classification_critical_coverage']) < .50 else '',
                str(row['classification_conflicts'] or ''),
            ])),
            axis=1,
        )
    return {
        'classification_fact_df': pd.DataFrame(fact_rows),
        'classification_df': classification_df,
        'fingerprint_df': pd.DataFrame(fingerprint_rows),
        'peer_edges_df': pd.DataFrame(peer_rows),
        'classification_review_df': review_df,
    }


def build_peer_positioning_from_edges(base_id: str, peer_edges_df: pd.DataFrame, limit: int = 8) -> str:
    if peer_edges_df.empty:
        return '列表内没有可计算的同类关系。建议补充官方策略、持仓和风险敞口资料。'
    matched = peer_edges_df[
        peer_edges_df['fund_a_id'].astype(str).eq(str(base_id))
        | peer_edges_df['fund_b_id'].astype(str).eq(str(base_id))
    ].copy()
    matched = matched[matched['peer_tier'].isin(['core', 'near', 'extended'])]
    if matched.empty:
        return '列表内没有通过策略指纹兼容性检查的同类基金。'
    matched = matched.sort_values(['peer_similarity', 'peer_coverage'], ascending=False)
    pieces = []
    for _, row in matched.head(limit).iterrows():
        is_left = str(row['fund_a_id']) == str(base_id)
        name = row['fund_b_name'] if is_left else row['fund_a_name']
        pieces.append(f"{row['peer_tier']}:{name}({float(row['peer_similarity']):.2f})")
    return f"策略指纹同类：{'；'.join(pieces)}。分层依据包含资产/收益机制、地区、风格、信用久期和衍生品用途。"
