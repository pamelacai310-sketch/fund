from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd


SCORING_VERSION = 'fund-peer-score-v1.0.0'
WINDOW_TOLERANCE_MONTHS = 0.25

INVESTMENT_WEIGHTS = {
    'return_score': 0.30,
    'risk_score': 0.30,
    'risk_adjusted_score': 0.25,
    'consistency_score': 0.15,
}

SHARE_WEIGHTS = {
    'return_score': 0.50,
    'risk_score': 0.30,
    'consistency_score': 0.20,
}


@dataclass(frozen=True)
class PeerAssignment:
    peer_group_id: str
    peer_group_name: str
    peer_role: str
    rationale: str


def _text(row: pd.Series) -> str:
    return ' '.join(str(row.get(col, '') or '') for col in ['base_fund_name', 'fund_names_cn', 'fund_names_en']).lower()


def assign_peer_group(row: pd.Series) -> PeerAssignment:
    text = _text(row)
    if any(term in text for term in ['classic fund', 'hk-growth', 'hong kong prc', '香港基金']):
        return PeerAssignment('EQ_HK_ACTIVE', '香港主动股票', 'core', '同为香港主动股票；价值/成长风格作为差异化维度。')
    if 'asia equity high income' in text or '亚洲高息股票' in text:
        return PeerAssignment('EQ_ASIA_INCOME', '亚洲收益型股票', 'near', '收益增强机制可能包含期权覆盖，不与普通股息策略视为完全同类。')
    if any(term in text for term in ['high-dividend stocks', 'asian dividend', '高息股票基金', '亚洲股息']):
        return PeerAssignment('EQ_ASIA_INCOME', '亚洲收益型股票', 'core', '以亚洲股息或高息股票为主要收益来源。')
    if any(term in text for term in ['pacific securities', 'asia growth', 'asia pacific ex japan']):
        role = 'near' if any(term in text for term in ['volatility', '波幅']) else 'core'
        return PeerAssignment('EQ_ASIA_BROAD', '亚洲及亚太主动股票', role, '区域相近；低波动或风格约束作为同类层级调整。')
    if 'disruptive' in text or '创新动力' in text:
        return PeerAssignment('EQ_ASIA_THEMATIC', '亚洲主题成长股票', 'core', '颠覆式创新主题与宽基成长股票分池。')
    if 'global equity' in text or '环球股票' in text:
        return PeerAssignment('EQ_GLOBAL', '环球主动股票', 'core', '全球权益投资范围。')
    if 'asian total return bond' in text or '亚洲总收益债券' in text:
        return PeerAssignment('FI_ASIA_CORE', '亚洲综合债券', 'near', '灵活总回报、主动久期及汇率机制使其仅为近似同类。')
    if ('asian bond' in text or '亚洲债券' in text) and 'high yield' not in text and 'high income' not in text and '高收益' not in text and '高入息' not in text:
        return PeerAssignment('FI_ASIA_CORE', '亚洲综合债券', 'core', '标准亚洲债券策略。')
    if 'high income bond' in text or '高入息债券' in text:
        return PeerAssignment('FI_ASIA_HIGH_INCOME', '亚洲高入息债券', 'core', '高入息目标单列，避免与标准亚洲债券混排。')
    if 'high yield bond' in text or '高收益债券' in text:
        return PeerAssignment('FI_ASIA_HIGH_YIELD', '亚洲高收益债券', 'core', '明确非投资级或高收益信用暴露。')
    if 'global bond' in text or '环球债券' in text:
        return PeerAssignment('FI_GLOBAL', '环球综合债券', 'core', '全球债券投资范围。')
    if 'defensive balanced' in text or '平衡基金' in text:
        return PeerAssignment('MA_BALANCED', '多资产平衡', 'core', '同系列平衡配置，风险预算差异用于定位。')
    if 'portfolio - balanced' in text or '灵活配置基金' in text:
        return PeerAssignment('MA_BALANCED', '多资产平衡', 'core', '同系列平衡配置，风险预算差异用于定位。')
    if 'strategic income' in text or '策略收益' in text:
        return PeerAssignment('MA_INCOME', '多资产收益', 'extended', '总组合动态收益策略，仅与区域多资产收益作扩展诊断。')
    if 'multi-asset high income' in text or '多元资产入息' in text:
        return PeerAssignment('MA_INCOME', '多资产收益', 'extended', '区域多资产高入息策略，仅作扩展诊断。')
    return PeerAssignment('UNCLASSIFIED', '待分类', 'unclassified', '缺少足够策略事实。')


def build_peer_pool_definition(base_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, fund in base_df.iterrows():
        assignment = assign_peer_group(fund)
        rows.append({
            'base_fund_id': fund.get('base_fund_id', ''),
            'base_fund_name': fund.get('base_fund_name', ''),
            'peer_group_id': assignment.peer_group_id,
            'peer_group_name': assignment.peer_group_name,
            'peer_role': assignment.peer_role,
            'assignment_rationale': assignment.rationale,
            'main_ranking_eligible_role': assignment.peer_role == 'core',
            'peer_definition_version': SCORING_VERSION,
        })
    return pd.DataFrame(rows)


def _clean_series(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    work = frame[['date', 'adjusted_nav']].copy()
    work['date'] = pd.to_datetime(work['date'], errors='coerce')
    work['adjusted_nav'] = pd.to_numeric(work['adjusted_nav'], errors='coerce')
    work = work.dropna().query('adjusted_nav > 0').sort_values('date').drop_duplicates('date', keep='last')
    return work.set_index('date')['adjusted_nav'].astype(float)


def _window(values: pd.Series, months: int) -> pd.Series:
    if values.empty:
        return values
    cutoff = values.index.max() - pd.DateOffset(months=months)
    selected = values[values.index >= cutoff]
    if selected.empty or selected.index.min() > cutoff + pd.Timedelta(days=45):
        return pd.Series(dtype=float)
    return selected


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator is None or not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return None
    value = numerator / denominator
    if not np.isfinite(value):
        return None
    return float(np.clip(value, -20, 20))


def _period_metrics(values: pd.Series, risk_free_rate: float) -> dict[str, float | None]:
    if len(values) < 2:
        return {}
    days = (values.index[-1] - values.index[0]).days
    if days <= 0 or values.iloc[0] <= 0:
        return {}
    annual_return = float((values.iloc[-1] / values.iloc[0]) ** (365.25 / days) - 1)
    monthly_values = values.resample('ME').last().dropna()
    monthly_returns = monthly_values.pct_change().dropna()
    if len(monthly_returns) < 3:
        return {'annualized_return': annual_return}
    annualized_volatility = float(monthly_returns.std(ddof=1) * math.sqrt(12))
    downside = np.minimum(monthly_returns.to_numpy(dtype=float), 0.0)
    downside_volatility = float(math.sqrt(np.mean(np.square(downside))) * math.sqrt(12))
    drawdown = values / values.cummax() - 1
    max_drawdown = float(drawdown.min())
    worst_month = float(monthly_returns.min())
    rolling_returns = monthly_values.pct_change(12).dropna()
    positive_rate = float((rolling_returns > 0).mean()) if len(rolling_returns) else None
    return {
        'annualized_return': annual_return,
        'annualized_volatility': annualized_volatility,
        'downside_volatility': downside_volatility,
        'max_drawdown': max_drawdown,
        'worst_month': worst_month,
        'sharpe': _safe_ratio(annual_return - risk_free_rate, annualized_volatility),
        'sortino': _safe_ratio(annual_return - risk_free_rate, downside_volatility),
        'calmar': _safe_ratio(annual_return, abs(max_drawdown)),
        'positive_rolling_12m_rate': positive_rate,
    }


def _series_metrics(series_row: pd.Series, observations_df: pd.DataFrame, risk_free_rate: float) -> dict[str, Any]:
    series_id = str(series_row.get('series_id', ''))
    frame = observations_df[observations_df['series_id'].astype(str).eq(series_id)] if not observations_df.empty else pd.DataFrame()
    values = _clean_series(frame)
    if values.empty:
        return {
            'series_id': series_id,
            'history_months': 0.0,
            'primary_window_months': 0,
            'metric_status': 'no_data',
        }
    history_months = (values.index.max() - values.index.min()).days / 30.4375
    periods = {}
    for months in [12, 36, 60]:
        period_values = _window(values, months) if history_months >= months - WINDOW_TOLERANCE_MONTHS else pd.Series(dtype=float)
        periods[months] = _period_metrics(period_values, risk_free_rate) if not period_values.empty else {}
    primary = 36 if periods[36].get('annualized_return') is not None else (12 if periods[12].get('annualized_return') is not None else 0)
    selected = periods.get(primary, {})
    result = {
        'series_id': series_id,
        'history_start': values.index.min().date().isoformat(),
        'history_end': values.index.max().date().isoformat(),
        'history_months': round(history_months, 2),
        'observation_count': len(values),
        'primary_window_months': primary,
        'metric_status': 'eligible' if primary else 'insufficient_history',
        'annualized_return': selected.get('annualized_return'),
        'annualized_volatility': selected.get('annualized_volatility'),
        'downside_volatility': selected.get('downside_volatility'),
        'max_drawdown': selected.get('max_drawdown'),
        'worst_month': selected.get('worst_month'),
        'sharpe': selected.get('sharpe'),
        'sortino': selected.get('sortino'),
        'calmar': selected.get('calmar'),
        'positive_rolling_12m_rate': selected.get('positive_rolling_12m_rate'),
        'return_12m': periods[12].get('annualized_return'),
        'return_36m': periods[36].get('annualized_return'),
        'return_60m': periods[60].get('annualized_return'),
        'risk_free_rate_assumption': risk_free_rate,
        'risk_free_source': 'user_parameter_or_zero_default',
    }
    metric_fields = [
        'annualized_return', 'annualized_volatility', 'downside_volatility',
        'max_drawdown', 'worst_month', 'sharpe', 'sortino', 'calmar',
        'positive_rolling_12m_rate',
    ]
    result['metric_completeness'] = round(sum(pd.notna(result.get(field)) for field in metric_fields) / len(metric_fields), 4)
    return result


def build_all_series_metrics(
    series_master_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    rows = []
    for _, series in series_master_df.iterrows():
        metrics = _series_metrics(series, observations_df, risk_free_rate)
        rows.append({**series.to_dict(), **metrics})
    return pd.DataFrame(rows)


def select_underlying_series(series_metrics_df: pd.DataFrame) -> pd.DataFrame:
    if series_metrics_df.empty:
        return series_metrics_df.copy()
    work = series_metrics_df.copy()
    work['_eligible_36m'] = pd.to_numeric(work['history_months'], errors='coerce').fillna(0).ge(36 - WINDOW_TOLERANCE_MONTHS).astype(int)
    work['_accumulation'] = work['distribution_type'].eq('accumulation').astype(int)
    work['_unhedged'] = (~work['is_hedged'].fillna(False).astype(bool)).astype(int)
    work = work.sort_values(
        ['base_fund_id', '_eligible_36m', '_accumulation', '_unhedged', 'history_months', 'observation_count'],
        ascending=[True, False, False, False, False, False],
    )
    selected = work.groupby('base_fund_id', as_index=False).head(1).copy()
    selected['underlying_series_role'] = 'canonical_cny_share_proxy'
    selected['underlying_proxy_warning'] = selected.apply(
        lambda row: '人民币对冲份额代理，含对冲成本与份额费率影响。' if bool(row.get('is_hedged'))
        else '人民币份额代理，仍可能包含汇率与份额费率影响。',
        axis=1,
    )
    return selected.drop(columns=['_eligible_36m', '_accumulation', '_unhedged'])


def _percentile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors='coerce')
    ranked = numeric.rank(method='average', pct=True, ascending=higher_is_better)
    return ranked * 100


def _component_scores(population: pd.DataFrame) -> pd.DataFrame:
    scored = population.copy()
    scored['return_score'] = _percentile(scored['annualized_return'], True)
    risk_parts = pd.concat([
        _percentile(scored['annualized_volatility'], False),
        _percentile(scored['downside_volatility'], False),
        _percentile(scored['max_drawdown'].abs(), False),
        _percentile(scored['worst_month'].abs(), False),
    ], axis=1)
    scored['risk_score'] = risk_parts.mean(axis=1, skipna=True)
    adjusted_parts = pd.concat([
        _percentile(scored['sharpe'], True),
        _percentile(scored['sortino'], True),
        _percentile(scored['calmar'], True),
    ], axis=1)
    scored['risk_adjusted_score'] = adjusted_parts.mean(axis=1, skipna=True)
    scored['consistency_score'] = _percentile(scored['positive_rolling_12m_rate'], True)
    scored['investment_quality_score'] = sum(
        scored[field].fillna(0) * weight for field, weight in INVESTMENT_WEIGHTS.items()
    )
    return scored


def _broad_group(classification_path: Any) -> str:
    text = str(classification_path or '')
    return text.split('/')[0] if '/' in text else (text or 'Unclassified')


def _broad_group_from_peer(peer_group_id: Any, classification_path: Any) -> str:
    peer_id = str(peer_group_id or '')
    if peer_id.startswith('EQ_'):
        return 'Equity'
    if peer_id.startswith('FI_'):
        return 'FixedIncome'
    if peer_id.startswith('MA_'):
        return 'MultiAsset'
    return _broad_group(classification_path)


def build_investment_scores(
    underlying_metrics_df: pd.DataFrame,
    peer_pool_df: pd.DataFrame,
    classification_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = underlying_metrics_df.merge(peer_pool_df, on=['base_fund_id', 'base_fund_name'], how='left')
    class_cols = [col for col in ['base_fund_id', 'classification_path'] if col in classification_df]
    if len(class_cols) == 2:
        work = work.merge(classification_df[class_cols].drop_duplicates('base_fund_id'), on='base_fund_id', how='left')
    else:
        work['classification_path'] = ''
    work['broad_peer_group'] = work.apply(
        lambda row: _broad_group_from_peer(row.get('peer_group_id'), row.get('classification_path')),
        axis=1,
    )
    work['broad_diagnostic_score'] = np.nan
    work['broad_diagnostic_rank'] = np.nan
    for _, index in work[work['metric_status'].eq('eligible')].groupby('broad_peer_group').groups.items():
        if len(index) < 3:
            continue
        scored = _component_scores(work.loc[index])
        work.loc[index, 'broad_diagnostic_score'] = scored['investment_quality_score'].to_numpy()
        work.loc[index, 'broad_diagnostic_rank'] = scored['investment_quality_score'].rank(ascending=False, method='min').to_numpy()

    result_rows = []
    blocked_rows = []
    for _, candidate in work.iterrows():
        group_id = candidate.get('peer_group_id', 'UNCLASSIFIED')
        core = work[
            work['peer_group_id'].eq(group_id)
            & work['peer_role'].eq('core')
            & work['metric_status'].eq('eligible')
        ].copy()
        core_n = len(core)
        history = float(candidate.get('history_months') or 0)
        role = str(candidate.get('peer_role') or 'unclassified')
        blocked_reason = ''
        score_status = 'blocked'
        candidate_scores: dict[str, Any] = {}
        if candidate.get('metric_status') != 'eligible' or history < 12 - WINDOW_TOLERANCE_MONTHS:
            blocked_reason = '有效总回报历史少于12个月'
        elif role not in {'core', 'near'}:
            blocked_reason = '仅为Extended/未分类同类关系，不进入主排名'
        elif core_n < 3:
            blocked_reason = f'Core同类有效样本仅{core_n}只，少于临时评分门槛3只'
        else:
            population = core.copy()
            if str(candidate['base_fund_id']) not in set(population['base_fund_id'].astype(str)):
                population = pd.concat([population, candidate.to_frame().T], ignore_index=True)
            scored = _component_scores(population)
            matched = scored[scored['base_fund_id'].astype(str).eq(str(candidate['base_fund_id']))].iloc[0]
            candidate_scores = {field: matched.get(field) for field in [
                'return_score', 'risk_score', 'risk_adjusted_score',
                'consistency_score', 'investment_quality_score',
            ]}
            score_status = 'provisional'
            blocked_reason = '缺少统一可授权基准序列，按同类百分位发布临时评分'
        confidence = min(1.0, float(candidate.get('metric_completeness') or 0) * min(1.0, core_n / 5))
        row = {
            'base_fund_id': candidate.get('base_fund_id', ''),
            'base_fund_name': candidate.get('base_fund_name', ''),
            'canonical_product_code': candidate.get('product_code', ''),
            'canonical_series_id': candidate.get('series_id', ''),
            'peer_group_id': group_id,
            'peer_group_name': candidate.get('peer_group_name', ''),
            'peer_role': role,
            'core_peer_sample_size': core_n,
            'primary_window_months': candidate.get('primary_window_months', 0),
            'score_status': score_status,
            'investment_quality_score': candidate_scores.get('investment_quality_score'),
            'return_score': candidate_scores.get('return_score'),
            'risk_score': candidate_scores.get('risk_score'),
            'risk_adjusted_score': candidate_scores.get('risk_adjusted_score'),
            'consistency_score': candidate_scores.get('consistency_score'),
            'broad_peer_group': candidate.get('broad_peer_group', ''),
            'broad_diagnostic_score': candidate.get('broad_diagnostic_score'),
            'broad_diagnostic_rank': candidate.get('broad_diagnostic_rank'),
            'score_confidence': round(confidence, 4),
            'benchmark_available': False,
            'blocked_or_provisional_reason': blocked_reason,
            'underlying_proxy_warning': candidate.get('underlying_proxy_warning', ''),
            'scoring_version': SCORING_VERSION,
            'llm_tokens_used': 0,
        }
        result_rows.append(row)
        if score_status != 'formal':
            blocked_rows.append({
                'entity_type': 'underlying_fund',
                'entity_id': candidate.get('base_fund_id', ''),
                'entity_name': candidate.get('base_fund_name', ''),
                'score_name': 'InvestmentQualityScore',
                'status': score_status,
                'blocked_reason': blocked_reason,
                'required_action': '补充统一基准和Core同类序列；满36个月且同类不少于5只后升级正式评分。',
            })
    result = pd.DataFrame(result_rows)
    if not result.empty:
        result['peer_rank'] = result.groupby('peer_group_id')['investment_quality_score'].rank(ascending=False, method='min')
    return result, pd.DataFrame(blocked_rows)


def _parse_tokens(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return {str(item) for item in parsed if str(item)}
        if isinstance(parsed, dict):
            return {str(key) for key, item in parsed.items() if item}
    except (json.JSONDecodeError, TypeError):
        pass
    return {token.strip() for token in re.split(r'[；,|/]', text) if token.strip()}


def _rare_strategy_signals(row: pd.Series) -> tuple[list[str], str]:
    text = _text(row)
    official_text = ' '.join(str(row.get(col, '') or '') for col in [
        'investment_strategy', 'derivative_types', 'derivative_purpose',
        'return_mechanism', 'hard_investment_policy',
    ]).lower()
    signals = []
    if 'asia equity high income' in text or '亚洲高息股票' in text:
        signals.append('option_premium_overlay' if 'option' in official_text or '期权' in official_text else 'potential_option_income_overlay_unverified')
    if 'total return bond' in text or '总收益债券' in text:
        signals.append('flexible_total_return_bond')
    if 'strategic income' in text or '策略收益' in text:
        signals.append('dynamic_multi_asset_income')
    if 'disruptive' in text or '创新动力' in text:
        signals.append('disruptive_innovation_theme')
    if 'volatility' in text or '波幅' in text:
        signals.append('low_volatility_equity')
    if 'high yield' in text or '高收益债券' in text:
        signals.append('below_investment_grade_credit')
    if 'defensive balanced' in text or '防守' in text:
        signals.append('defensive_risk_budget')
    evidence = 'official_document' if official_text.strip() else ('fund_name_only' if signals else 'none')
    return signals, evidence


def build_differentiation_scores(
    underlying_metrics_df: pd.DataFrame,
    peer_pool_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    fingerprint_df: pd.DataFrame,
    product_master_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = underlying_metrics_df.merge(peer_pool_df, on=['base_fund_id', 'base_fund_name'], how='left')
    if not classification_df.empty:
        cols = [col for col in ['base_fund_id', 'classification_path'] if col in classification_df]
        if len(cols) == 2:
            work = work.merge(classification_df[cols].drop_duplicates('base_fund_id'), on='base_fund_id', how='left')
    if not fingerprint_df.empty:
        fp_cols = [
            'base_fund_id', 'return_drivers', 'primary_assets', 'strategy_mechanics',
            'geography', 'sector_theme', 'equity_style', 'credit_quality',
            'duration_band', 'derivative_role', 'leverage_role',
        ]
        work = work.merge(fingerprint_df[[col for col in fp_cols if col in fingerprint_df]].drop_duplicates('base_fund_id'), on='base_fund_id', how='left')
    if not product_master_df.empty:
        master_cols = [
            'base_fund_id', 'investment_strategy', 'derivative_types', 'derivative_purpose',
            'return_mechanism', 'hard_investment_policy', 'master_data_quality_score',
            'evidence_links',
        ]
        work = work.merge(product_master_df[[col for col in master_cols if col in product_master_df]].drop_duplicates('base_fund_id'), on='base_fund_id', how='left')
    work['broad_peer_group'] = work.apply(
        lambda row: _broad_group_from_peer(row.get('peer_group_id'), row.get('classification_path')),
        axis=1,
    )
    token_fields = [
        'return_drivers', 'primary_assets', 'strategy_mechanics', 'geography',
        'sector_theme', 'equity_style', 'credit_quality', 'duration_band',
        'derivative_role', 'leverage_role',
    ]
    rows = []
    blocked_rows = []
    for _, candidate in work.iterrows():
        population = work[work['broad_peer_group'].eq(candidate['broad_peer_group'])]
        eligible_population = population[population['metric_status'].eq('eligible')]
        group_n = len(population)
        candidate_tokens = {field: _parse_tokens(candidate.get(field)) for field in token_fields}
        rarity_parts = []
        observed_fields = 0
        for field, tokens in candidate_tokens.items():
            if not tokens:
                continue
            observed_fields += 1
            for token in tokens:
                frequency = sum(token in _parse_tokens(value) for value in population.get(field, pd.Series(dtype=str)))
                rarity_parts.append((1 - frequency / max(1, group_n)) * 100)
        structural_rarity = float(np.mean(rarity_parts)) if rarity_parts else None
        signals, evidence_strength = _rare_strategy_signals(candidate)
        evidence_multiplier = 1.0 if evidence_strength == 'official_document' else (0.4 if evidence_strength == 'fund_name_only' else 0.0)
        strategy_score = min(100.0, len(signals) * 35.0) * evidence_multiplier

        behavior_metrics = ['annualized_return', 'annualized_volatility', 'max_drawdown', 'worst_month']
        distances = []
        for field in behavior_metrics:
            values = pd.to_numeric(eligible_population.get(field, pd.Series(dtype=float)), errors='coerce').dropna()
            value = pd.to_numeric(pd.Series([candidate.get(field)]), errors='coerce').iloc[0]
            if len(values) < 3 or pd.isna(value):
                continue
            median = float(values.median())
            mad = float((values - median).abs().median())
            scale = max(mad * 1.4826, float(values.std(ddof=1)) * 0.25, 1e-9)
            distances.append(abs(float(value) - median) / scale)
        behavior_score = float(100 * (1 - math.exp(-np.mean(distances)))) if distances else None
        master_quality_raw = pd.to_numeric(pd.Series([candidate.get('master_data_quality_score')]), errors='coerce').iloc[0]
        master_quality = 0.0 if pd.isna(master_quality_raw) else min(1.0, float(master_quality_raw) / (100 if master_quality_raw > 1 else 1))
        structural_coverage = observed_fields / len(token_fields)
        performance_coverage = float(candidate.get('metric_completeness') or 0)
        evidence_completeness = 0.50 * structural_coverage + 0.30 * performance_coverage + 0.20 * master_quality
        raw_score = None
        adjusted_score = None
        status = 'blocked'
        reason = ''
        if group_n < 3:
            reason = f'宽口径同类样本仅{group_n}只，少于差异化评分门槛3只'
        elif candidate.get('metric_status') != 'eligible':
            reason = '有效总回报历史少于12个月，无法识别行为差异'
        elif structural_rarity is None:
            reason = '策略指纹字段不足，无法量化结构稀缺性'
        else:
            raw_score = (
                0.60 * structural_rarity
                + 0.25 * strategy_score
                + 0.15 * (behavior_score if behavior_score is not None else 0)
            )
            adjusted_score = raw_score * evidence_completeness
            status = 'provisional'
            reason = '持仓/暴露及官方策略证据不完整，差异化分数已应用完整度折扣'
        rows.append({
            'base_fund_id': candidate.get('base_fund_id', ''),
            'base_fund_name': candidate.get('base_fund_name', ''),
            'broad_peer_group': candidate.get('broad_peer_group', ''),
            'peer_sample_size': group_n,
            'score_status': status,
            'differentiation_score': adjusted_score,
            'raw_differentiation_score': raw_score,
            'structural_rarity_score': structural_rarity,
            'documented_strategy_score': strategy_score,
            'behavior_divergence_score': behavior_score,
            'evidence_completeness': round(evidence_completeness, 4),
            'rare_strategy_signals': '；'.join(signals),
            'strategy_evidence_strength': evidence_strength,
            'interpretation': '分数表示与同类的稀缺程度，不表示收益更高或风险更低。',
            'blocked_or_provisional_reason': reason,
            'evidence_links': candidate.get('evidence_links', ''),
            'scoring_version': SCORING_VERSION,
            'llm_tokens_used': 0,
        })
        if status != 'formal':
            blocked_rows.append({
                'entity_type': 'underlying_fund',
                'entity_id': candidate.get('base_fund_id', ''),
                'entity_name': candidate.get('base_fund_name', ''),
                'score_name': 'DifferentiationScore',
                'status': status,
                'blocked_reason': reason,
                'required_action': '补充最新持仓、资产/地区/行业暴露及官方策略机制证据。',
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        result['differentiation_rank'] = result.groupby('broad_peer_group')['differentiation_score'].rank(ascending=False, method='min')
    return result, pd.DataFrame(blocked_rows)


def build_share_experience_scores(
    all_series_metrics_df: pd.DataFrame,
    product_master_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = all_series_metrics_df.copy()
    if not product_master_df.empty:
        cols = [
            'base_fund_id', 'management_fee', 'ongoing_charges', 'subscription_fee',
            'redemption_fee', 'dealing_frequency', 'settlement_period',
            'redemption_liquidity',
        ]
        work = work.merge(product_master_df[[col for col in cols if col in product_master_df]].drop_duplicates('base_fund_id'), on='base_fund_id', how='left')
    rows = []
    blocked_rows = []
    for base_id, group in work.groupby('base_fund_id', dropna=False):
        eligible = group[group['metric_status'].eq('eligible')].copy()
        scored = None
        if len(eligible) >= 2:
            scored = eligible.copy()
            scored['return_score'] = _percentile(scored['annualized_return'], True)
            scored['risk_score'] = pd.concat([
                _percentile(scored['max_drawdown'].abs(), False),
                _percentile(scored['annualized_volatility'], False),
            ], axis=1).mean(axis=1, skipna=True)
            scored['consistency_score'] = _percentile(scored['positive_rolling_12m_rate'], True)
            scored['share_experience_score'] = sum(
                scored[field].fillna(0) * weight for field, weight in SHARE_WEIGHTS.items()
            )
        for _, share in group.iterrows():
            matched = (
                scored[scored['series_id'].astype(str).eq(str(share['series_id']))]
                if scored is not None else pd.DataFrame()
            )
            status = 'provisional' if not matched.empty else 'blocked'
            if share.get('metric_status') != 'eligible':
                reason = '有效总回报历史少于12个月'
            elif len(eligible) < 2:
                reason = f'同一底层基金可比份额仅{len(eligible)}个，无法计算相对份额体验'
            else:
                reason = '份额级官方费率和赎回流动性不完整，按总回报/风险/持续性发布临时评分'
            scored_row = matched.iloc[0] if not matched.empty else None
            rows.append({
                'base_fund_id': base_id,
                'base_fund_name': share.get('base_fund_name', ''),
                'series_id': share.get('series_id', ''),
                'product_code': share.get('product_code', ''),
                'fund_name_cn': share.get('fund_name_cn', ''),
                'reported_currency': share.get('reported_currency', ''),
                'is_hedged': share.get('is_hedged', False),
                'distribution_type': share.get('distribution_type', ''),
                'history_months': share.get('history_months', 0),
                'primary_window_months': share.get('primary_window_months', 0),
                'annualized_return': share.get('annualized_return'),
                'annualized_volatility': share.get('annualized_volatility'),
                'max_drawdown': share.get('max_drawdown'),
                'positive_rolling_12m_rate': share.get('positive_rolling_12m_rate'),
                'share_experience_score': scored_row.get('share_experience_score') if scored_row is not None else None,
                'return_score': scored_row.get('return_score') if scored_row is not None else None,
                'risk_score': scored_row.get('risk_score') if scored_row is not None else None,
                'consistency_score': scored_row.get('consistency_score') if scored_row is not None else None,
                'comparable_share_count': len(eligible),
                'score_status': status,
                'component_coverage': 0.75 if status == 'provisional' else 0.0,
                'management_fee': share.get('management_fee', ''),
                'ongoing_charges': share.get('ongoing_charges', ''),
                'subscription_fee': share.get('subscription_fee', ''),
                'redemption_fee': share.get('redemption_fee', ''),
                'dealing_frequency': share.get('dealing_frequency', ''),
                'settlement_period': share.get('settlement_period', ''),
                'redemption_liquidity': share.get('redemption_liquidity', ''),
                'blocked_or_provisional_reason': reason,
                'scoring_version': SCORING_VERSION,
                'llm_tokens_used': 0,
            })
            if status != 'formal':
                blocked_rows.append({
                    'entity_type': 'share_class',
                    'entity_id': share.get('product_code', ''),
                    'entity_name': share.get('fund_name_cn', ''),
                    'score_name': 'ShareExperienceScore',
                    'status': status,
                    'blocked_reason': reason,
                    'required_action': '补充份额级费率、分派、对冲及赎回条款后升级正式评分。',
                })
    result = pd.DataFrame(rows)
    if not result.empty:
        result['share_rank_within_fund'] = result.groupby('base_fund_id')['share_experience_score'].rank(ascending=False, method='min')
    return result, pd.DataFrame(blocked_rows)


def build_methodology_table(risk_free_rate: float) -> pd.DataFrame:
    return pd.DataFrame([
        {'section': '版本', 'item': 'scoring_version', 'value': SCORING_VERSION, 'note': '确定性计算，0 LLM token'},
        {'section': '底层主评分', 'item': 'InvestmentQualityScore', 'value': '收益30% + 风险30% + 风险调整25% + 持续性15%', 'note': '仅Core同类池进入主排名'},
        {'section': '差异化', 'item': 'DifferentiationScore', 'value': '结构稀缺60% + 官方策略25% + 行为差异15%', 'note': '乘以证据完整度；不等于投资质量'},
        {'section': '份额体验', 'item': 'ShareExperienceScore', 'value': '收益50% + 风险30% + 持续性20%', 'note': '费率/流动性缺失时仅发布临时评分'},
        {'section': '正式门槛', 'item': 'formal', 'value': '>=36个月 + 基准有效 + Core样本>=5', 'note': '当前免费数据缺少统一基准时不发布正式分'},
        {'section': '临时门槛', 'item': 'provisional', 'value': '>=12个月 + Core样本>=3', 'note': '缺少基准时醒目标记临时评分'},
        {'section': '受阻门槛', 'item': 'blocked', 'value': '<12个月或Core样本<3', 'note': '不以0分替代缺失分数'},
        {'section': '观察窗口', 'item': 'primary_window', 'value': '36个月；不足时12个月', 'note': '60个月仅作稳定性补充'},
        {'section': '总回报', 'item': 'return_series', 'value': '供应商累计净值代理', 'note': '派息份额禁止仅用单位净值排名'},
        {'section': '无风险利率', 'item': 'risk_free_rate', 'value': risk_free_rate, 'note': '命令行参数；默认0并保留来源标记'},
        {'section': '底层代理', 'item': 'canonical_series', 'value': '优先>=36个月、累积、非对冲、最长历史', 'note': '人民币份额仍含汇率、对冲成本和份额费率影响'},
        {'section': '宽口径诊断', 'item': 'broad_diagnostic_score', 'value': '按Equity/FixedIncome/MultiAsset比较', 'note': '仅用于雷达观察，不替代Core同类排名'},
    ])


def build_peer_score_outputs(
    base_df: pd.DataFrame,
    series_master_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    fingerprint_df: pd.DataFrame,
    product_master_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> dict[str, pd.DataFrame]:
    all_metrics = build_all_series_metrics(series_master_df, observations_df, risk_free_rate)
    underlying_metrics = select_underlying_series(all_metrics)
    peer_pool = build_peer_pool_definition(base_df)
    investment_scores, investment_blocked = build_investment_scores(underlying_metrics, peer_pool, classification_df)
    differentiation_scores, differentiation_blocked = build_differentiation_scores(
        underlying_metrics,
        peer_pool,
        classification_df,
        fingerprint_df,
        product_master_df,
    )
    share_scores, share_blocked = build_share_experience_scores(all_metrics, product_master_df)
    blocked_frames = [frame for frame in [investment_blocked, differentiation_blocked, share_blocked] if not frame.empty]
    blocked = pd.concat(blocked_frames, ignore_index=True) if blocked_frames else pd.DataFrame(columns=[
        'entity_type', 'entity_id', 'entity_name', 'score_name', 'status',
        'blocked_reason', 'required_action',
    ])
    ranking = investment_scores.merge(
        differentiation_scores[[
            'base_fund_id', 'differentiation_score', 'differentiation_rank',
            'score_status', 'rare_strategy_signals', 'evidence_completeness',
        ]].rename(columns={'score_status': 'differentiation_status'}),
        on='base_fund_id',
        how='left',
    ) if not investment_scores.empty else pd.DataFrame()
    return {
        'series_metrics_df': all_metrics,
        'underlying_metrics_df': underlying_metrics,
        'investment_scores_df': investment_scores,
        'differentiation_scores_df': differentiation_scores,
        'share_experience_df': share_scores,
        'peer_ranking_df': ranking,
        'peer_pool_definition_df': peer_pool,
        'scoring_blocked_df': blocked,
        'scoring_methodology_df': build_methodology_table(risk_free_rate),
    }
