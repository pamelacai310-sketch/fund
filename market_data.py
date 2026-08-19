from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd
import requests


SINA_NAV_URL = 'https://stock.finance.sina.com.cn/fundInfo/api/openapi.php/CaihuiFundInfoService.getNav'
MARKET_DATA_VERSION = 'fund-market-data-loader-free-v1.0.0'
WINDOW_TOLERANCE_MONTHS = 0.25

SERIES_MASTER_COLUMNS = [
    'series_id', 'product_code', 'base_fund_id', 'base_fund_name', 'fund_company',
    'fund_name_cn', 'fund_name_en', 'isin', 'reported_currency', 'is_hedged',
    'distribution_type', 'history_start', 'history_end', 'history_months',
    'observation_count', 'return_series_field', 'return_reconstruction_method',
    'source_name', 'source_url', 'source_quality', 'series_status',
]

OBSERVATION_COLUMNS = [
    'series_id', 'product_code', 'base_fund_id', 'date', 'unit_nav',
    'cumulative_nav', 'adjusted_nav', 'reported_currency', 'source_name',
    'source_url', 'retrieved_at',
]

PROVENANCE_COLUMNS = [
    'series_id', 'product_code', 'source_name', 'source_url', 'retrieved_at',
    'content_sha256', 'endpoint_validation_status', 'record_count',
    'history_start', 'history_end', 'cache_status', 'source_note',
]

COVERAGE_COLUMNS = [
    'series_id', 'product_code', 'base_fund_id', 'base_fund_name',
    'observation_count', 'history_start', 'history_end', 'history_months',
    'eligible_12m', 'eligible_36m', 'eligible_60m', 'coverage_status',
]

REVIEW_COLUMNS = [
    'entity_type', 'entity_id', 'base_fund_id', 'issue_type', 'severity',
    'issue_detail', 'recommended_action',
]


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _clean_product_code(value: str) -> str:
    return str(value or '').upper().replace('.OF', '').strip()


def _share_features(row: pd.Series) -> tuple[str, bool, str]:
    text = ' '.join(str(row.get(col, '') or '') for col in ['fund_name_cn', 'fund_name_en']).lower()
    currency = ''
    for code, terms in {
        'CNY': ['cny', 'rmb', '人民币'],
        'USD': ['usd', '美元'],
        'HKD': ['hkd', '港元'],
    }.items():
        if any(term in text for term in terms):
            currency = code
            break
    is_hedged = any(term in text for term in ['hedged', 'hdg', '对冲', '(hedged)'])
    if any(term in text for term in ['mdist', ' dist', '-dist', ' dis.', '派息', '分派']):
        distribution = 'distribution'
    elif any(term in text for term in [' acc', '-acc', 'acc.', '累计', '累积']):
        distribution = 'accumulation'
    else:
        distribution = 'unspecified'
    return currency, is_hedged, distribution


def _prepared_url(params: dict[str, Any]) -> str:
    return requests.Request('GET', SINA_NAV_URL, params=params).prepare().url or SINA_NAV_URL


def _request_json(params: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    response = requests.get(
        SINA_NAV_URL,
        params=params,
        timeout=timeout,
        headers={'User-Agent': 'Mozilla/5.0 fund-research-audit/1.0'},
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError('Sina NAV response is not a JSON object')
    return payload


def _extract_payload_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = payload.get('result', {})
    status = result.get('status', {}) if isinstance(result, dict) else {}
    if status.get('code') not in {0, '0'}:
        raise ValueError(f"Sina NAV status code is not zero: {status.get('code')}")
    data = result.get('data', {}) if isinstance(result, dict) else {}
    rows = data.get('data', []) if isinstance(data, dict) else []
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise ValueError('Sina NAV data field is not a list')
    for row in rows[:5]:
        if not isinstance(row, dict) or not {'fbrq', 'jjjz'}.issubset(row):
            raise ValueError('Sina NAV row schema changed')
    return rows


def validate_sina_endpoint(real_symbol: str = '968040', nonexistent_symbol: str = '000000') -> dict[str, str]:
    params = {'datefrom': '', 'dateto': '', 'page': 1, 'num': 5}
    real_rows = _extract_payload_rows(_request_json({**params, 'symbol': real_symbol}))
    nonexistent_rows = _extract_payload_rows(_request_json({**params, 'symbol': nonexistent_symbol}))
    if not real_rows:
        raise ValueError(f'Endpoint validation failed: real symbol {real_symbol} returned no rows')
    if nonexistent_rows:
        raise ValueError(f'Endpoint validation failed: nonexistent symbol {nonexistent_symbol} returned rows')
    return {
        'status': 'validated',
        'real_symbol': real_symbol,
        'nonexistent_symbol': nonexistent_symbol,
        'validated_at': datetime.now(timezone.utc).isoformat(),
    }


def _cache_path(cache_dir: Path, symbol: str, start: str, end: str) -> Path:
    cache_key = sha256(f'{MARKET_DATA_VERSION}|{symbol}|{start}|{end}'.encode('utf-8')).hexdigest()[:16]
    return cache_dir / f'sina_nav_{symbol}_{cache_key}.json'


def _load_symbol(
    symbol: str,
    start: str,
    end: str,
    cache_dir: Path,
    force_refresh: bool,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, symbol, start, end)
    cache_status = 'miss'
    if path.exists() and not force_refresh:
        payload = json.loads(path.read_text(encoding='utf-8'))
        cache_status = 'hit'
    else:
        params = {'symbol': symbol, 'datefrom': start, 'dateto': end, 'page': 1, 'num': 2000}
        payload = _request_json(params)
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding='utf-8')
    rows = _extract_payload_rows(payload)
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    params = {'symbol': symbol, 'datefrom': start, 'dateto': end, 'page': 1, 'num': 2000}
    return {
        'symbol': symbol,
        'rows': rows,
        'sha256': sha256(content.encode('utf-8')).hexdigest(),
        'cache_status': cache_status,
        'source_url': _prepared_url(params),
        'retrieved_at': datetime.now(timezone.utc).isoformat(),
    }


def _to_float(value: Any) -> float | None:
    try:
        parsed = float(str(value).replace(',', '').strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _normalise_nav_rows(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in result['rows']:
        nav_date = pd.to_datetime(item.get('fbrq'), errors='coerce')
        unit_nav = _to_float(item.get('jjjz'))
        cumulative_nav = _to_float(item.get('ljjz'))
        adjusted_nav = cumulative_nav or unit_nav
        if pd.isna(nav_date) or adjusted_nav is None:
            continue
        rows.append({
            'date': nav_date.normalize(),
            'unit_nav': unit_nav,
            'cumulative_nav': cumulative_nav,
            'adjusted_nav': adjusted_nav,
        })
    if not rows:
        return pd.DataFrame(columns=['date', 'unit_nav', 'cumulative_nav', 'adjusted_nav'])
    return (
        pd.DataFrame(rows)
        .sort_values('date')
        .drop_duplicates(subset=['date'], keep='last')
        .reset_index(drop=True)
    )


def _history_months(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return max(0.0, (end - start).days / 30.4375)


def build_market_data_outputs(
    share_df: pd.DataFrame,
    as_of_date: str | date | None = None,
    cache_dir: str | Path = '.cache/fund_market_data',
    lookback_years: int = 5,
    force_refresh: bool = False,
    max_workers: int = 8,
) -> dict[str, pd.DataFrame]:
    if share_df.empty:
        return {
            'series_master_df': _empty(SERIES_MASTER_COLUMNS),
            'observations_df': _empty(OBSERVATION_COLUMNS),
            'market_provenance_df': _empty(PROVENANCE_COLUMNS),
            'source_coverage_df': _empty(COVERAGE_COLUMNS),
            'market_review_df': _empty(REVIEW_COLUMNS),
            'market_audit_df': pd.DataFrame([{
                'metric': 'loader_status', 'value': 'no_input', 'note': MARKET_DATA_VERSION,
            }]),
        }

    as_of = pd.Timestamp(as_of_date or date.today()).normalize()
    start = as_of - pd.DateOffset(years=lookback_years)
    cache_path = Path(cache_dir)
    validation_symbol = next(
        (_clean_product_code(x) for x in share_df.get('product_code', pd.Series(dtype=str)) if _clean_product_code(x)),
        '968040',
    )
    validation = validate_sina_endpoint(validation_symbol)

    indexed_rows = [(index, row.copy()) for index, row in share_df.reset_index(drop=True).iterrows()]
    fetched: dict[int, dict[str, Any] | Exception] = {}

    def load(index: int, row: pd.Series) -> tuple[int, dict[str, Any]]:
        symbol = _clean_product_code(row.get('product_code', ''))
        if not re.fullmatch(r'\d{6}', symbol):
            raise ValueError(f'Unsupported product code for Sina NAV: {row.get("product_code", "")}')
        return index, _load_symbol(
            symbol,
            start.date().isoformat(),
            as_of.date().isoformat(),
            cache_path,
            force_refresh,
        )

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        future_map = {executor.submit(load, index, row): index for index, row in indexed_rows}
        for future in as_completed(future_map):
            index = future_map[future]
            try:
                _, fetched[index] = future.result()
            except Exception as exc:  # Each failed series remains auditable and does not abort other funds.
                fetched[index] = exc

    series_rows = []
    observation_frames = []
    provenance_rows = []
    coverage_rows = []
    review_rows = []

    for index, share in indexed_rows:
        product_code = str(share.get('product_code', '') or '')
        symbol = _clean_product_code(product_code)
        series_id = f'SHARE_{symbol}' if symbol else f'SHARE_ROW_{index + 1:04d}'
        currency, is_hedged, distribution = _share_features(share)
        result = fetched.get(index)
        if isinstance(result, Exception) or result is None:
            detail = str(result or 'unknown loader error')
            review_rows.append({
                'entity_type': 'share_series', 'entity_id': product_code,
                'base_fund_id': share.get('base_fund_id', ''), 'issue_type': 'market_data_fetch_failed',
                'severity': 'high', 'issue_detail': detail,
                'recommended_action': '核对产品代码并从基金公司官网补充净值序列。',
            })
            nav_df = pd.DataFrame(columns=['date', 'unit_nav', 'cumulative_nav', 'adjusted_nav'])
            source_url = SINA_NAV_URL
            retrieved_at = datetime.now(timezone.utc).isoformat()
            content_hash = ''
            cache_status = 'error'
        else:
            nav_df = _normalise_nav_rows(result)
            source_url = result['source_url']
            retrieved_at = result['retrieved_at']
            content_hash = result['sha256']
            cache_status = result['cache_status']

        if nav_df.empty:
            history_start = history_end = ''
            months = 0.0
            series_status = 'no_data'
            return_field = ''
            method = ''
            review_rows.append({
                'entity_type': 'share_series', 'entity_id': product_code,
                'base_fund_id': share.get('base_fund_id', ''), 'issue_type': 'no_market_data',
                'severity': 'high', 'issue_detail': '免费行情源未返回有效净值。',
                'recommended_action': '从基金公司官网或授权数据源补充历史净值。',
            })
        else:
            first_date = nav_df['date'].min()
            last_date = nav_df['date'].max()
            months = _history_months(first_date, last_date)
            history_start = first_date.date().isoformat()
            history_end = last_date.date().isoformat()
            has_cumulative = nav_df['cumulative_nav'].notna().any()
            return_field = 'cumulative_nav' if has_cumulative else 'unit_nav'
            method = 'vendor_cumulative_nav_proxy' if has_cumulative else 'unit_nav_price_return_only'
            series_status = 'usable' if months >= 12 - WINDOW_TOLERANCE_MONTHS else 'short_history'
            frame = nav_df.copy()
            frame.insert(0, 'base_fund_id', share.get('base_fund_id', ''))
            frame.insert(0, 'product_code', product_code)
            frame.insert(0, 'series_id', series_id)
            frame['reported_currency'] = currency
            frame['source_name'] = 'Sina Finance CaihuiFundInfoService'
            frame['source_url'] = source_url
            frame['retrieved_at'] = retrieved_at
            observation_frames.append(frame[OBSERVATION_COLUMNS])
            if months < 12 - WINDOW_TOLERANCE_MONTHS:
                review_rows.append({
                    'entity_type': 'share_series', 'entity_id': product_code,
                    'base_fund_id': share.get('base_fund_id', ''), 'issue_type': 'history_below_12_months',
                    'severity': 'medium', 'issue_detail': f'有效历史仅 {months:.1f} 个月。',
                    'recommended_action': '不生成数字评分，仅保留结构和策略比较。',
                })
            if method == 'unit_nav_price_return_only' and distribution == 'distribution':
                review_rows.append({
                    'entity_type': 'share_series', 'entity_id': product_code,
                    'base_fund_id': share.get('base_fund_id', ''), 'issue_type': 'distribution_not_reconstructed',
                    'severity': 'high', 'issue_detail': '派息份额缺少累计净值，无法重建总回报。',
                    'recommended_action': '补充分派记录后再计算份额体验分。',
                })

        series_rows.append({
            'series_id': series_id,
            'product_code': product_code,
            'base_fund_id': share.get('base_fund_id', ''),
            'base_fund_name': share.get('base_fund_name', ''),
            'fund_company': share.get('fund_company', ''),
            'fund_name_cn': share.get('fund_name_cn', ''),
            'fund_name_en': share.get('fund_name_en', ''),
            'isin': share.get('isin', ''),
            'reported_currency': currency,
            'is_hedged': is_hedged,
            'distribution_type': distribution,
            'history_start': history_start,
            'history_end': history_end,
            'history_months': round(months, 2),
            'observation_count': len(nav_df),
            'return_series_field': return_field,
            'return_reconstruction_method': method,
            'source_name': 'Sina Finance CaihuiFundInfoService',
            'source_url': source_url,
            'source_quality': 'public_market_data_fallback',
            'series_status': series_status,
        })
        provenance_rows.append({
            'series_id': series_id,
            'product_code': product_code,
            'source_name': 'Sina Finance CaihuiFundInfoService',
            'source_url': source_url,
            'retrieved_at': retrieved_at,
            'content_sha256': content_hash,
            'endpoint_validation_status': validation['status'],
            'record_count': len(nav_df),
            'history_start': history_start,
            'history_end': history_end,
            'cache_status': cache_status,
            'source_note': '累计净值用于总回报代理；正式用途需与基金公司官方序列复核。',
        })
        coverage_rows.append({
            'series_id': series_id,
            'product_code': product_code,
            'base_fund_id': share.get('base_fund_id', ''),
            'base_fund_name': share.get('base_fund_name', ''),
            'observation_count': len(nav_df),
            'history_start': history_start,
            'history_end': history_end,
            'history_months': round(months, 2),
            'eligible_12m': months >= 12 - WINDOW_TOLERANCE_MONTHS,
            'eligible_36m': months >= 36 - WINDOW_TOLERANCE_MONTHS,
            'eligible_60m': months >= 60 - WINDOW_TOLERANCE_MONTHS,
            'coverage_status': '60m' if months >= 60 - WINDOW_TOLERANCE_MONTHS else ('36m' if months >= 36 - WINDOW_TOLERANCE_MONTHS else ('12m' if months >= 12 - WINDOW_TOLERANCE_MONTHS else 'insufficient')),
        })

    observations_df = (
        pd.concat(observation_frames, ignore_index=True)
        if observation_frames else _empty(OBSERVATION_COLUMNS)
    )
    audit_df = pd.DataFrame([
        {'metric': 'loader_version', 'value': MARKET_DATA_VERSION, 'note': '确定性免费行情加载器'},
        {'metric': 'as_of_date', 'value': as_of.date().isoformat(), 'note': '用户指定或运行日'},
        {'metric': 'endpoint_validation', 'value': validation['status'], 'note': f"real={validation['real_symbol']}; nonexistent={validation['nonexistent_symbol']}"},
        {'metric': 'input_share_count', 'value': len(share_df), 'note': '份额主数据行数'},
        {'metric': 'series_with_data', 'value': sum(int(row['observation_count'] > 0) for row in series_rows), 'note': '至少一条有效净值'},
        {'metric': 'series_eligible_12m', 'value': sum(int(row['history_months'] >= 12 - WINDOW_TOLERANCE_MONTHS) for row in series_rows), 'note': '允许0.25个月交易日容差'},
        {'metric': 'series_eligible_36m', 'value': sum(int(row['history_months'] >= 36 - WINDOW_TOLERANCE_MONTHS) for row in series_rows), 'note': '允许0.25个月交易日容差'},
        {'metric': 'llm_tokens_used', 'value': 0, 'note': '行情加载及计算不调用大模型'},
    ])
    return {
        'series_master_df': pd.DataFrame(series_rows, columns=SERIES_MASTER_COLUMNS),
        'observations_df': observations_df,
        'market_provenance_df': pd.DataFrame(provenance_rows, columns=PROVENANCE_COLUMNS),
        'source_coverage_df': pd.DataFrame(coverage_rows, columns=COVERAGE_COLUMNS),
        'market_review_df': pd.DataFrame(review_rows, columns=REVIEW_COLUMNS),
        'market_audit_df': audit_df,
    }
