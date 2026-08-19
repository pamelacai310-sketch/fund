from __future__ import annotations

import pandas as pd


TRANCHE_ANALYSIS_COLUMNS = [
    'base_fund_id', 'base_fund_name', 'fund_company', 'is_structured_product',
    'tranche_structure', 'senior_tranche_terms', 'mezzanine_tranche_terms',
    'subordinated_tranche_terms', 'waterfall_order', 'loss_absorption_order',
    'investor_choice_check', 'analysis_conclusion',
]

CASHFLOW_CUSHION_COLUMNS = [
    'base_fund_id', 'base_fund_name', 'waterfall_order', 'loss_absorption_order',
    'junior_cushion_ratio', 'warning_line', 'stop_loss_line',
    'underlying_liquidity_assessment', 'senior_coverage_check',
    'stop_loss_executability', 'cashflow_cushion_conclusion',
]

STRUCTURED_RISK_AUDIT_COLUMNS = [
    'base_fund_id', 'base_fund_name', 'is_structured_product',
    'structured_product_risk_flags', 'priority_investor_fit',
    'mezzanine_investor_fit', 'junior_investor_fit',
    'required_due_diligence', 'audit_decision',
]


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _is_structured(value: str) -> bool:
    return str(value or '') == '是'


def _flag_set(value: str) -> set[str]:
    return {x.strip() for x in str(value or '').split('；') if x.strip()}


def _investor_choice_check(row: pd.Series) -> str:
    if not _is_structured(row.get('is_structured_product', '')):
        return '未识别结构化分层；按普通基金主数据继续分析。'
    checks = [
        '必须明确投资人认购的是优先、夹层/平层还是劣后份额',
        '必须核验不同层级收益分配、亏损吸收、退出和费用条款',
        '不得只按预期收益率判断风险收益',
    ]
    return '；'.join(checks)


def _analysis_conclusion(row: pd.Series) -> str:
    if not _is_structured(row.get('is_structured_product', '')):
        return '未识别优先/夹层/劣后条款；不能做结构化分层结论。'
    flags = _flag_set(row.get('structured_product_risk_flags', ''))
    if 'fake_subordination' in flags:
        return '劣后可能没有实质保障作用，需人工复核合同权利义务。'
    if 'low_liquidity_stop_loss' in flags:
        return '底层资产流动性可能不足，预警线/止损线的可执行性偏弱。'
    if flags:
        return '已识别结构化分层，但安全垫、预警线、止损线或层级条款披露不完整。'
    return '结构化条款、现金流顺序和安全垫字段相对完整，可进入进一步收益风险测算。'


def build_tranche_analysis(product_master_df: pd.DataFrame) -> pd.DataFrame:
    if product_master_df.empty:
        return _empty(TRANCHE_ANALYSIS_COLUMNS)
    rows = []
    for _, row in product_master_df.iterrows():
        rows.append({
            'base_fund_id': row.get('base_fund_id', ''),
            'base_fund_name': row.get('base_fund_name', ''),
            'fund_company': row.get('fund_company', ''),
            'is_structured_product': row.get('is_structured_product', ''),
            'tranche_structure': row.get('tranche_structure', ''),
            'senior_tranche_terms': row.get('senior_tranche_terms', ''),
            'mezzanine_tranche_terms': row.get('mezzanine_tranche_terms', ''),
            'subordinated_tranche_terms': row.get('subordinated_tranche_terms', ''),
            'waterfall_order': row.get('waterfall_order', ''),
            'loss_absorption_order': row.get('loss_absorption_order', ''),
            'investor_choice_check': _investor_choice_check(row),
            'analysis_conclusion': _analysis_conclusion(row),
        })
    return pd.DataFrame(rows)[TRANCHE_ANALYSIS_COLUMNS]


def _senior_coverage_check(row: pd.Series) -> str:
    if not _is_structured(row.get('is_structured_product', '')):
        return ''
    pieces = []
    if row.get('junior_cushion_ratio'):
        pieces.append(f"劣后安全垫: {row.get('junior_cushion_ratio')}")
    else:
        pieces.append('缺少劣后安全垫比例')
    if row.get('senior_tranche_terms'):
        pieces.append('已披露优先级条款')
    else:
        pieces.append('缺少优先级条款')
    return '；'.join(pieces)


def _stop_loss_executability(row: pd.Series) -> str:
    if not _is_structured(row.get('is_structured_product', '')):
        return ''
    flags = _flag_set(row.get('structured_product_risk_flags', ''))
    if 'low_liquidity_stop_loss' in flags:
        return '弱：底层资产低流动性可能导致止损线无法执行'
    if row.get('warning_line') and row.get('stop_loss_line') and row.get('underlying_liquidity_assessment'):
        return '中：已披露预警线/止损线和流动性线索，仍需核验交易执行'
    return '低置信：预警线、止损线或流动性披露不足'


def build_cashflow_cushion_analysis(product_master_df: pd.DataFrame) -> pd.DataFrame:
    if product_master_df.empty:
        return _empty(CASHFLOW_CUSHION_COLUMNS)
    rows = []
    for _, row in product_master_df.iterrows():
        is_structured = _is_structured(row.get('is_structured_product', ''))
        rows.append({
            'base_fund_id': row.get('base_fund_id', ''),
            'base_fund_name': row.get('base_fund_name', ''),
            'waterfall_order': row.get('waterfall_order', ''),
            'loss_absorption_order': row.get('loss_absorption_order', ''),
            'junior_cushion_ratio': row.get('junior_cushion_ratio', ''),
            'warning_line': row.get('warning_line', ''),
            'stop_loss_line': row.get('stop_loss_line', ''),
            'underlying_liquidity_assessment': row.get('underlying_liquidity_assessment', ''),
            'senior_coverage_check': _senior_coverage_check(row),
            'stop_loss_executability': _stop_loss_executability(row),
            'cashflow_cushion_conclusion': (
                '优先级卖点必须由底层现金流、劣后安全垫和止损可执行性共同支撑。'
                if is_structured else '未识别结构化现金流瀑布。'
            ),
        })
    return pd.DataFrame(rows)[CASHFLOW_CUSHION_COLUMNS]


def _audit_decision(row: pd.Series) -> str:
    if not _is_structured(row.get('is_structured_product', '')):
        return '不适用：未识别结构化分层条款。'
    flags = _flag_set(row.get('structured_product_risk_flags', ''))
    if 'fake_subordination' in flags:
        return '阻断：劣后可能无实质保障，不能作为优先级增信卖点。'
    if 'low_liquidity_stop_loss' in flags:
        return '限制：底层流动性不足时，必须弱化止损线和安全垫卖点。'
    if flags:
        return '需补证：分层、安全垫、预警线或止损线披露不完整。'
    return '可进入进一步测算：仍需结合历史回撤和底层现金流压力测试。'


def build_structured_risk_audit(product_master_df: pd.DataFrame) -> pd.DataFrame:
    if product_master_df.empty:
        return _empty(STRUCTURED_RISK_AUDIT_COLUMNS)
    rows = []
    for _, row in product_master_df.iterrows():
        rows.append({
            'base_fund_id': row.get('base_fund_id', ''),
            'base_fund_name': row.get('base_fund_name', ''),
            'is_structured_product': row.get('is_structured_product', ''),
            'structured_product_risk_flags': row.get('structured_product_risk_flags', ''),
            'priority_investor_fit': '风险厌恶型投资者仅适合在安全垫、止损线和底层流动性均可核验时考虑优先级。',
            'mezzanine_investor_fit': '夹层/平层需确认收益补偿是否覆盖次序靠后的风险。',
            'junior_investor_fit': '劣后级适合能承受杠杆放大亏损并理解剩余收益分配的人群。',
            'required_due_diligence': '合同分层条款；现金流分配顺序；亏损吸收顺序；劣后安全垫；预警/止损线；底层资产流动性；退出安排。',
            'audit_decision': _audit_decision(row),
        })
    return pd.DataFrame(rows)[STRUCTURED_RISK_AUDIT_COLUMNS]


def build_structured_product_outputs(product_master_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        'structured_tranche_df': build_tranche_analysis(product_master_df),
        'cashflow_cushion_df': build_cashflow_cushion_analysis(product_master_df),
        'structured_risk_audit_df': build_structured_risk_audit(product_master_df),
    }
