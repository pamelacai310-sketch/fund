from __future__ import annotations

from collections import Counter
import re

import pandas as pd


MASTER_SKILLS = [
    {
        'skill_id': 'fund-mechanism-analyzer',
        'priority': 'P0',
        'objective': '穿透基金产品设计逻辑、收益来源、风险机制和适配痛点。',
        'required_inputs': '基金名称、官方月报/factsheet、持仓、派息、费用、份额条款',
        'outputs': '收益来源；风险来源；产品机制；隐藏风险；适配痛点',
    },
    {
        'skill_id': 'fund-peer-performance-underwriter',
        'priority': 'P0',
        'objective': '像投委会一样盘问基金：相对同类凭什么值得推荐。',
        'required_inputs': '历史净值、同类基金池、指数基准、费用、风险指标',
        'outputs': '历史业绩证据；风险收益指标；同类竞争力；反证风险；推荐置信度',
    },
    {
        'skill_id': 'fund-peer-comparator',
        'priority': 'P0',
        'objective': '构建同资产、同区域、同策略、同币种/对冲份额的同类基金对标。',
        'required_inputs': '基金分类、策略、区域、资产类别、份额币种、同类产品列表',
        'outputs': 'peer group；同类排名；优势/劣势；替代品；差异化定位',
    },
    {
        'skill_id': 'fund-market-narrative-radar',
        'priority': 'P0',
        'objective': '判断哪些产品特质可能成为下个月市场卖点爆点。',
        'required_inputs': '利率、汇率、估值、信用利差、资金流、社媒热词、渠道话术',
        'outputs': '下月主题候选；市场顺风程度；卖点爆点概率；需跟踪催化剂',
    },
    {
        'skill_id': 'fund-investor-pain-mapper',
        'priority': 'P0',
        'objective': '把投资者痛点映射到可验证的产品机制，而不是直接映射到营销词。',
        'required_inputs': '标准化评论、痛点标签、产品机制、官方资料证据',
        'outputs': '痛点；对应机制；解决力度；缺失证据；风险提示',
    },
    {
        'skill_id': 'fund-evidence-auditor',
        'priority': 'P0',
        'objective': '给每个结论附证据、反证和置信度，阻止伪卖点进入报告。',
        'required_inputs': '官方资料、业绩数据、同类对标、用户评论、市场数据',
        'outputs': '支持证据；反证风险；缺失数据；人工复核项；不可宣传结论',
    },
    {
        'skill_id': 'fund-cost-fee-analyzer',
        'priority': 'P1',
        'objective': '分析管理费、申赎费、派息成本、隐性成本和同类费用位置。',
        'required_inputs': '费率表、KFS、招募说明书、销售平台费率、同类基金费用',
        'outputs': '费用合理性；同类费用对标；成本拖累；需披露风险',
    },
    {
        'skill_id': 'fund-report-strategist',
        'priority': 'P1',
        'objective': '把专业投研结论转成业务可用但合规的产品定位和传播建议。',
        'required_inputs': '产品机制、业绩证据、同类对标、痛点映射、反证审计',
        'outputs': '产品定位；下月卖点；风险提示；渠道话术边界；不可宣传项',
    },
]


MECHANISM_RULES = {
    '亚洲债券': {
        'return_sources': '票息收入、亚洲信用利差、久期管理、币种与对冲管理',
        'risk_sources': '利率上行、信用利差走阔、区域信用事件、汇率或对冲成本',
        'suitable_pains': '稳健配置；降息受益；人民币对冲；现金流需求',
        'hidden_risks': '债券基金也可能回撤；高票息不等于低风险；久期和信用质量需核验',
    },
    '亚洲高收益债': {
        'return_sources': '高票息、信用利差收窄、亚洲高收益债价格修复',
        'risk_sources': '违约风险、流动性风险、信用利差急剧走阔、集中持仓',
        'suitable_pains': '高派息/现金流；收益增强',
        'hidden_risks': '高收益往往来自信用下沉；需要披露违约和净值波动风险',
    },
    '环球债券': {
        'return_sources': '全球利率周期、债券票息、久期配置、区域分散',
        'risk_sources': '全球利率波动、汇率、信用利差、对冲成本',
        'suitable_pains': '多资产分散；稳健配置；降息受益',
        'hidden_risks': '全球分散不能消除利率风险；需核验久期和币种暴露',
    },
    '亚洲股息/高息股票': {
        'return_sources': '股息收入、亚洲权益市场上涨、估值修复、行业配置',
        'risk_sources': '股票市场回撤、股息不可持续、行业集中、汇率波动',
        'suitable_pains': '高派息/现金流；亚洲增长；低估值修复',
        'hidden_risks': '派息不等于保本；权益型高息仍需承受股票波动',
    },
    '亚洲/亚太股票': {
        'return_sources': '亚洲企业盈利增长、区域市场估值修复、行业成长',
        'risk_sources': '区域政策风险、行业集中、汇率、市场 beta',
        'suitable_pains': '亚洲增长；低估值修复；品牌信任',
        'hidden_risks': '高成长叙事可能伴随高波动；需核验持仓集中度',
    },
    '香港股票': {
        'return_sources': '港股估值修复、分红、政策与流动性改善',
        'risk_sources': '港股 beta、政策预期落空、流动性、行业集中',
        'suitable_pains': '低估值修复；高派息/现金流；亚洲增长',
        'hidden_risks': '便宜不等于马上修复；需结合资金流和盈利周期',
    },
    '大中华价值股票': {
        'return_sources': '低估值修复、价值股分红、基本面改善',
        'risk_sources': '价值陷阱、持仓集中、市场风格不利、回撤',
        'suitable_pains': '低估值修复；高派息/现金流；品牌信任',
        'hidden_risks': '价值策略可能长期跑输成长风格；需核验估值和盈利证据',
    },
    '主题成长股票': {
        'return_sources': '创新主题成长、科技/行业景气、估值扩张',
        'risk_sources': '估值压缩、主题拥挤、换手和行业集中',
        'suitable_pains': '亚洲增长；主题机会；下月热点传播',
        'hidden_risks': '主题叙事易过热；不能用短期热点替代基本面',
    },
    '多资产/平衡': {
        'return_sources': '股债配置、资产再平衡、票息和分红、多资产分散',
        'risk_sources': '股债同跌、资产配置失误、费用拖累、派息不可持续',
        'suitable_pains': '稳健配置；多资产分散；现金流需求',
        'hidden_risks': '平衡不等于低风险；需要核验回撤控制和资产暴露',
    },
    '环球股票': {
        'return_sources': '全球权益增长、行业和区域分散、企业盈利',
        'risk_sources': '全球权益 beta、估值波动、汇率、行业集中',
        'suitable_pains': '多资产分散；全球配置；品牌信任',
        'hidden_risks': '全球配置仍有权益回撤；需核验区域和行业暴露',
    },
    '策略收益': {
        'return_sources': '多元收益资产、债券票息、信用、货币和灵活配置',
        'risk_sources': '策略黑箱、信用/利率/汇率叠加、衍生品或杠杆工具',
        'suitable_pains': '稳健配置；现金流需求；多资产分散',
        'hidden_risks': '策略灵活也可能降低透明度；需核验资产配置和风险预算',
    },
}

PAIN_TO_MECHANISM = {
    '亏损/回撤': '需要用最大回撤、波动率、下行捕获和回撤修复时间验证，而不是仅凭稳健话术。',
    '分红误解': '需要拆分派息来源、净值除息、是否动用资本和分红可持续性。',
    '净值波动': '需要核验资产波动、久期、信用利差、权益 beta 和对冲成本。',
    '汇率/人民币对冲': '需要核验 CNY/RMB hedged 份额、对冲成本、汇率暴露和历史拖累。',
    '高收益债信用风险': '需要核验信用评级、违约暴露、行业/地区集中和流动性。',
    '港股/亚洲市场波动': '需要核验市场 beta、估值、行业集中、政策敏感度和资金流。',
    '赎回/申购/费率': '需要核验销售平台、申赎规则、费用、转换限制和流动性安排。',
    '平台客服/信息披露': '需要核验月报及时性、KFS 信息完整度、销售材料一致性和客服反馈。',
}

NARRATIVE_RULES = {
    '降息受益': {
        'categories': {'亚洲债券', '环球债券', '策略收益', '多资产/平衡'},
        'trigger': '利率下行预期、债券久期收益、票息吸引力',
        'risk': '若利率反弹或信用利差走阔，债券价格可能回撤',
    },
    '高派息/现金流': {
        'categories': {'亚洲高收益债', '亚洲股息/高息股票', '香港股票', '大中华价值股票', '策略收益'},
        'trigger': '投资者现金流需求、红利资产关注度、派息份额传播性',
        'risk': '派息不代表收益，需说明除息、净值波动和资本分派风险',
    },
    '港股/亚洲低估值修复': {
        'categories': {'香港股票', '大中华价值股票', '亚洲/亚太股票', '亚洲股息/高息股票'},
        'trigger': '港股估值修复、政策预期、南向资金或风险偏好改善',
        'risk': '低估值可能持续，需警惕盈利下修和流动性不足',
    },
    '人民币对冲安心感': {
        'categories': {'亚洲债券', '环球债券', '亚洲/亚太股票', '多资产/平衡', '策略收益'},
        'trigger': '人民币汇率波动、投资者对外币波动的规避需求',
        'risk': '对冲成本可能侵蚀收益，且对冲不等于无汇率风险',
    },
    '多资产稳健配置': {
        'categories': {'多资产/平衡', '策略收益', '环球债券'},
        'trigger': '单一资产波动上升、投资者追求一站式配置',
        'risk': '股债同跌时多资产也可能回撤，需核验历史回撤控制',
    },
}


def build_master_skill_catalog() -> pd.DataFrame:
    return pd.DataFrame(MASTER_SKILLS)


def _category_for(row: pd.Series, official_analysis_df: pd.DataFrame) -> str:
    if not official_analysis_df.empty and 'base_fund_id' in official_analysis_df.columns:
        matched = official_analysis_df[official_analysis_df['base_fund_id'].astype(str).eq(str(row.get('base_fund_id', '')))]
        if not matched.empty:
            return str(matched.iloc[0].get('category') or '其他')
    return '其他'


def _docs_for(base_fund_id: str, docs_df: pd.DataFrame) -> pd.DataFrame:
    if docs_df.empty or 'base_fund_id' not in docs_df.columns:
        return pd.DataFrame()
    return docs_df[docs_df['base_fund_id'].astype(str).eq(str(base_fund_id))]


def _split_hit_labels(value: str) -> list[str]:
    labels = []
    for part in re.split(r'[；;|,]', str(value or '')):
        label = re.sub(r'\(\d+\)', '', part).strip()
        if label and label != '未识别出高频信号':
            labels.append(label)
    return labels


def _company_social_row(fund_company: str, social_analysis_df: pd.DataFrame) -> pd.Series | None:
    if social_analysis_df.empty or 'fund_company' not in social_analysis_df.columns:
        return None
    matched = social_analysis_df[social_analysis_df['fund_company'].astype(str).eq(str(fund_company))]
    if matched.empty:
        return None
    return matched.iloc[0]


def analyze_product_mechanisms(
    base_df: pd.DataFrame,
    official_analysis_df: pd.DataFrame,
    docs_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, fund in base_df.iterrows():
        category = _category_for(fund, official_analysis_df)
        rules = MECHANISM_RULES.get(category, {
            'return_sources': '待从月报、持仓和策略说明中确认',
            'risk_sources': '待从历史净值、持仓、费用和风险揭示中确认',
            'suitable_pains': '待结合产品机制和用户评论确认',
            'hidden_risks': '分类不明确，需人工复核产品定位',
        })
        docs = _docs_for(fund.get('base_fund_id', ''), docs_df)
        missing = ['历史净值', '费用明细', '完整持仓', '外部同类基金池', '派息历史', '市场指标']
        if not docs.empty:
            missing = [x for x in missing if x not in {'完整持仓'}]
        rows.append({
            'base_fund_id': fund.get('base_fund_id', ''),
            'base_fund_name': fund.get('base_fund_name', ''),
            'fund_company': fund.get('fund_company', ''),
            'category': category,
            'return_sources': rules['return_sources'],
            'risk_sources': rules['risk_sources'],
            'suitable_investor_pains': rules['suitable_pains'],
            'hidden_risks': rules['hidden_risks'],
            'available_doc_count': len(docs),
            'missing_evidence': '；'.join(missing),
            'mechanism_confidence': '中' if len(docs) >= 1 and category != '其他' else '低',
        })
    return pd.DataFrame(rows)


def build_pain_mechanism_map(
    base_df: pd.DataFrame,
    mechanism_df: pd.DataFrame,
    social_analysis_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, fund in base_df.iterrows():
        company = str(fund.get('fund_company', ''))
        social_row = _company_social_row(company, social_analysis_df)
        pains = []
        selling_points = []
        if social_row is not None:
            pains = _split_hit_labels(social_row.get('top_pain_aspects', ''))
            selling_points = _split_hit_labels(social_row.get('top_selling_aspects', ''))
        if not pains:
            pains = ['亏损/回撤', '净值波动']
        mechanism = mechanism_df[mechanism_df['base_fund_id'].astype(str).eq(str(fund.get('base_fund_id', '')))]
        mechanism_text = mechanism.iloc[0].get('suitable_investor_pains', '') if not mechanism.empty else ''
        for pain in pains:
            rows.append({
                'base_fund_id': fund.get('base_fund_id', ''),
                'base_fund_name': fund.get('base_fund_name', ''),
                'fund_company': company,
                'investor_pain': pain,
                'required_product_mechanism': PAIN_TO_MECHANISM.get(pain, '需要把痛点回连到可核验的产品机制。'),
                'current_product_fit': mechanism_text,
                'social_selling_signals': '；'.join(selling_points) if selling_points else '未识别出高频卖点信号',
                'fit_confidence': '中' if pain in mechanism_text else '低',
                'must_not_claim': '不得把痛点直接包装成卖点；必须同时提供历史业绩、同类对标和风险提示。',
            })
    return pd.DataFrame(rows)


def build_market_narrative_radar(
    base_df: pd.DataFrame,
    official_analysis_df: pd.DataFrame,
    social_analysis_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, fund in base_df.iterrows():
        category = _category_for(fund, official_analysis_df)
        company = str(fund.get('fund_company', ''))
        social_row = _company_social_row(company, social_analysis_df)
        selling_labels = _split_hit_labels(social_row.get('top_selling_aspects', '')) if social_row is not None else []
        for narrative, rule in NARRATIVE_RULES.items():
            if category not in rule['categories']:
                continue
            social_support = any(label in narrative or narrative in label for label in selling_labels)
            rows.append({
                'base_fund_id': fund.get('base_fund_id', ''),
                'base_fund_name': fund.get('base_fund_name', ''),
                'fund_company': company,
                'category': category,
                'next_month_narrative': narrative,
                'why_it_may_sell': rule['trigger'],
                'product_trait_to_use': MECHANISM_RULES.get(category, {}).get('suitable_pains', ''),
                'social_support': '有社媒卖点信号' if social_support else '缺少社媒高频信号',
                'evidence_needed_before_campaign': '最新月报；历史净值/回撤；同类排名；费用；市场指标',
                'risk_disclosure': rule['risk'],
                'burst_confidence': '中' if social_support else '低',
            })
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=[
        'base_fund_id', 'base_fund_name', 'fund_company', 'category', 'next_month_narrative',
        'why_it_may_sell', 'product_trait_to_use', 'social_support',
        'evidence_needed_before_campaign', 'risk_disclosure', 'burst_confidence',
    ])


def build_evidence_audit(
    base_df: pd.DataFrame,
    official_analysis_df: pd.DataFrame,
    docs_df: pd.DataFrame,
    social_analysis_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, fund in base_df.iterrows():
        base_id = str(fund.get('base_fund_id', ''))
        docs = _docs_for(base_id, docs_df)
        category = _category_for(fund, official_analysis_df)
        company = str(fund.get('fund_company', ''))
        social_row = _company_social_row(company, social_analysis_df)
        evidence = []
        missing = ['历史净值', '最大回撤/波动率', '费用明细', '完整持仓', '外部同类基金池', '派息历史', '下月市场指标']
        if not docs.empty:
            evidence.append(f'官方/第三方资料 {len(docs)} 条')
        if social_row is not None and int(social_row.get('valid_comment_count') or 0) > 0:
            evidence.append(f'基金公司评论样本 {social_row.get("valid_comment_count")} 条')
        if category != '其他':
            evidence.append(f'产品分类 {category}')
        contradiction = []
        if any(x in category for x in ['高收益', '股息', '股票', '香港']):
            contradiction.append('若历史回撤大或净值长期下跌，不得单独强化高派息/低估值卖点')
        if any(x in category for x in ['债券', '策略收益']):
            contradiction.append('若久期、信用评级或对冲成本缺失，不得强化稳健/降息受益卖点')
        rows.append({
            'base_fund_id': base_id,
            'base_fund_name': fund.get('base_fund_name', ''),
            'fund_company': company,
            'category': category,
            'supporting_evidence': '；'.join(evidence) if evidence else '证据不足',
            'missing_evidence': '；'.join(missing),
            'counter_evidence_to_check': '；'.join(contradiction) if contradiction else '需补充历史业绩和同类对标后再判断',
            'confidence': '中' if len(evidence) >= 2 else '低',
            'audit_decision': '可作为初步定位，不可作为最终推荐' if len(evidence) >= 1 else '必须补证后再输出结论',
        })
    return pd.DataFrame(rows)


def build_master_skill_outputs(
    base_df: pd.DataFrame,
    official_analysis_df: pd.DataFrame,
    docs_df: pd.DataFrame,
    social_analysis_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    mechanism_df = analyze_product_mechanisms(base_df, official_analysis_df, docs_df)
    return {
        'skill_catalog_df': build_master_skill_catalog(),
        'mechanism_df': mechanism_df,
        'pain_map_df': build_pain_mechanism_map(base_df, mechanism_df, social_analysis_df),
        'market_radar_df': build_market_narrative_radar(base_df, official_analysis_df, social_analysis_df),
        'evidence_audit_df': build_evidence_audit(base_df, official_analysis_df, docs_df, social_analysis_df),
    }
