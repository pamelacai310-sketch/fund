import re

import pandas as pd

PAIN_KEYWORDS = {
    '回撤/亏损焦虑': ['亏', '亏损', '跌', '大跌', '回撤', '浮亏', '净值跌', '套住'],
    '分红/派息疑惑': ['分红', '派息', '除息', '分派', '派了', '现金流', '红利'],
    '收益不及预期': ['收益低', '没赚钱', '不涨', '跑输', '收益不行', '表现差'],
    '汇率/对冲困惑': ['汇率', '人民币', '美元', '对冲', '汇损', '汇兑'],
    '风险认知不足': ['债基也会亏', '稳健也亏', '不是低风险', '风险', '暴雷', '违约'],
    '流动性/购买问题': ['怎么买', '哪里买', '赎回', '申购', '限购', '费率', '手续费'],
}

SELLING_POINT_KEYWORDS = {
    '高股息/高派息': ['高股息', '高息', '分红', '派息', '现金流', '红利'],
    '稳健/低波动': ['稳健', '低波动', '防守', '配置', '平衡', '低风险'],
    '低估值修复': ['低估', '估值修复', '便宜', '反弹', '港股'],
    '降息受益': ['降息', '债券', '利率下行', '久期'],
    '亚洲/中国增长': ['亚洲', '中国', '成长', '科技', '创新', 'AI'],
    '全球分散': ['全球', '分散', '多资产', '一站式'],
}

CATEGORY_RULES = {
    '亚洲债券': ['Asian Total Return Bond', 'Asian Bond', 'Asia Bond', '亚洲债券', '亚洲总回报债券'],
    '亚洲高收益债': ['High Yield Bond', 'High Income Bond', '亚洲高收益', '高收益债'],
    '环球债券': ['Global Bond', '环球债券', '全球债券'],
    '亚洲股息/高息股票': ['Asian Dividend', 'High-Dividend', 'Equity High Income', '亚洲股息', '高股息'],
    '亚洲/亚太股票': ['Pacific Securities', 'Asia Growth', 'Asia Pacific', '亚太股票', '亚洲股票'],
    '香港股票': ['Hong Kong', '香港股票', '港股'],
    '大中华价值股票': ['Classic Fund', 'Value Partners Classic', '大中华', '价值'],
    '主题成长股票': ['Disruptive Opportunities', 'Innovation', '科技', '创新', '主题'],
    '多资产/平衡': ['Balanced Fund', 'Defensive Balanced', 'Multi-Asset', '多资产', '平衡'],
    '环球股票': ['Global Equity', '全球股票', '环球股票'],
    '策略收益': ['Strategic Income', '策略收益'],
}

RARE_STRATEGY_RULES = {
    '低波动因子/控波动策略': ['Volatility', '低波动', '控波动', 'minimum volatility'],
    '高股息权益收益化策略': ['Dividend', 'High-Dividend', 'Equity High Income', '高股息', '股息'],
    '灵活总收益债券策略': ['Total Return Bond', 'Strategic Income', '总回报', '策略收益'],
    '人民币/汇率对冲份额设计': ['HDG', 'Hedged', '对冲', '人民币对冲'],
    '多资产收益策略': ['Multi-Asset', 'Balanced', 'Defensive Balanced', '多资产', '平衡'],
    '颠覆式创新主题': ['Disruptive', 'Innovation', '创新', '颠覆'],
    '信用下沉/高收益策略': ['High Yield', 'High Income', '高收益', '信用'],
    '久期主动管理': ['duration', '久期', '利率'],
    '衍生品/杠杆或套保工具': ['derivative', 'swap', 'futures', 'option', '衍生', '期货', '掉期'],
}


def classify_fund_category(text: str) -> str:
    text = str(text)
    for category, keywords in CATEGORY_RULES.items():
        if any(kw.lower() in text.lower() for kw in keywords):
            return category
    return '其他'


def detect_rare_strategies(text: str) -> list[str]:
    found = []
    for strategy, keywords in RARE_STRATEGY_RULES.items():
        if any(kw.lower() in str(text).lower() for kw in keywords):
            found.append(strategy)
    return found


def count_keyword_groups(text: str, groups: dict[str, list[str]]) -> dict[str, int]:
    return {name: sum(len(re.findall(re.escape(kw), text, flags=re.I)) for kw in kws) for name, kws in groups.items()}


def format_top_hits(hit_dict: dict[str, int], top_n: int = 3) -> str:
    hits = sorted([(k, v) for k, v in hit_dict.items() if v > 0], key=lambda x: x[1], reverse=True)
    return '；'.join([f'{k}({v})' for k, v in hits[:top_n]]) if hits else '未识别出高频信号'


def summarize_doc_evidence(docs: pd.DataFrame) -> str:
    if docs.empty:
        return '未检索到可用官方资料。'
    candidates = docs.sort_values(['is_latest_candidate', 'relevance_score'], ascending=[False, False]).head(3)
    pieces = []
    for _, row in candidates.iterrows():
        date_part = f"，日期 {row.get('document_date')}" if row.get('document_date') else ''
        pieces.append(f"{row.get('doc_type_guess')}：{row.get('title')}{date_part}（{row.get('source_domain')}）")
    return '；'.join(pieces)


def product_features_for(category: str, strategies: list[str], docs: pd.DataFrame) -> str:
    base = {
        '亚洲债券': '以亚洲债券为核心，重点看久期、信用利差、币种和收益来源的主动管理。',
        '亚洲高收益债': '收益弹性和信用利差暴露更高，票息吸引力与违约/流动性风险并存。',
        '环球债券': '全球债券配置，分散单一区域信用与利率风险。',
        '亚洲股息/高息股票': '用股息和收益型股票表达权益收益，本质仍需承受股票波动。',
        '亚洲/亚太股票': '提供亚洲或亚太股票敞口，受区域行业结构、政策周期和汇率影响。',
        '香港股票': '集中港股 Beta 和估值修复弹性，政策预期和流动性影响较大。',
        '大中华价值股票': '偏价值投资，核心看低估值、基本面修复和持仓集中度。',
        '主题成长股票': '主题成长属性强，需关注估值、行业集中和换手。',
        '多资产/平衡': '通过股债或多资产配置降低单一资产择时难度，核心看回撤控制。',
        '环球股票': '全球权益配置，核心看区域和行业分散度。',
        '策略收益': '收益来源多元，可能跨债券、信用、货币和其他收益资产灵活配置。',
    }.get(category, '分类不明确，需要依赖最新月报/持仓进一步确认定位。')
    if strategies:
        return f"{base} 已识别差异化线索：{'、'.join(strategies)}。"
    if docs.empty:
        return f"{base} 当前缺少可核验资料，结论需补充官方月报。"
    return base


def build_peer_positioning(base_df: pd.DataFrame, categories: dict[str, str]) -> dict[str, str]:
    by_category: dict[str, list[str]] = {}
    for _, row in base_df.iterrows():
        category = categories[row['base_fund_id']]
        by_category.setdefault(category, []).append(row['base_fund_name'])
    out = {}
    for _, row in base_df.iterrows():
        category = categories[row['base_fund_id']]
        peers = [name for name in by_category.get(category, []) if name != row['base_fund_name']]
        if peers:
            out[row['base_fund_id']] = f"同类基金：{'；'.join(peers[:8])}。对比重点：收益来源、回撤、久期/行业集中、币种或对冲份额、费用。"
        else:
            out[row['base_fund_id']] = f"列表内没有完全同类基金。建议与外部同类别产品比较收益、风险、费用和最新持仓。"
    return out


def analyze_official_docs(base_df: pd.DataFrame, docs_df: pd.DataFrame) -> pd.DataFrame:
    categories = {}
    for _, fund in base_df.iterrows():
        text = ' '.join(str(fund.get(col, '')) for col in ['base_fund_name', 'fund_names_cn', 'fund_names_en'])
        categories[fund['base_fund_id']] = classify_fund_category(text)
    peer_positioning = build_peer_positioning(base_df, categories)

    rows = []
    for _, fund in base_df.iterrows():
        docs = docs_df[docs_df['base_fund_id'] == fund['base_fund_id']] if not docs_df.empty else pd.DataFrame()
        doc_text = ' '.join(
            docs.get(col, pd.Series(dtype=str)).fillna('').astype(str).str.cat(sep=' ')
            for col in ['title', 'snippet', 'text_excerpt'] if col in docs
        )
        category = categories[fund['base_fund_id']]
        strategies = detect_rare_strategies(f"{fund.get('base_fund_name', '')} {doc_text}")
        latest = docs[docs.get('is_latest_candidate', pd.Series(dtype=bool)).astype(str).str.lower().isin(['true', '1'])] if not docs.empty else pd.DataFrame()
        rows.append({
            'base_fund_id': fund['base_fund_id'],
            'base_fund_name': fund['base_fund_name'],
            'category': category,
            'official_doc_count': len(docs),
            'latest_document_title': latest.iloc[0]['title'] if not latest.empty else '',
            'latest_document_date': latest.iloc[0]['document_date'] if not latest.empty else '',
            'latest_document_link': latest.iloc[0]['link'] if not latest.empty else '',
            'product_features': product_features_for(category, strategies, docs),
            'rare_or_differentiated_strategies': '；'.join(strategies) if strategies else '未从可用资料中识别出明显罕见策略',
            'peer_comparison': peer_positioning[fund['base_fund_id']],
            'evidence_summary': summarize_doc_evidence(docs),
        })
    return pd.DataFrame(rows)


def analyze_social_comments(social_df: pd.DataFrame) -> pd.DataFrame:
    if social_df.empty:
        return pd.DataFrame(columns=[
            'product_code', 'social_result_count', 'recent_result_count', 'platforms',
            'source_mix', 'top_pain_points', 'top_selling_points',
            'investor_pain_analysis', 'content_selling_angle', 'sample_comments_or_snippets',
        ])
    df = social_df.copy()
    if 'user_text' not in df.columns:
        df['user_text'] = df.get('snippet', '')
    df['is_recent_bool'] = df.get('is_recent', '').astype(str).str.lower().isin(['true', '1'])
    rows = []
    for product_code, g in df.groupby('product_code', dropna=False):
        texts = ' '.join(g['user_text'].fillna('').astype(str).tolist())
        pain_hits = count_keyword_groups(texts, PAIN_KEYWORDS)
        selling_hits = count_keyword_groups(texts, SELLING_POINT_KEYWORDS)
        top_pain = format_top_hits(pain_hits)
        top_selling = format_top_hits(selling_hits)
        rows.append({
            'product_code': product_code,
            'social_result_count': len(g),
            'recent_result_count': int(g['is_recent_bool'].sum()) if 'is_recent_bool' in g else '',
            'platforms': ', '.join(sorted(set(g.get('platform', pd.Series(dtype=str)).astype(str)))),
            'source_mix': ', '.join(sorted(set(g.get('source_type', pd.Series(dtype=str)).astype(str)))),
            'top_pain_points': top_pain,
            'top_selling_points': top_selling,
            'investor_pain_analysis': f"主要痛点集中在：{top_pain}。若样本来自公开搜索摘要，需用导出评论复核情绪强度。",
            'content_selling_angle': f"内容卖点可围绕：{top_selling}。传播时需同时提示对应风险。",
            'sample_comments_or_snippets': ' | '.join(g['user_text'].fillna('').astype(str).head(5)),
        })
    return pd.DataFrame(rows)
