import re
from collections import Counter
import pandas as pd
from config import OPENAI_API_KEY

PAIN_KEYWORDS = {
    '回撤大': ['亏', '亏损', '跌', '大跌', '回撤', '浮亏', '净值跌'],
    '分红疑惑': ['分红', '派息', '除息', '分派', '派了', '现金流'],
    '收益不及预期': ['收益低', '没赚钱', '不涨', '跑输', '收益不行'],
    '汇率/对冲困惑': ['汇率', '人民币', '美元', '对冲', '汇损', '汇兑'],
    '风险认知不足': ['债基也会亏', '稳健也亏', '不是低风险', '风险'],
    '流动性/购买问题': ['怎么买', '哪里买', '赎回', '申购', '限购'],
}

SELLING_POINT_KEYWORDS = {
    '高股息/高派息': ['高股息', '高息', '分红', '派息', '现金流'],
    '稳健配置': ['稳健', '低波动', '防守', '配置', '平衡'],
    '低估值修复': ['低估', '估值修复', '便宜', '反弹', '港股'],
    '降息受益': ['降息', '债券', '利率下行', '久期'],
    '亚洲增长': ['亚洲', '成长', '科技', '创新', 'AI'],
    '全球分散': ['全球', '分散', '多资产', '一站式'],
}

CATEGORY_RULES = {
    '亚洲债券': ['Asian Total Return Bond', 'Asian Bond'],
    '亚洲高收益债': ['High Yield Bond', 'High Income Bond'],
    '环球债券': ['Global Bond'],
    '亚洲股息/高息股票': ['Asian Dividend', 'High-Dividend', 'Equity High Income'],
    '亚洲/亚太股票': ['Pacific Securities', 'Asia Growth', 'Asia Pacific'],
    '香港股票': ['Hong Kong'],
    '大中华价值股票': ['Classic Fund', 'Value Partners Classic'],
    '主题成长股票': ['Disruptive Opportunities'],
    '多资产/平衡': ['Balanced Fund', 'Defensive Balanced', 'Multi-Asset'],
    '环球股票': ['Global Equity'],
    '策略收益': ['Strategic Income'],
}

RARE_STRATEGY_RULES = {
    '低波动因子/控波动策略': ['Volatility'],
    '高股息权益收益化策略': ['Dividend', 'High-Dividend', 'Equity High Income'],
    '灵活总收益债券策略': ['Total Return Bond', 'Strategic Income'],
    '人民币对冲份额设计': ['HDG', 'Hedged', '对冲'],
    '多资产收益策略': ['Multi-Asset', 'Balanced', 'Defensive Balanced'],
    '颠覆式创新主题': ['Disruptive', 'Innovation', '创新'],
}


def classify_fund_category(base_fund_name: str) -> str:
    text = str(base_fund_name)
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


def rule_based_product_features(category: str) -> str:
    mapping = {
        '亚洲债券': '以亚洲债券为核心，通常关注久期、信用利差、币种和收益来源的主动管理。',
        '亚洲高收益债': '收益弹性较高，信用利差暴露更明显，适合关注票息但需承受信用风险的投资者。',
        '环球债券': '全球债券配置，分散单一区域信用与利率风险。',
        '亚洲股息/高息股票': '以股息和收益型股票为主要卖点，本质仍属于权益资产。',
        '亚洲/亚太股票': '提供亚太或亚洲区域股票敞口，受区域市场、行业结构和汇率影响。',
        '香港股票': '集中配置香港市场，弹性较强，但受港股流动性、估值和政策预期影响较大。',
        '大中华价值股票': '偏价值投资风格，关注低估值和基本面修复。',
        '主题成长股票': '主题成长属性明显，可能集中于科技、创新或颠覆式商业模式。',
        '多资产/平衡': '通过股债或多资产配置控制波动，适合不想自行择时和调仓的投资者。',
        '策略收益': '收益来源较多元，可能在债券、信用、货币和其他收益资产间灵活配置。',
    }
    return mapping.get(category, '需要结合最新月报和持仓进一步判断产品特色。')


def peer_comparison_angle(category: str) -> str:
    mapping = {
        '亚洲债券': '与亚洲综合债、高收益债、环球债券比较久期、信用评级、收益率、回撤。',
        '亚洲高收益债': '与亚洲总收益债、亚洲综合债比较信用风险、票息、违约风险和净值波动。',
        '环球债券': '与亚洲债券比较区域分散、美元利率敏感度、汇率风险。',
        '亚洲股息/高息股票': '与普通亚洲股票、低波动股票比较股息率、派息稳定性、资本增长能力。',
        '亚洲/亚太股票': '与香港股票、大中华价值、亚洲成长比较区域暴露和行业集中度。',
        '香港股票': '与大中华价值、亚洲股票比较港股 Beta、估值弹性和政策敏感度。',
        '主题成长股票': '与普通成长基金比较主题集中度、估值风险和换手率。',
        '多资产/平衡': '与纯债、股息股票、策略收益基金比较波动控制和收益稳定性。',
        '策略收益': '与纯债和多资产基金比较收益来源、久期、信用和衍生品使用。',
    }
    return mapping.get(category, '与同风险等级、同区域、同资产类别产品比较。')


def analyze_official_docs(base_df: pd.DataFrame, docs_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, fund in base_df.iterrows():
        docs = docs_df[docs_df['base_fund_id'] == fund['base_fund_id']] if not docs_df.empty else pd.DataFrame()
        text = ' '.join(docs.get('title', pd.Series(dtype=str)).astype(str).tolist() + docs.get('snippet', pd.Series(dtype=str)).astype(str).tolist())
        category = classify_fund_category(fund['base_fund_name'])
        rare = detect_rare_strategies(fund['base_fund_name'] + ' ' + text)
        rows.append({
            'base_fund_id': fund['base_fund_id'],
            'base_fund_name': fund['base_fund_name'],
            'category': category,
            'official_doc_count': len(docs),
            'product_features': rule_based_product_features(category),
            'rare_or_differentiated_strategies': '；'.join(rare) if rare else '未从名称/摘要中识别出明显罕见策略',
            'peer_comparison_angle': peer_comparison_angle(category),
        })
    return pd.DataFrame(rows)


def count_keyword_groups(text: str, groups: dict[str, list[str]]) -> dict[str, int]:
    return {name: sum(len(re.findall(re.escape(kw), text, flags=re.I)) for kw in kws) for name, kws in groups.items()}


def format_top_hits(hit_dict: dict[str, int], top_n: int = 3) -> str:
    hits = sorted([(k, v) for k, v in hit_dict.items() if v > 0], key=lambda x: x[1], reverse=True)
    return '；'.join([f'{k}({v})' for k, v in hits[:top_n]]) if hits else '公开搜索结果中未识别出高频信号'


def analyze_social_comments(social_df: pd.DataFrame) -> pd.DataFrame:
    if social_df.empty:
        return pd.DataFrame()
    df = social_df.copy()
    if 'user_text' not in df.columns:
        df['user_text'] = df.get('snippet', '')
    rows = []
    for product_code, g in df.groupby('product_code'):
        texts = ' '.join(g['user_text'].fillna('').astype(str).tolist())
        rows.append({
            'product_code': product_code,
            'social_result_count': len(g),
            'platforms': ', '.join(sorted(set(g['platform'].astype(str)))) if 'platform' in g.columns else '',
            'top_pain_points': format_top_hits(count_keyword_groups(texts, PAIN_KEYWORDS)),
            'top_selling_points': format_top_hits(count_keyword_groups(texts, SELLING_POINT_KEYWORDS)),
            'sample_comments_or_snippets': ' | '.join(g['user_text'].fillna('').astype(str).head(5)),
        })
    return pd.DataFrame(rows)


def llm_analyze_fund(base_name: str, official_text: str, social_text: str = '') -> str:
    if not OPENAI_API_KEY:
        return '未设置 OPENAI_API_KEY，跳过 LLM 分析。'
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f'''你是基金产品研究员。请基于以下资料分析基金：\n基金名称：{base_name}\n\n官方资料：\n{official_text[:12000]}\n\n社媒讨论：\n{social_text[:6000]}\n\n请输出：产品定位、同类基金对比、产品特色、罕见或差异化投资策略、投资者痛点、内容卖点、风险提示。不要编造资料，无法确认请标注“需核验”。'''
    resp = client.chat.completions.create(model='gpt-4.1-mini', messages=[{'role': 'user', 'content': prompt}], temperature=0.2)
    return resp.choices[0].message.content
