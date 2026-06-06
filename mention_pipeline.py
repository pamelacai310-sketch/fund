from __future__ import annotations

from collections import Counter
from datetime import date, datetime
import hashlib
import re

import pandas as pd

from schemas import (
    FundAspect,
    FundCompanyMention,
    Intent,
    RiskFlag,
    SentimentLabel,
    SourceRef,
    SourceType,
)
from social_search import is_recent_date, parse_date
from window import default_window


MENTION_COLUMNS = [
    'id_hash', 'fund_company', 'fund_names', 'product_codes', 'platform', 'source_type',
    'url', 'title', 'query', 'published_at', 'crawled_at', 'raw_text', 'redacted_text',
    'sentiment_label', 'aspects', 'intent', 'risk_flags', 'duplicate_of', 'metadata',
]

PAIN_ASPECT_RULES = {
    FundAspect.LOSS_DRAWDOWN: ['亏', '亏损', '回撤', '大跌', '套住', '浮亏'],
    FundAspect.DIVIDEND_MISUNDERSTANDING: ['分红', '派息', '除息', '分派', '红利'],
    FundAspect.NAV_VOLATILITY: ['净值', '波动', '涨跌', '净值跌'],
    FundAspect.FX_OR_RMB_HEDGING: ['汇率', '人民币', '美元', '对冲', '汇损', '汇兑'],
    FundAspect.HIGH_YIELD_CREDIT_RISK: ['高收益债', '信用', '违约', '暴雷', '债基也会亏'],
    FundAspect.HK_ASIA_MARKET_VOLATILITY: ['港股', '亚洲', '亚太', '香港', '市场波动'],
    FundAspect.SUBSCRIPTION_REDEMPTION_FEE: ['申购', '赎回', '费率', '手续费', '限购'],
    FundAspect.SERVICE_OR_DISCLOSURE: ['客服', '披露', '看不懂', '资料', '月报'],
}

SELLING_ASPECT_RULES = {
    FundAspect.HIGH_DISTRIBUTION_CASHFLOW: ['高股息', '高息', '派息', '分红', '现金流'],
    FundAspect.STABLE_ALLOCATION: ['稳健', '配置', '平衡', '低波动', '防守'],
    FundAspect.RATE_CUT_BENEFICIARY: ['降息', '利率下行', '债券', '久期'],
    FundAspect.ASIA_GROWTH: ['亚洲增长', '亚洲', '成长', '创新', '科技'],
    FundAspect.VALUATION_RECOVERY: ['低估', '估值修复', '便宜', '反弹'],
    FundAspect.MULTI_ASSET_DIVERSIFICATION: ['多资产', '分散', '一站式', '股债'],
    FundAspect.RMB_HEDGING: ['人民币对冲', 'RMB hedged', 'CNY HDG', '对冲'],
    FundAspect.BRAND_TRUST: ['摩根', '汇丰', '惠理', '百达', '东方汇理', '东亚联丰', '品牌'],
}

NEGATIVE_WORDS = ['亏', '跌', '回撤', '套住', '差', '不行', '暴雷', '违约', '赎回难', '客服差']
POSITIVE_WORDS = ['稳健', '不错', '高息', '派息', '分红', '现金流', '信任', '低估', '反弹']
QUESTION_WORDS = ['吗', '怎么', '哪里', '为什么', '?', '？', '能买吗', '怎么买']


def normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', str(text or '')).strip().lower()


def redact_pii(text: str) -> tuple[str, bool]:
    redacted = str(text or '')
    patterns = [
        r'1[3-9]\d{9}',
        r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+',
        r'\b\d{6,}\b',
    ]
    changed = False
    for pattern in patterns:
        redacted, count = re.subn(pattern, '[REDACTED]', redacted)
        changed = changed or count > 0
    return redacted, changed


def build_content_hash(platform: str, url: str, redacted_text: str) -> str:
    key = '|'.join([normalize_text(platform), normalize_text(url), normalize_text(redacted_text)])
    return hashlib.sha256(key.encode('utf-8')).hexdigest()[:24]


def _match_aspects(text: str, rules: dict[FundAspect, list[str]]) -> list[FundAspect]:
    lowered = str(text or '').lower()
    return [aspect for aspect, keywords in rules.items() if any(kw.lower() in lowered for kw in keywords)]


def infer_aspects(text: str) -> list[FundAspect]:
    out = []
    for aspect in _match_aspects(text, PAIN_ASPECT_RULES) + _match_aspects(text, SELLING_ASPECT_RULES):
        if aspect not in out:
            out.append(aspect)
    return out


def infer_sentiment(text: str) -> SentimentLabel:
    lowered = str(text or '').lower()
    neg = sum(1 for word in NEGATIVE_WORDS if word.lower() in lowered)
    pos = sum(1 for word in POSITIVE_WORDS if word.lower() in lowered)
    if pos and neg:
        return SentimentLabel.MIXED
    if neg:
        return SentimentLabel.NEGATIVE
    if pos:
        return SentimentLabel.POSITIVE
    return SentimentLabel.UNKNOWN


def infer_intent(text: str) -> Intent:
    lowered = str(text or '').lower()
    if any(word.lower() in lowered for word in QUESTION_WORDS):
        return Intent.QUESTION
    if any(word.lower() in lowered for word in ['亏', '投诉', '客服差', '赎回难', '不行']):
        return Intent.COMPLAINT
    if any(word.lower() in lowered for word in ['能买吗', '怎么买', '考虑', '适合买吗']):
        return Intent.PURCHASE_CONSIDERATION
    if any(word.lower() in lowered for word in ['不错', '信任', '满意']):
        return Intent.PRAISE
    if str(text or '').strip():
        return Intent.EXPERIENCE_SHARE
    return Intent.UNKNOWN


def normalize_source_type(value: str) -> SourceType:
    text = normalize_text(value).replace('-', '_')
    mapping = {
        'public_search_result': SourceType.SEARCH_RESULT,
        'search_result': SourceType.SEARCH_RESULT,
        'user_exported_comment': SourceType.SOCIAL,
        'comment': SourceType.SOCIAL,
        'social': SourceType.SOCIAL,
        'video_comment': SourceType.VIDEO_COMMENT,
        'forum': SourceType.FORUM,
        'qna': SourceType.QNA,
        'complaint_portal': SourceType.COMPLAINT_PORTAL,
        'app_store': SourceType.APP_STORE,
        'news_comment': SourceType.NEWS_COMMENT,
    }
    return mapping.get(text, SourceType.OTHER)


def _split_values(value: str) -> list[str]:
    values = re.split(r'[,;；|]', str(value or ''))
    return [x.strip() for x in values if x.strip()]


def build_mentions_from_social_results(
    social_df: pd.DataFrame,
    window: tuple[date, date] | None = None,
) -> pd.DataFrame:
    since, until = window or default_window()
    if social_df.empty:
        return pd.DataFrame(columns=MENTION_COLUMNS)
    crawled_at = datetime.now().replace(microsecond=0).isoformat()
    mentions: list[FundCompanyMention] = []
    for _, row in social_df.iterrows():
        raw_text = str(row.get('user_text') or row.get('snippet') or '')
        redacted_text, has_pii = redact_pii(raw_text)
        source_type = normalize_source_type(row.get('source_type', ''))
        url = str(row.get('link') or row.get('url') or '')
        platform = str(row.get('platform') or '')
        published_at = parse_date(row.get('publish_time', ''))
        flags: list[RiskFlag] = []
        if has_pii:
            flags.append(RiskFlag.PII)
        if source_type == SourceType.SEARCH_RESULT:
            flags.append(RiskFlag.SEARCH_RESULT_ONLY)
        if not published_at:
            flags.extend([RiskFlag.LOW_CONFIDENCE_DATE, RiskFlag.MANUAL_REVIEW])
        elif not is_recent_date(published_at, since, until):
            flags.append(RiskFlag.STALE)
        if not redacted_text.strip():
            flags.extend([RiskFlag.IRRELEVANT, RiskFlag.MANUAL_REVIEW])
        if len(redacted_text.strip()) < 8 and RiskFlag.MANUAL_REVIEW not in flags:
            flags.append(RiskFlag.MANUAL_REVIEW)
        id_hash = build_content_hash(platform, url, redacted_text)
        mentions.append(FundCompanyMention(
            id_hash=id_hash,
            source=SourceRef(
                platform=platform,
                source_type=source_type,
                url=url,
                title=str(row.get('title') or ''),
                query=str(row.get('query') or ''),
            ),
            published_at=published_at,
            crawled_at=crawled_at,
            raw_text=raw_text,
            redacted_text=redacted_text,
            fund_company=str(row.get('fund_company') or ''),
            fund_names=_split_values(str(row.get('base_fund_name') or row.get('fund_names') or '')),
            product_codes=_split_values(str(row.get('product_code') or row.get('product_codes') or '')),
            sentiment_label=infer_sentiment(redacted_text),
            aspects=infer_aspects(redacted_text),
            intent=infer_intent(redacted_text),
            risk_flags=list(dict.fromkeys(flags)),
            metadata={
                'like_count': str(row.get('like_count') or ''),
                'date_source': str(row.get('date_source') or ''),
                'window_start': since.isoformat(),
                'window_end': until.isoformat(),
                'timezone': 'Asia/Shanghai',
            },
        ))
    mention_df = pd.DataFrame([mention.to_row() for mention in mentions])
    return dedupe_mentions(mention_df)[MENTION_COLUMNS]


def dedupe_mentions(mentions_df: pd.DataFrame) -> pd.DataFrame:
    if mentions_df.empty:
        return mentions_df
    df = mentions_df.copy()
    first_seen: dict[str, str] = {}
    duplicate_of = []
    for id_hash in df['id_hash'].astype(str):
        duplicate_of.append(first_seen.get(id_hash, ''))
        first_seen.setdefault(id_hash, id_hash)
    df['duplicate_of'] = duplicate_of
    return df


def _has_flag(value: str, flag: RiskFlag) -> bool:
    return flag.value in str(value or '').split('；')


def build_review_queue(mentions_df: pd.DataFrame) -> pd.DataFrame:
    if mentions_df.empty:
        return pd.DataFrame(columns=MENTION_COLUMNS + ['review_reason'])
    df = mentions_df.copy()
    reasons = []
    mask = []
    for _, row in df.iterrows():
        flags = str(row.get('risk_flags') or '').split('；')
        row_reasons = []
        if RiskFlag.MANUAL_REVIEW.value in flags:
            row_reasons.append('需要人工确认文本/归属/日期')
        if RiskFlag.LOW_CONFIDENCE_DATE.value in flags:
            row_reasons.append('发布时间缺失或低置信')
        if RiskFlag.SEARCH_RESULT_ONLY.value in flags:
            row_reasons.append('仅为搜索结果摘要，需确认是否真实用户评论')
        if RiskFlag.IRRELEVANT.value in flags:
            row_reasons.append('文本为空或疑似不相关')
        reasons.append('；'.join(row_reasons))
        mask.append(bool(row_reasons))
    df['review_reason'] = reasons
    return df[mask].reset_index(drop=True)


def build_audit_summary(
    raw_social_df: pd.DataFrame,
    mentions_df: pd.DataFrame,
    window: tuple[date, date] | None = None,
) -> pd.DataFrame:
    since, until = window or default_window()
    rows = [{
        'metric': 'window',
        'value': f'{since.isoformat()} 至 {until.isoformat()}',
        'note': 'Asia/Shanghai，自然月回推 6 个月',
    }, {
        'metric': 'raw_social_rows',
        'value': len(raw_social_df),
        'note': '搜索结果和导出评论原始行数',
    }, {
        'metric': 'standardized_mentions',
        'value': len(mentions_df),
        'note': '标准化后的 FundCompanyMention 行数',
    }]
    if mentions_df.empty:
        return pd.DataFrame(rows)
    valid = mentions_df[mentions_df['duplicate_of'].astype(str).eq('')]
    recent = valid[
        ~valid['risk_flags'].astype(str).str.contains(RiskFlag.STALE.value)
        & ~valid['risk_flags'].astype(str).str.contains(RiskFlag.LOW_CONFIDENCE_DATE.value)
    ]
    low_conf = mentions_df['risk_flags'].astype(str).str.contains(RiskFlag.LOW_CONFIDENCE_DATE.value).sum()
    review = len(build_review_queue(mentions_df))
    rows.extend([
        {'metric': 'deduped_comment_count', 'value': len(valid), 'note': 'id_hash 去重后的唯一记录数'},
        {'metric': 'recent_hit_rate', 'value': round(len(recent) / len(valid), 4) if len(valid) else 0, 'note': '去重记录中未被标记 stale 的比例'},
        {'metric': 'low_confidence_count', 'value': int(low_conf), 'note': '发布时间缺失或低置信记录'},
        {'metric': 'manual_review_count', 'value': int(review), 'note': '建议人工复核记录'},
    ])
    for platform, count in Counter(valid['platform'].astype(str)).items():
        rows.append({'metric': f'platform_count:{platform}', 'value': int(count), 'note': '去重后平台命中数'})
    for source_type, count in Counter(valid['source_type'].astype(str)).items():
        rows.append({'metric': f'source_type_count:{source_type}', 'value': int(count), 'note': '去重后来源类型命中数'})
    return pd.DataFrame(rows)
