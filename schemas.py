from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class SourceType(str, Enum):
    SEARCH_RESULT = 'search_result'
    SOCIAL = 'social'
    VIDEO_COMMENT = 'video_comment'
    FORUM = 'forum'
    QNA = 'qna'
    COMPLAINT_PORTAL = 'complaint_portal'
    APP_STORE = 'app_store'
    NEWS_COMMENT = 'news_comment'
    OTHER = 'other'


class SentimentLabel(str, Enum):
    POSITIVE = 'positive'
    NEGATIVE = 'negative'
    NEUTRAL = 'neutral'
    MIXED = 'mixed'
    UNKNOWN = 'unknown'


class Intent(str, Enum):
    COMPLAINT = 'complaint'
    QUESTION = 'question'
    PURCHASE_CONSIDERATION = 'purchase_consideration'
    EXPERIENCE_SHARE = 'experience_share'
    PRAISE = 'praise'
    UNKNOWN = 'unknown'


class RiskFlag(str, Enum):
    PII = 'pii'
    AD_LIKE = 'ad_like'
    IRRELEVANT = 'irrelevant'
    STALE = 'stale'
    LOW_CONFIDENCE_DATE = 'low_confidence_date'
    SEARCH_RESULT_ONLY = 'search_result_only'
    MANUAL_REVIEW = 'manual_review'


class FundAspect(str, Enum):
    LOSS_DRAWDOWN = 'loss_drawdown'
    DIVIDEND_MISUNDERSTANDING = 'dividend_misunderstanding'
    NAV_VOLATILITY = 'nav_volatility'
    FX_OR_RMB_HEDGING = 'fx_or_rmb_hedging'
    HIGH_YIELD_CREDIT_RISK = 'high_yield_credit_risk'
    HK_ASIA_MARKET_VOLATILITY = 'hk_asia_market_volatility'
    SUBSCRIPTION_REDEMPTION_FEE = 'subscription_redemption_fee'
    SERVICE_OR_DISCLOSURE = 'service_or_disclosure'
    HIGH_DISTRIBUTION_CASHFLOW = 'high_distribution_cashflow'
    STABLE_ALLOCATION = 'stable_allocation'
    RATE_CUT_BENEFICIARY = 'rate_cut_beneficiary'
    ASIA_GROWTH = 'asia_growth'
    VALUATION_RECOVERY = 'valuation_recovery'
    MULTI_ASSET_DIVERSIFICATION = 'multi_asset_diversification'
    RMB_HEDGING = 'rmb_hedging'
    BRAND_TRUST = 'brand_trust'


@dataclass
class SourceRef:
    platform: str
    source_type: SourceType
    url: str = ''
    title: str = ''
    query: str = ''

    def __post_init__(self) -> None:
        if not isinstance(self.source_type, SourceType):
            self.source_type = SourceType(str(self.source_type))


@dataclass
class FundCompanyMention:
    id_hash: str
    source: SourceRef
    published_at: str = ''
    crawled_at: str = ''
    raw_text: str = ''
    redacted_text: str = ''
    fund_company: str = ''
    fund_names: list[str] = field(default_factory=list)
    product_codes: list[str] = field(default_factory=list)
    sentiment_label: SentimentLabel = SentimentLabel.UNKNOWN
    aspects: list[FundAspect] = field(default_factory=list)
    intent: Intent = Intent.UNKNOWN
    risk_flags: list[RiskFlag] = field(default_factory=list)
    duplicate_of: str = ''
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id_hash:
            raise ValueError('FundCompanyMention.id_hash is required')
        if not isinstance(self.source, SourceRef):
            self.source = SourceRef(**self.source)
        if not isinstance(self.sentiment_label, SentimentLabel):
            self.sentiment_label = SentimentLabel(str(self.sentiment_label))
        if not isinstance(self.intent, Intent):
            self.intent = Intent(str(self.intent))
        self.aspects = [x if isinstance(x, FundAspect) else FundAspect(str(x)) for x in self.aspects]
        self.risk_flags = [x if isinstance(x, RiskFlag) else RiskFlag(str(x)) for x in self.risk_flags]

    def to_row(self) -> dict[str, Any]:
        data = asdict(self)
        source = data.pop('source')
        data['platform'] = source['platform']
        data['source_type'] = source['source_type'].value if hasattr(source['source_type'], 'value') else source['source_type']
        data['url'] = source['url']
        data['title'] = source['title']
        data['query'] = source['query']
        data['sentiment_label'] = self.sentiment_label.value
        data['aspects'] = '；'.join(x.value for x in self.aspects)
        data['intent'] = self.intent.value
        data['risk_flags'] = '；'.join(x.value for x in self.risk_flags)
        data['fund_names'] = '；'.join(self.fund_names)
        data['product_codes'] = '；'.join(self.product_codes)
        data['metadata'] = str(self.metadata)
        return data
