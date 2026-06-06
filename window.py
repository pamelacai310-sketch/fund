from __future__ import annotations

from datetime import date
import calendar

from config import SOCIAL_RECENT_MONTHS


def subtract_months(value: date, months: int) -> date:
    month_index = value.month - 1 - months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def default_window(today: date | None = None, months: int = SOCIAL_RECENT_MONTHS) -> tuple[date, date]:
    end = today or date.today()
    return subtract_months(end, months), end
