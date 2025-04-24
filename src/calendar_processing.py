import pandas as pd
import datetime
import holidays

CALIFORNIA_HOLIDAYS = holidays.country_holidays("US", subdiv="CA", categories=holidays.GOVERNMENT)

def build_calendar(start="2023-01-01", end="2025-04-14") -> pd.DataFrame:
    start = datetime.datetime.strptime(start, "%Y-%m-%d").date()
    end = datetime.datetime.strptime(end, "%Y-%m-%d").date()
    date = start
    calendar = []
    while date <= end:
        calendar.append([
            date,
            int(date in CALIFORNIA_HOLIDAYS),
            date.weekday(),
        ])
        date += datetime.timedelta(1)
    return pd.DataFrame(calendar, columns=["date", "holiday", "weekday"]).set_index("date")
