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
        day_of_week = date.strftime("%A")
        calendar.append([
            date,
            int(date in CALIFORNIA_HOLIDAYS),
            int(day_of_week == "Monday"),
            int(day_of_week == "Tuesday"),
            int(day_of_week == "Wednesday"),
            int(day_of_week == "Thursday"),
            int(day_of_week == "Friday"),
            int(day_of_week == "Saturday"),
            int(day_of_week == "Sunday"),
        ])
        date += datetime.timedelta(1)
    return pd.DataFrame(calendar, columns=[
        "date",
        "holiday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday"
    ]).set_index("date")
