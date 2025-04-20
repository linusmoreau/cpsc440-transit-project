"""Stores constants"""

import os
import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

SCHEDULES = []
for schedule in os.listdir(os.path.join(DATA_DIR, "bus-static", "sf")):
    start, end = schedule.split("_")
    start_date = datetime.datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(end, "%Y-%m-%d").date()
    SCHEDULES.append({
        "start": start_date,
        "end": end_date
    })
SCHEDULES.sort(key=lambda s: s["start"], reverse=True)