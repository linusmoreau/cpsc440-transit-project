"""Module for processing bus data into more compact aggregates."""

import os
import sys
import datetime
import pandas as pd
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt

from constants import DATA_DIR

SCHEDULES = []
for schedule in os.listdir(os.path.join(DATA_DIR, "bus-static", "sf")):
    start, end = schedule.split("_")
    start_date = datetime.datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(end, "%Y-%m-%d").date()
    SCHEDULES.append({"start": start_date, "end": end_date, "dirname": schedule})
SCHEDULES.sort(key=lambda s: s["start"], reverse=True)


def load_vehicle_data(date: datetime.date) -> pd.DataFrame | None:
    """Loads and processes vehicle data for the given date"""
    path = os.path.join(DATA_DIR, "bus", "sf", str(date) + ".parquet")
    try:
        df = pd.read_parquet(path, engine="fastparquet")
    except:
        print("Failed to load bus data for", date)
        return None

    # Remove rows without valid stop_id
    df = df[df["vehicle.stop_id"].notnull()]
    df = df.astype({"vehicle.stop_id": "int64"})

    # Remove rows without valid trip_id
    df = df[df["vehicle.trip.trip_id"].notnull()]
    
    # Only keep necessary columns
    df = df[["vehicle.trip.trip_id", "vehicle.timestamp", "vehicle.stop_id", "vehicle.trip.route_id", "vehicle.trip.direction_id"]]

    # Only keep the last record before a bus arrives
    df.drop_duplicates(subset=["vehicle.trip.trip_id", "vehicle.stop_id"], keep="last", inplace=True)
    
    # Drop last stop of each trip as it may not indicate an arrival
    mask = df.duplicated(["vehicle.trip.trip_id"], keep="last")
    df = df[mask]
    
    return df
    

def load_schedule_data_for_date(date: datetime.date) -> pd.DataFrame:
    """Loads and processes schedule data for the given date"""
    return load_schedule_data(get_dirname_for_gtfs_static(date))


def load_schedule_data(dirname: str) -> pd.DataFrame:
    """Loads and processes schedule data for the given directory name"""
    schedule_dir = os.path.join(DATA_DIR, "bus-static", "sf", dirname)
    path = os.path.join(schedule_dir, "stop_times.txt")
    stop_time_df = pd.read_csv(path)
    stop_time_df = stop_time_df[["trip_id", "departure_time", "stop_id"]]

    path = os.path.join(schedule_dir, "stops.txt")
    stop_df = pd.read_csv(path)
    stop_df = stop_df[["stop_id", "stop_code"]]
    return process_schedule_data(stop_time_df, stop_df)


def process_schedule_data(stop_time_df: pd.DataFrame, stop_df: pd.DataFrame) -> pd.DataFrame:
    # Add stop code info to stop times and only keep necessary columns
    df = stop_time_df.join(stop_df.set_index("stop_id"), on="stop_id")[["trip_id", "departure_time", "stop_code"]]

    # Drop all stops that appear twice in a trip to simplify comparisons
    df.drop_duplicates(subset=["trip_id", "stop_code"], keep=False, inplace=True)
    
    return df
    
    
def get_dirname_for_gtfs_static(date: datetime.date) -> str:
    """Returns the directory name for the most recent schedule that includes the given date."""
    for schedule in SCHEDULES:
        if schedule["start"] <= date and schedule["end"] >= date:
            return str(schedule["start"]) + "_" + str(schedule["end"])
    raise ValueError("No schedule satisfies the given date", date)


def get_delay(actual_time: datetime.datetime, scheduled_time: str) -> int:
    tz = ZoneInfo("America/Los_Angeles")
    t = scheduled_time.split(":")
    h = int(t[0]) % 24
    m = int(t[1])
    s = int(t[2])
    schedule_timestamp = actual_time.replace(hour=h, minute=m, second=s, tzinfo=tz)
    naive = schedule_timestamp.replace(tzinfo=None) - datetime.timedelta(1)
    schedule_timestamp_adjust = naive.replace(tzinfo=tz)
    dif = actual_time - schedule_timestamp
    dif_adjust = actual_time - schedule_timestamp_adjust
    if abs(dif_adjust) < abs(dif):
        dif = dif_adjust
        schedule_timestamp = schedule_timestamp_adjust
    return int(dif.total_seconds() / 60), schedule_timestamp.astimezone(datetime.UTC)


def get_delay_df(vehicle_df: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(
        left=vehicle_df, 
        right=schedule_df, 
        left_on=["vehicle.trip.trip_id", "vehicle.stop_id"], 
        right_on=["trip_id", "stop_code"],
        validate="1:1"
    )[["trip_id", "stop_code", "vehicle.timestamp", "departure_time", "vehicle.trip.route_id", "vehicle.trip.direction_id"]]
    
    # Get delay and formatted schedule time
    df[["delay", "schedule_time"]] = df.apply(lambda row: get_delay(row["vehicle.timestamp"], row["departure_time"]), axis=1, result_type="expand")
    
    # Only keep necessary columns
    df = df[["trip_id", "stop_code", "vehicle.timestamp", "schedule_time", "delay", "vehicle.trip.route_id", "vehicle.trip.direction_id"]]
    return df


def bucket_time(t: datetime.datetime) -> datetime.datetime:
    return t.replace(minute=t.minute//20*20, second=0, microsecond=0).astimezone(ZoneInfo("America/Los_Angeles"))


def bucket_by_time(delay_df: pd.DataFrame) -> pd.DataFrame:
    # Place every bus update into a time bucket
    delay_df["time_bucket"] = delay_df.apply(lambda row: bucket_time(row["vehicle.timestamp"]), axis=1)
    
    # Only include the first instance of the trip in the time bucket
    delay_df = delay_df.drop_duplicates(subset=["trip_id", "time_bucket"], keep="first")
    
    bucket_df = delay_df[["time_bucket", "delay"]]
    bucket_df = bucket_df.groupby(["time_bucket"]).agg(
        delay_total=("delay", "sum"),
        late_5_min=("delay", lambda x: x[x >= 5].count()),
        early_5_min=("delay", lambda x: x[x <= -5].count()),
        count=("delay", "count")
    ).reset_index()
    return bucket_df


def store_time_buckets(date: datetime.date, buckets: pd.DataFrame, route: str):
    path = os.path.join(DATA_DIR, "bus-aggregate", route)
    try:
        os.makedirs(path)
    except FileExistsError:
        # directory already exists
        pass
    fpath = os.path.join(path, str(date) + ".csv")
    buckets.to_csv(fpath, index=False)
    
    
def process_all():
    date = datetime.datetime.today().date()
    for schedule in SCHEDULES:
        dirname = schedule["dirname"]
        start = schedule["start"]
        end = schedule["end"]
        print(f"Processing schedule {dirname}...")
        if start > date:
            continue
        schedule_df = load_schedule_data(dirname)
        if end < date:
            date = end
        while date >= start:    
            print(f"Processing date {date}...")
            vehicle_df = load_vehicle_data(date)
            if vehicle_df is not None:
                try:
                    delay_df = get_delay_df(vehicle_df, schedule_df)
                    bucket_df = bucket_by_time(delay_df)
                    store_time_buckets(date, bucket_df, "all")
                    print(f"Processing date {date} complete!")
                except ValueError:
                    print(f"Failed to process data for {date}")
                    break
            date -= datetime.timedelta(1)
        print(f"Processing schedule {dirname} complete!")
        
        
def process_between(start: datetime.date, end: datetime.date):
    date = end
    for schedule in SCHEDULES:
        dirname = schedule["dirname"]
        schedule_start = schedule["start"]
        schedule_end = schedule["end"]
        print(f"Processing schedule {dirname}...")
        if schedule_start > date:
            continue
        schedule_df = load_schedule_data(dirname)
        if schedule_end < date:
            date = schedule_end
        while date >= schedule_start and date >= start:
            print(f"Processing date {date}...")
            vehicle_df = load_vehicle_data(date)
            if vehicle_df is not None:
                try:
                    delay_df = get_delay_df(vehicle_df, schedule_df)
                    bucket_df = bucket_by_time(delay_df)
                    store_time_buckets(date, bucket_df, "all")
                    print(f"Processing date {date} complete!")
                except ValueError:
                    print(f"Failed to process data for {date}")
                    break
            date -= datetime.timedelta(1)
        print(f"Processing schedule {dirname} complete!")
        if date < start:
            break
        
        
def combine_bus_aggregates():
    dir = os.path.join(DATA_DIR, "bus-aggregate", "all")
    all_file_name = "all.csv"
    all_file_path = os.path.join(dir, all_file_name)
    if os.path.exists(all_file_path):
        os.remove(all_file_path)
    aggregates = os.listdir(dir)
    dfs = []
    for a in aggregates:
        if a == all_file_name:
            continue
        path = os.path.join(dir, a)
        df = pd.read_csv(path, parse_dates=["time_bucket"])
        dfs.append(df)
    df = pd.concat(dfs)
    df = df.groupby(["time_bucket"]).sum().reset_index()
    df.to_csv(all_file_path, index=False)
        
        
def load_bucket_statistics(date: datetime.date = None) -> pd.DataFrame:
    """Loads aggregate data for the given date. If no date is given, gets data for all dates."""
    if date is None:
        name = "all"
    else:
        name = str(date)
    path = os.path.join(DATA_DIR, "bus-aggregate", "all", name + ".csv")
    df = pd.read_csv(path, parse_dates=["time_bucket"])
    return df


def plot_bucket_statistics_from_file(date: datetime.date):
    plot_bucket_statistics(load_bucket_statistics(date))

    
def plot_bucket_statistics(agg: pd.DataFrame):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.axhline(0, color='black')
    # Use 1:-1 to remove edge buckets which may be cut off
    x = agg["time_bucket"][:-1]
    ax1.plot(x, agg["delay_total"][:-1] / agg["count"][:-1], "g-", label="Mean delay")
    ax2.plot(x, agg["count"][:-1], "b-", label="Bus count")
    ax2.plot(x, agg["late_5_min"][:-1], "r-", label=">5 mins late")
    ax2.plot(x, agg["early_5_min"][:-1], "y-", label=">5 mins early")
    ax1.set_ylabel("Delays (minutes)", color="g")
    ax2.set_ylabel("Number of buses", color="b")
    fig.autofmt_xdate()
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
        fancybox=True, shadow=True, ncol=5)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
        fancybox=True, shadow=True, ncol=5)
    
    # Y-axis zero-alignment as given by KobusNell on StackOverflow: https://stackoverflow.com/a/65824524
    ax1_ylims = ax1.axes.get_ylim()           # Find y-axis limits set by the plotter
    ax1_yratio = ax1_ylims[0] / ax1_ylims[1]  # Calculate ratio of lowest limit to highest limit

    ax2_ylims = ax2.axes.get_ylim()           # Find y-axis limits set by the plotter
    ax2_yratio = ax2_ylims[0] / ax2_ylims[1]  # Calculate ratio of lowest limit to highest limit


    # If the plot limits ratio of plot 1 is smaller than plot 2, the first data set has
    # a wider range range than the second data set. Calculate a new low limit for the
    # second data set to obtain a similar ratio to the first data set.
    # Else, do it the other way around

    if ax1_yratio < ax2_yratio: 
        ax2.set_ylim(bottom = ax2_ylims[1]*ax1_yratio)
    else:
        ax1.set_ylim(bottom = ax1_ylims[1]*ax2_yratio)

    plt.tight_layout()
    
    plt.show()
    
    
if __name__ == "__main__":
    if len(sys.argv) == 1:
        process_all()
    elif len(sys.argv) == 3:
        start_date = datetime.datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(sys.argv[2], "%Y-%m-%d").date()
        process_between(start_date, end_date)
    else:
        print("To process all: python bus_processing.py")
        print("To process between dates: python bus_processing.py [start-date] [end-date]")
        print("To process between dates (example): python bus_processing.py 2025-05-09 2024-08-16")
        sys.exit()
    combine_bus_aggregates()
