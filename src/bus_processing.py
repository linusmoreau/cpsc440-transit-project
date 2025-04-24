"""Module for processing bus data into more compact aggregates."""

import os
import sys
import datetime
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt

from constants import DATA_DIR

SCHEDULES = []
try:
    for schedule in os.listdir(os.path.join(DATA_DIR, "bus-static", "sf")):
        start, end = schedule.split("_")
        start_date = datetime.datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(end, "%Y-%m-%d").date()
        SCHEDULES.append({"start": start_date, "end": end_date, "dirname": schedule})
    SCHEDULES.sort(key=lambda s: s["start"], reverse=True)
except FileNotFoundError:
    pass


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


def bucket_time(t: datetime.datetime, bucket_size: int) -> datetime.datetime:
    return t.replace(minute=t.minute//bucket_size*bucket_size, second=0, microsecond=0).astimezone(ZoneInfo("America/Los_Angeles"))


def assign_buckets(delay_df: pd.DataFrame, bucket_size: int):
    # Reject invalid bucket_size
    if 60 % bucket_size != 0:
        raise ValueError("bucket_size must be an integer divisor of 60")
    
    # Place every bus update into a time bucket
    delay_df["time_bucket"] = delay_df.apply(lambda row: bucket_time(row["vehicle.timestamp"], bucket_size), axis=1)
    
    # Only include the first instance of the trip in the time bucket
    delay_df = delay_df.drop_duplicates(subset=["trip_id", "time_bucket"], keep="first")
    
    return delay_df


def bucket_by_time(bucket_df: pd.DataFrame, setting: str = "all") -> pd.DataFrame:
    """Setting is either `all` for aggregation of all buses or `routes` for aggregation by route."""
    if setting == "all":
        group_columns = ["time_bucket"]
    elif setting == "routes":
        group_columns = ["vehicle.trip.route_id", "vehicle.trip.direction_id", "time_bucket"]
    bucket_df = bucket_df[group_columns + ["delay"]]
    bucket_df = bucket_df.groupby(group_columns).agg(
        delay_total=("delay", "sum"),
        late_5_min=("delay", lambda x: x[x >= 5].count()),
        early_5_min=("delay", lambda x: x[x <= -5].count()),
        count=("delay", "count")
    ).reset_index()
    return bucket_df


def store_time_buckets(date: datetime.date, buckets: pd.DataFrame, setting: str):
    path = os.path.join(DATA_DIR, "bus-aggregate", setting)
    try:
        os.makedirs(path)
    except FileExistsError:
        # directory already exists
        pass
    fpath = os.path.join(path, str(date) + ".csv")
    buckets.to_csv(fpath, index=False)
    
    
def process_all(bucket_size: int = 20):
    """`bucket_size` describes how long the time segments are in minutes. It must be an integer divisor of 60."""
    process_between("1970-01-01", str(datetime.datetime.today().date()), bucket_size)
        
        
def process_between(start: str, end: str, bucket_size: int = 20):
    start = datetime.datetime.strptime(start, "%Y-%m-%d").date()
    end = datetime.datetime.strptime(end, "%Y-%m-%d").date()
    date = end
    for schedule in SCHEDULES:
        dirname = schedule["dirname"]
        schedule_start = schedule["start"]
        schedule_end = schedule["end"]
        if schedule_start > date:
            continue
        print(f"Processing schedule {dirname}...")
        schedule_df = load_schedule_data(dirname)
        if schedule_end < date:
            date = schedule_end
        while date >= schedule_start and date >= start:
            print(f"Processing date {date}...")
            vehicle_df = load_vehicle_data(date)
            if vehicle_df is not None:
                try:
                    delay_df = get_delay_df(vehicle_df, schedule_df)
                    bucket_df = assign_buckets(delay_df, bucket_size)
                    bucket_all_df = bucket_by_time(bucket_df, "all")
                    store_time_buckets(date, bucket_all_df, "all")
                    bucket_route_df = bucket_by_time(bucket_df, "routes")
                    store_time_buckets(date, bucket_route_df, "routes")
                    print(f"Processing date {date} complete!")
                except ValueError:
                    print(f"Failed to process data for {date}")
                    break
            date -= datetime.timedelta(1)
        if date < start:
            break
    print("Processing complete!")
        
        
def combine_bus_aggregates(setting: str = "all"):
    """Setting is either `all` for aggregation of all buses or `routes` for aggregation by route."""
    dir = os.path.join(DATA_DIR, "bus-aggregate", setting)
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
    if setting == "all":
        group_columns = ["time_bucket"]
    elif setting == "routes":
        group_columns = ["vehicle.trip.route_id", "vehicle.trip.direction_id", "time_bucket"]
    else:
        raise ValueError(f"Unknown setting \"{setting}\"")
    df = df.groupby(group_columns).sum().reset_index()
    if setting == "routes":
        df = df.sort_values(group_columns, ascending=True)
    df.to_csv(all_file_path, index=False)
        
        
def load_bucket_statistics(setting: str = "all", date: datetime.date = None) -> pd.DataFrame:
    """Loads aggregate data for the given setting and date. 
    Setting can be either `all` or `routes`.
    If no date is given, gets data for all dates."""
    if date is None:
        name = "all"
    else:
        name = str(date)
    dtypes = {
        "time_bucket": "object",
        "delay_total": "Int32",
        "late_5_min": "UInt32",
        "early_5_min": "UInt32",
        "count": "UInt32",
    }
    if setting == "all":
        pass
    elif setting == "routes":
        dtypes["vehicle.trip.route_id"] = "string"
        dtypes["vehicle.trip.direction_id"] = "Float32"
    else:
        raise ValueError("Setting must be either all or routes.")
    path = os.path.join(DATA_DIR, "bus-aggregate", setting, name + ".csv")
    df = pd.read_csv(path, dtype=dtypes, engine="python", parse_dates=["time_bucket"])
    return df


def load_dataset(setting: str = "all") -> pd.DataFrame:
    if setting == "all":
        name = "dataset.csv"
    elif setting == "routes":
        name = "dataset_routes.csv"
    else:
        raise ValueError("Setting must be either all or routes.")
    path = os.path.join(DATA_DIR, name)
    df = pd.read_csv(path, engine="python", parse_dates=["time_bucket"])
    return df


def encode_cyclical_feature(data, col, max_val):
    data[col + "_sin"] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + "_cos"] = np.cos(2 * np.pi * data[col]/max_val)
    return data


def prepare_bus_data(data: pd.DataFrame):
    boundary_time = pd.Timestamp("2025-01-01 00:00:00-08:00")

    # add avg_delay feature
    data["avg_delay"] = data["delay_total"] / data["count"]
    data["avg_delay"] = data["avg_delay"].fillna(0)

    # add time features
    def get_time_features(row):
        d: datetime.datetime = row["time_bucket"]
        return [d.timetuple().tm_yday, 60*d.hour + d.minute]
    data[["day", "minute"]] = data.apply(get_time_features, axis=1, result_type="expand")
    data = encode_cyclical_feature(data, "day", 366)
    data = encode_cyclical_feature(data, "minute", 60*24)
    data = encode_cyclical_feature(data, "weekday", 7)
    data = data.drop(columns=["day", "minute", "weekday"])
    
    if "vehicle.trip.route_id" in list(data):
        data["vehicle.trip.route_id"] = data["vehicle.trip.route_id"].astype("category").cat.codes
    
    # using 2024 data as training, 2025 as test
    train_data = data[data["time_bucket"] < boundary_time]
    test_data = data[data["time_bucket"] >= boundary_time]

    drop_columns = ["delay_total", "count", "avg_delay", "late_5_min", "early_5_min"]
    X_train = train_data.drop(columns=drop_columns)
    y_train = train_data["avg_delay"]
    X_test = test_data.drop(columns=drop_columns)
    y_test = test_data["avg_delay"]

    return X_train, y_train, X_test, y_test


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
        process_between(sys.argv[1], sys.argv[2])
    else:
        print("To process all: python bus_processing.py")
        print("To process between dates: python bus_processing.py [start-date] [end-date]")
        print("To process between dates (example): python bus_processing.py 2025-05-09 2024-08-16")
        sys.exit()
    combine_bus_aggregates()
