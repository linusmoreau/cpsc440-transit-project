# CPSC 440 Bus Delay Project

## Obtaining Data

Bus data can be downloaded using `downloader.py` by:

```zsh
python src/downloader.py
```

Static GTFS data can be acquired from https://mobilitydatabase.org/feeds/gtfs/mdb-50. The static GTFS data is provided as a zip file. 

To add a schedule to the project:
1. Download the static GTFS zip file.
2. Decompress the zip file as a directory.
3. Name the directory according to the start and end dates in the `calendar.txt` file within (i.e. if the dates are `20250315` and `20250509`, name the directory `2025-03-15_2025-05-09`).
4. Place the directory within `data/bus-static/sf/`.

## Sources
Static GTFS data: https://mobilitydatabase.org/feeds/gtfs/mdb-50
