import pandas as pd
from pathlib import Path

DATA_ROOT = Path(__file__).parent.parent.parent / "data"

def get_prices(round_num: int, day_num: int) -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / f"round{round_num}" / f"prices_round_{round_num}_day_{day_num}.csv", sep=";")

def get_trades(round_num: int, day_num: int) -> pd.DataFrame:
    for suffix in ["wn", "nn"]:
        file = DATA_ROOT / f"round{round_num}" / f"trades_round_{round_num}_day_{day_num}_{suffix}.csv"
        if file.is_file():
            return pd.read_csv(file, sep=";")

    raise ValueError(f"Cannot find trades data for round {round_num} day {day_num}")
