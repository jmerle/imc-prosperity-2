import itertools
import json
import numpy as np
from abc import abstractmethod
from datamodel import Order, Symbol, TradingState
from pathlib import Path
from prosperity2bt.data import read_day_data
from prosperity2bt.file_reader import PackageResourcesReader
from prosperity2bt.runner import run_backtest
from tqdm.contrib.concurrent import process_map
from typing import TypeAlias

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders = []
        self.conversions = 0

        self.act(state)

        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def convert(self, amount: int) -> None:
        self.conversions += amount

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

class GiftBasketStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int, long_threshold: float, short_threshold: float) -> None:
        super().__init__(symbol, limit)

        self.long_threshold = long_threshold
        self.short_threshold = short_threshold

    def act(self, state: TradingState) -> None:
        if any(symbol not in state.order_depths for symbol in ["CHOCOLATE", "STRAWBERRIES", "ROSES", "GIFT_BASKET"]):
            return

        chocolate = self.get_mid_price(state, "CHOCOLATE")
        strawberries = self.get_mid_price(state, "STRAWBERRIES")
        roses = self.get_mid_price(state, "ROSES")
        gift_basket = self.get_mid_price(state, "GIFT_BASKET")

        content = 4 * chocolate + 6 * strawberries + roses
        diff = gift_basket - content

        if diff < self.long_threshold:
            self.go_long(state)
        elif diff > self.short_threshold:
            self.go_short(state)

    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return (popular_buy_price + popular_sell_price) / 2

    def go_long(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = max(order_depth.sell_orders.keys())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position

        self.buy(price, to_buy)

    def go_short(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = min(order_depth.buy_orders.keys())

        position = state.position.get(self.symbol, 0)
        to_sell = self.limit + position

        self.sell(price, to_sell)

class Trader:
    def __init__(self, long_threshold: int, short_threshold: int) -> None:
        limits = {
            "CHOCOLATE": 250,
            "STRAWBERRIES": 350,
            "ROSES": 60,
            "GIFT_BASKET": 60,
        }

        self.strategies: dict[Symbol, Strategy] = {symbol: clazz(symbol, limits[symbol], long_threshold, short_threshold) for symbol, clazz in {
            "CHOCOLATE": GiftBasketStrategy,
            "STRAWBERRIES": GiftBasketStrategy,
            "ROSES": GiftBasketStrategy,
            "GIFT_BASKET": GiftBasketStrategy,
        }.items()}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0
        trader_data = ""

        for symbol, strategy in self.strategies.items():
            if symbol in state.order_depths:
                strategy_orders, strategy_conversions = strategy.run(state)
                orders[symbol] = strategy_orders
                conversions += strategy_conversions

        return orders, conversions, trader_data

def run(long_threshold: float, short_threshold: float) -> dict[str, float]:
    file_reader = PackageResourcesReader()

    out = {
        "long_threshold": long_threshold,
        "short_threshold": short_threshold,
        "total_pnl": 0,
        "total_CHOCOLATE_pnl": 0,
        "total_STRAWBERRIES_pnl": 0,
        "total_ROSES_pnl": 0,
        "total_GIFT_BASKET_pnl": 0,
    }

    for product in ["CHOCOLATE", "STRAWBERRIES", "ROSES", "GIFT_BASKET"]:
        out[f"total_{product}_pnl"] = 0
        out[f"{product}_min"] = 1e9
        out[f"{product}_max"] = -1e9

    for day_num in range(3):
        trader = Trader(long_threshold, short_threshold)
        data = read_day_data(file_reader, round_num=3, day_num=day_num)
        result = run_backtest(trader, data, print_output=False, disable_trades_matching=False, disable_progress_bar=True)

        out[f"day{day_num}_pnl"] = 0

        for product in ["CHOCOLATE", "STRAWBERRIES", "ROSES", "GIFT_BASKET"]:
            pnls = [row.columns[-1] for row in result.activity_logs if row.columns[2] == product]

            min_pnl = min(pnls)
            max_pnl = max(pnls)
            final_pnl = pnls[-1]

            out[f"day{day_num}_pnl"] += final_pnl
            out[f"day{day_num}_{product}_pnl"] = final_pnl

            out["total_pnl"] += final_pnl
            out[f"total_{product}_pnl"] += final_pnl

            out[f"day{day_num}_{product}_min"] = min_pnl
            out[f"day{day_num}_{product}_max"] = max_pnl

            out[f"{product}_min"] = min(out[f"{product}_min"], min_pnl)
            out[f"{product}_max"] = max(out[f"{product}_max"], max_pnl)

    return out

long_threshold_values = []
short_threshold_values = []

for long_threshold in range(100, 601, 5):
    for short_threshold in range(long_threshold + 5, 601, 5):
        long_threshold_values.append(long_threshold)
        short_threshold_values.append(short_threshold)

results = process_map(run, long_threshold_values, short_threshold_values, max_workers=12, chunksize=1, ascii=True)

output_file = Path(__file__).parent / f"{Path(__file__).stem}-long-short-threshold.json"
with output_file.open("w+", encoding="utf-8") as file:
    file.write(json.dumps(results, separators=(",", ":")))
