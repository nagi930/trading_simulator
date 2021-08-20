import datetime
from datetime import datetime as dt
import pprint
from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd
from pandas import DataFrame
import numpy as np
import sqlalchemy
from database import STKDB, Session


@dataclass
class DataFrameConfig:
    """Represents DataFrame Configuration for market, sector and simulation date(start, end)"""
    market: str
    sector: str
    start: str
    end: str

    def set_dataframe(self) -> DataFrame:
        """Set and return DataFrame by configured setting"""
        with Session() as session:
            query = session.query(STKDB)
            query = query.filter((self.start <= STKDB.open_date) &
                                 (STKDB.open_date < self.end))
            if self.market != 'ALL':
                query = query.filter(STKDB.mkt_nm.like(f'%{self.market}%'))
            if self.sector != 'ALL' and isinstance(self.sector, Iterable):
                query = query.filter(
                    sqlalchemy.func.REGEXP_LIKE(STKDB.std_ind_cd, '|'.join(map(lambda x: f'..{x}..', self.sector)))
                )
            df = pd.read_sql(query.statement, query.session.bind)
            df.columns = [column.upper() for column in df.columns]
            df = df.set_index('OPEN_DATE')
        return df


@dataclass
class UserConfig:
    """Represents a User Configuration for cash, min_hold_period, max_hold_count, take_profit and cut_loss"""
    cash: int
    min_hold_period: int
    max_hold_count: int
    take_profit: float
    cut_loss: float


class TradingAlgorithm(list):
    """Represents a trading algorithm list"""
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return ', '.join(str(condition) for condition in self)


@dataclass
class TradingCondition:
    """Represents a Trading condition. Can be buy condition or sell condition"""
    column: str
    operator: str
    value: float

    def __str__(self):
        return f'{self.column} {self.operator} {self.value}'

    def __repr__(self):
        return f'({self.column} {self.operator} {self.value})'


def filter_query(df: DataFrame, algorithm: TradingAlgorithm, and_or: str = '&') -> DataFrame:
    """Return DataFrame filtered by Trading Algorithm"""
    comparisons = [condition for condition in algorithm if condition.operator in ('>=', '>', '==', '!=', '<=', '<')]
    if comparisons:
        df = df.query(f'{and_or}'.join(map(repr, comparisons)))
    sorts = [condition for condition in algorithm if condition.operator == 'sort']
    if sorts:
        df = df.sort_values(
            by=[condition.column for condition in sorts],
            ascending=[condition.value for condition in sorts]
        )
    return df


class Simulator:
    """Trading simulator computed by user setting"""
    def __init__(self, df: DataFrame, user_config: UserConfig,
                 sell_algorithm: TradingAlgorithm, buy_algorithm: TradingAlgorithm, init_holdings=None):
        self.df = df
        self.user_config = user_config
        self.sell_algorithm = sell_algorithm
        self.buy_algorithm = buy_algorithm
        self._result = {}
        self.profits = []
        self._holdings = init_holdings or []
        self.dates = [pd.to_datetime(str(date)).strftime('%Y-%m-%d') for date in dataframe.index.unique()]
        self.clsprc = self.df.groupby(['OPEN_DATE', 'ISU_SRT_CD'])['TDD_CLSPRC'].mean().unstack()
        self.clsprc.index = self.clsprc.index.strftime("%Y-%m-%d")

    @property
    def result(self) -> dict:
        """Return trading result"""
        return {date: [trading.to_dict for trading in tradings]for date, tradings in self._result.items()}

    @property
    def cash_per_count(self) -> float:
        """Return amount that user can buy"""
        return self.user_config.cash // (self.user_config.max_hold_count - len(self._holdings))

    @property
    def holding_stock_code(self) -> tuple:
        """Return name array(tuple) of holding stocks"""
        return tuple(stock.code for stock in self._holdings)

    def holding_value(self, date: str) -> float:
        """Return total amount of holding stocks"""
        return sum([stock.quantity * self.clsprc.at[date, stock.code] for stock in self._holdings])

    def run(self) -> None:
        """Run simulation"""
        sell_temp = filter_query(self.df, self.sell_algorithm, and_or='|')
        buy_temp = filter_query(self.df, self.buy_algorithm, and_or='&')
        for idx, date in enumerate(self.dates):
            self._result[date] = []
            self.cut_trade(date)
            df_by_date = sell_temp.query(f'OPEN_DATE == "{date}"')
            sells = df_by_date.query(f'ISU_SRT_CD in {self.holding_stock_code}')[['ISU_SRT_CD', 'ISU_ABBRV', 'TDD_CLSPRC']].values.tolist()
            buys = buy_temp.query(f'OPEN_DATE == "{date}"')[['ISU_SRT_CD', 'ISU_ABBRV', 'TDD_CLSPRC']].values.tolist()

            self.trade(date, sells, buys)
            self.profits.append((self.user_config.cash + self.holding_value(date), self.user_config.cash))

    def cut_trade(self, date: str) -> None:
        """Sell in holding stocks if condition set by user is satisfied"""
        for stock in self._holdings[:]:
            try:
                clsprc = self.clsprc.at[date, stock.code]
            except KeyError:
                clsprc = stock.evaluation_price
                self.user_config.cash += (clsprc * stock.quantity)
                self._result[date].append(
                    TradingInfo(date, stock.code, stock.name, stock.quantity, stock.evaluation_price, clsprc)
                )
                self._holdings.remove(stock)
                continue
            if (clsprc >= stock.evaluation_price * (self.user_config.take_profit + 100) / 100)\
                or (clsprc <= stock.evaluation_price * (100 - self.user_config.cut_loss) / 100)\
                    or (dt.strptime(stock.date, '%Y-%m-%d') + datetime.timedelta(self.user_config.min_hold_period) <= dt.strptime(date, '%Y-%m-%d')):
                if np.isnan(clsprc):
                    clsprc = stock.evaluation_price
                self.user_config.cash += (clsprc * stock.quantity)
                self._result[date].append(
                    TradingInfo(date, stock.code, stock.name, stock.quantity, stock.evaluation_price, clsprc)
                )
                self._holdings.remove(stock)
                continue

    def trade(self, date: str, sells: list, buys: list) -> None:
        """Sell or buy stocks if condition set by user is satisfied"""
        if self.sell_algorithm:
            for code, name, price in sells:
                for stock in self._holdings[:]:
                    if stock.code == code:
                        self.user_config.cash += (price * stock.quantity)
                        self._result[date].append(
                            TradingInfo(date, code, name, stock.quantity, stock.evaluation_price, price)
                        )
                        self._holdings.remove(stock)

        for code, name, price in buys:
            if self.user_config.max_hold_count == len(self._holdings):
                return

            quantity = self.cash_per_count // price
            if quantity < 1 or quantity == np.nan:
                break
            self.user_config.cash -= (price * quantity)
            self._result[date].append(TradingInfo(date, code, name, quantity, price))

            for idx, stock in enumerate(self._holdings):
                if stock.code == code:
                    amounts = self._holdings[idx].amount + (price * quantity)
                    quantities = self._holdings[idx].quantity + quantity
                    self._holdings[idx].evaluation_price = amounts / quantities
                    self._holdings[idx].quantity += quantity
                    break
            else:
                self._holdings.append(TradingInfo(date, code, name, quantity, price))


@dataclass
class TradingInfo:
    """Represents a trading information about date, what to trade, quantity and price."""
    date: str
    code: str
    name: str
    quantity: int
    buy_price: int
    evaluation_price: int = field(init=False)
    sell_price: int = None

    def __post_init__(self):
        self.evaluation_price = self.buy_price

    @property
    def amount(self):
        """Return an amount of the trade stock"""
        return self.evaluation_price * self.quantity

    @property
    def to_dict(self):
        """Return dictionary from object"""
        return {
            'date': self.date,
            'buy_or_sell': 'buy' if self.sell_price is None else 'sell',
            'code': self.code,
            'name': self.name,
            'quantity': self.quantity,
            'buy_price': self.buy_price,
            'evaluation_price': self.evaluation_price,
            'sell_price': self.sell_price,
            }

    def __repr__(self):
        return f'종목코드: {self.code} 종목명: {self.name} 수량: {self.quantity} 매수가: {self.buy_price} ' \
               f'평가가: {self.evaluation_price} 매도가: {self.sell_price}'


if __name__ == '__main__':
    dataframe_config = DataFrameConfig('ALL', 'ALL', '20200701', '20210701')
    dataframe = dataframe_config.set_dataframe()

    uc = UserConfig(10000000, 10, 10, 7, 5)

    ba = TradingAlgorithm()
    ba.append(TradingCondition('RSI_6', '<=', 30))
    ba.append(TradingCondition('PBR', '<=', 1))
    ba.append(TradingCondition('ACC_TRDVOL', 'sort', False))

    sa = TradingAlgorithm()
    sa.append(TradingCondition('RSI_6', '>=', 80))

    simulator = Simulator(df=dataframe, user_config=uc,
                          sell_algorithm=sa, buy_algorithm=ba)
    simulator.run()

    pprint.pprint(simulator.result)
    print(simulator.profits)





