import datetime
from datetime import datetime as dt
from collections.abc import Iterable

import pandas as pd
import numpy as np
import sqlalchemy
from database import STKDB, Session


def filter_query(df, algorithm, and_or='&'):
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


class DataFrameConfig:
    def __init__(self, market, sector, start, end):
        self.market = market
        self.sector = sector
        self.start = start
        self.end = end
        self.df = None

    def get_dataframe(self):
        with Session() as session:
            query = session.query(STKDB)
            query = query.filter((self.start <= STKDB.open_date) &
                                 (STKDB.open_date < self.end))
            if self.market != 'ALL':
                query = query.filter(STKDB.mkt_nm.like(f'%{self.market}%'))
            if self.sector != 'ALL' and isinstance(self.sector, Iterable):
                query = query.filter(sqlalchemy.func.REGEXP_LIKE(STKDB.std_ind_cd, '|'.join(map(lambda x: f'..{x}..', self.sector))))
            df = pd.read_sql(query.statement, query.session.bind)
            df.columns = [column.upper() for column in df.columns]
            self.df = df.set_index('OPEN_DATE')
        return self.df


class CustomerConfig:
    def __init__(self, cash, min_hold_period, max_hold_count, take_profit, cut_loss):
        self.cash = cash
        self.min_hold_period = min_hold_period
        self.max_hold_count = max_hold_count
        self.take_profit = take_profit
        self.cut_loss = cut_loss


class TradingAlgorithm(list):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return ', '.join(str(condition) for condition in self)


class TradingCondition:
    def __init__(self, column, operator, value):
        self.column = column
        self.operator = operator
        self.value = value

    def __str__(self):
        return f'{self.column} {self.operator} {self.value}'

    def __repr__(self):
        return f'({self.column} {self.operator} {self.value})'


class Simulator:
    def __init__(self, dataframe, customer_config, sell_algorithm, buy_algorithm, init_holdings=None):
        self.dataframe = dataframe
        self.customer_config = customer_config
        self.sell_algorithm = sell_algorithm
        self.buy_algorithm = buy_algorithm
        self._result = {}
        self.profits = []
        self.cash = customer_config.cash
        self._holdings = init_holdings or []
        self.dates = [pd.to_datetime(str(date)).strftime('%Y-%m-%d') for date in dataframe.index.unique()]
        self.clsprc = self.dataframe.groupby(['OPEN_DATE', 'ISU_SRT_CD'])['TDD_CLSPRC'].mean().unstack()
        self.clsprc.index = self.clsprc.index.strftime("%Y-%m-%d")

    @property
    def holdings(self):
        return [holding.to_dict for holding in self._holdings]

    @property
    def result(self):
        return {date: [trading.to_dict for trading in tradings]for date, tradings in self._result.items()}

    @property
    def cash_per_count(self):
        return self.cash // (self.customer_config.max_hold_count - len(self._holdings))

    @property
    def holding_stock_code(self):
        return tuple(stock.code for stock in self._holdings)

    def holding_value(self, date):
        return sum([stock.quantity * self.clsprc.at[date, stock.code] for stock in self._holdings])

    def run(self):
        sell_temp = filter_query(self.dataframe, self.sell_algorithm, and_or='|')
        buy_temp = filter_query(self.dataframe, self.buy_algorithm, and_or='&')
        for idx, date in enumerate(self.dates):
            self._result[date] = []
            self.cut_trade(date)
            df_by_date = sell_temp.query(f'OPEN_DATE == "{date}"')
            sells = df_by_date.query(f'ISU_SRT_CD in {self.holding_stock_code}')[['ISU_SRT_CD', 'ISU_ABBRV', 'TDD_CLSPRC']].values.tolist()
            buys = buy_temp.query(f'OPEN_DATE == "{date}"')[['ISU_SRT_CD', 'ISU_ABBRV', 'TDD_CLSPRC']].values.tolist()

            self.trade(date, sells, buys)
            self.profits.append((self.cash + self.holding_value(date), self.cash))

    def cut_trade(self, date):
        for stock in self._holdings[:]:
            try:
                clsprc = self.clsprc.at[date, stock.code]
            except KeyError:
                clsprc = stock.evaluation_price
                self.cash += (clsprc * stock.quantity)
                self._result[date].append(TradingInfo(date, stock.code, stock.name, stock.quantity, stock.evaluation_price, clsprc))
                self._holdings.remove(stock)
                continue
            if (clsprc >= stock.evaluation_price * (self.customer_config.take_profit + 100) / 100)\
                or (clsprc <= stock.evaluation_price * (100 - self.customer_config.cut_loss) / 100)\
                    or (dt.strptime(stock.date, '%Y-%m-%d') + datetime.timedelta(self.customer_config.min_hold_period) <= dt.strptime(date, '%Y-%m-%d')):
                if np.isnan(clsprc):
                    clsprc = stock.evaluation_price
                self.cash += (clsprc * stock.quantity)
                self._result[date].append(TradingInfo(date, stock.code, stock.name, stock.quantity, stock.evaluation_price, clsprc))
                self._holdings.remove(stock)
                continue

    def trade(self, date, sells, buys):
        if self.sell_algorithm:
            for code, name, price in sells:
                for stock in self._holdings[:]:
                    if stock.code == code:
                        self.cash += (price * stock.quantity)
                        self._result[date].append(TradingInfo(date, code, name, stock.quantity, stock.evaluation_price, price))
                        self._holdings.remove(stock)

        for code, name, price in buys:
            if self.customer_config.max_hold_count == len(self._holdings):
                return

            quantity = self.cash_per_count // price
            if quantity < 1 or quantity == np.nan:
                break
            self.cash -= (price * quantity)
            self._result[date].append(TradingInfo(date, code, name, quantity, price, None))

            for idx, stock in enumerate(self._holdings):
                if stock.code == code:
                    self._holdings[idx].evaluation_price = (self._holdings[idx].amount + (price * quantity)) / (self._holdings[idx].quantity + quantity)
                    self._holdings[idx].quantity += quantity
                    break
            else:
                self._holdings.append(TradingInfo(date, code, name, quantity, price, None))


class TradingInfo:
    def __init__(self, date, code, name, quantity, buy_price, sell_price=None):
        self.date = date
        self.code = code
        self.name = name
        self.quantity = quantity
        self.buy_price = buy_price
        self.evaluation_price = buy_price
        self.sell_price = sell_price or None

    @property
    def amount(self):
        return self.evaluation_price * self.quantity

    @property
    def to_dict(self):
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
    df = dataframe_config.get_dataframe()

    customer_config = CustomerConfig(10000000, 10, 10, 7, 5)

    buy_algorithm = TradingAlgorithm()
    buy_algorithm.append(TradingCondition('RSI_6', '<=', 30))
    buy_algorithm.append(TradingCondition('PBR', '<=', 1))
    buy_algorithm.append(TradingCondition('ACC_TRDVOL', 'sort', False))

    sell_algorithm = TradingAlgorithm()
    sell_algorithm.append(TradingCondition('RSI_6', '>=', 80))

    simulator = Simulator(dataframe=df, customer_config=customer_config,
                          sell_algorithm=sell_algorithm, buy_algorithm=buy_algorithm)
    simulator.run()




