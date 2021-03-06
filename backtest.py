import datetime

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jinja2 import Environment, FileSystemLoader
from database import Session, BacktestSummary, KospiIndex



def get_kospi_index(dataframe_config):
    with Session() as session:
        kospi_index = session.query(KospiIndex).filter((dataframe_config.start <= KospiIndex.open_date)
                                                  & (KospiIndex.open_date < dataframe_config.end)).all()
    kospi_date = [item.open_date for item in kospi_index]
    kospi_clsprc = [item.tdd_clsprc for item in kospi_index]
    return kospi_date, kospi_clsprc


def save_backtest_result(rth, user_id, market, sector):
    if isinstance(sector, list):
        sector = ','.join(sector)
    with Session() as session:
        result = BacktestSummary(user_id=user_id,
                                 backtest_time=rth.now,
                                 market=market,
                                 sector=sector,
                                 take_profit=rth.simulator.customer_config.take_profit,
                                 cut_loss=rth.simulator.customer_config.cut_loss,
                                 min_holding_period=rth.simulator.customer_config.min_hold_period,
                                 max_holding_count=rth.simulator.customer_config.max_hold_count,
                                 buy_condition=str(rth.simulator.buy_algorithm),
                                 sell_condition=str(rth.simulator.sell_algorithm),
                                 total_profit=rth.total_profit,
                                 html_path=f'./templates/html/{rth.fname}.html')
        session.add(result)
        session.commit()


def backfill(iter):
    tmp = []
    for idx, value in enumerate(iter):
        if np.isnan(value):
            tmp.append(tmp[-1])
        else:
            tmp.append(value)
    return tmp


class ResultToHtml:
    def __init__(self, simulator):
        self.simulator = simulator
        self.profit_change = []
        self.cash_change = []
        self.total_profit = 0
        self.daily_profit = 0
        self.rendered_html = None
        self.kospi_date = []
        self.kospi_clsprc = []
        self.now = None
        self.fname = None

    def setup(self, dataframe_config):
        self.profit_change = list(zip(*self.simulator.profits))[0]
        self.profit_change = backfill(self.profit_change)
        self.cash_change = list(zip(*self.simulator.profits))[1]
        self.total_profit = (self.profit_change[-1] - self.profit_change[0]) / self.profit_change[0] * 100
        self.daily_profit = self.total_profit / len(self.profit_change)
        self.kospi_date, self.kospi_clsprc = get_kospi_index(dataframe_config)

    def plot_profit(self):
        fig = make_subplots(specs=[[{'secondary_y': True}]])
        fig.update_layout(yaxis=dict(tickformat=',.0f'), xaxis=dict(tickformat='%y/%m/%d'), template='ggplot2',
                           title='?????????', width=1200, height=500, margin=dict(l=20, r=20, t=50, b=20))
        fig.add_trace(
            go.Scatter(x=self.simulator.dates, y=self.profit_change, mode='lines',
                       name='asset', line=dict(color='red', width=1.5)), secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=self.simulator.dates, y=self.cash_change, mode='lines',
                       name='cash', line=dict(color='blue', width=1.5)), secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=self.kospi_date, y=self.kospi_clsprc, mode='lines',
                       name='kospi ??????', line=dict(color='brown', width=1)), secondary_y=True
        )
        fig.add_hline(y=self.profit_change[0], line_width=2, line_color="green",
                      annotation_text=f'{self.profit_change[0]}', annotation_position='top left')
        fig.add_hline(y=self.profit_change[-1], line_width=2, line_color="orange",
                      annotation_text=f'{self.profit_change[-1]}', annotation_position='top right')
        profit_graph = fig.to_html(full_html=False)
        return profit_graph

    def render_html(self):
        profit_graph = self.plot_profit()
        tradings = self.simulator.result
        backtest_start = self.simulator.dates[0]
        backtest_end = self.simulator.dates[-1]
        buy_condition = self.simulator.buy_algorithm
        sell_condition = self.simulator.sell_algorithm
        env = Environment(loader=FileSystemLoader('templates'))
        template = env.get_template('result_template.html')
        self.rendered_html = template.render(tradings=tradings, profit_graph=profit_graph,
                                             total_profit=self.total_profit, daily_profit=self.daily_profit,
                                             start=backtest_start, end=backtest_end,
                                             buy_condition=buy_condition, sell_condition=sell_condition)

    def save_html(self, user_id):
        self.now = datetime.datetime.now()
        self.fname = user_id + '_' + datetime.datetime.strftime(self.now, "%Y%m%d%H%M%S")
        with open(f'./templates/html/{self.fname}.html', 'w', encoding='utf-8') as fh:
            fh.write(self.rendered_html)

        return self.fname