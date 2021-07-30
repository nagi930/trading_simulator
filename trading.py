from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from simulation import DataFrameConfig, CustomerConfig, TradingAlgorithm, TradingCondition, Simulator, QueryDataFrame, TradingInfo
from database import Session, TransactionHistory, HoldingStock, UserDetail
from sqlalchemy.orm import joinedload


def get_user_info(user_id):
    with Session() as session:
        user = session.query(UserDetail)\
            .options(joinedload(UserDetail.transactions))\
            .options(joinedload(UserDetail.holdings))\
            .filter_by(user_id=user_id).first()
    return user


def simulate_future(user_info, sdate, edate):
    dataframe_config = DataFrameConfig(user_info.market, user_info.sector, sdate, edate)
    df = QueryDataFrame(dataframe_config).get_dataframe()

    holdings = [TradingInfo(
        date=dt.strftime(holding.date, '%Y-%m-%d'),
        code=holding.isu_srt_cd,
        name=holding.isu_abbrv,
        quantity=holding.quantity,
        buy_price=holding.evaluation_price,
        sell_price=None
    ) for holding in user_info.holdings]

    cc = CustomerConfig(
        cash=user_info.cash,
        take_profit=user_info.take_profit,
        cut_loss=user_info.cut_loss,
        min_hold_period=user_info.min_holding_period,
        max_hold_count=user_info.max_holding_count
    )

    buy_algorithm = TradingAlgorithm()
    if user_info.buy_condition:
        conditions = user_info.buy_condition.split(', ')
        for condition in conditions:
            column, operator, value = condition.split()
            value = bool(value) if operator == 'sort' else int(value)
            buy_algorithm.append(TradingCondition(column, operator, value))

    sell_algorithm = TradingAlgorithm()
    if user_info.buy_condition:
        conditions = user_info.buy_condition.split(', ')
        for condition in conditions:
            column, operator, value = condition.split()
            value = bool(value) if operator == 'sort' else int(value)
            sell_algorithm.append(TradingCondition(column, operator, value))

    simulator = Simulator(
        dataframe=df,
        customer_config=cc,
        sell_algorithm=sell_algorithm,
        buy_algorithm=buy_algorithm,
        init_holdings=holdings
    )
    simulator.run()
    return simulator


def update_user_info(user_info, simulator, date):
    with Session.begin() as session:
        session.query(UserDetail).filter(UserDetail.user_id == user_info.user_id)\
            .update({'cash': simulator.cash, 'last_updated': date})
        session.commit()


def update_trading_history(user_info, simulator):
    with Session.begin() as session:
        for date, tradings in simulator.result.items():
            for trading in tradings:
                buy_or_sell = trading['buy_or_sell']
                isu_srt_cd = trading['code']
                isu_abbrv = trading['name']
                quantity = int(trading['quantity'])
                buy_price = int(trading['buy_price'])
                sell_price = trading['sell_price']
                th = TransactionHistory(
                    user_id=user_info.user_id,
                    trade_date=date,
                    buy_or_sell=buy_or_sell,
                    isu_srt_cd=isu_srt_cd,
                    isu_abbrv=isu_abbrv,
                    quantity=quantity,
                    buy_price=buy_price,
                    sell_price=sell_price
                )
                session.add(th)
        session.commit()


def update_holdings(user_info, simulator):
    with Session.begin() as session:
        holdings = session.query(HoldingStock).filter(HoldingStock.user_id == user_info.user_id)
        holdings.delete(synchronize_session=False)
        for holding in simulator.holdings:
            date = holding['date']
            isu_srt_cd = holding['code']
            isu_abbrv = holding['name']
            quantity = int(holding['quantity'])
            evaluation_price = int(holding['evaluation_price'])
            holding_stock = HoldingStock(
                user_id=user_info.user_id,
                date=date,
                isu_srt_cd=isu_srt_cd,
                isu_abbrv=isu_abbrv,
                quantity=quantity,
                evaluation_price=evaluation_price
            )
            session.add(holding_stock)
        session.commit()


if __name__ == '__main__':
    user_id = '김기남3'

    with Session.begin() as session:
        sdate = session.query(UserDetail).filter(UserDetail.user_id == user_id).first().last_updated
    months = 1
    sdate = dt.strftime(sdate, '%Y-%m-%d')
    edate = dt.strftime(dt.strptime(sdate, '%Y-%m-%d') + relativedelta(months=months), '%Y-%m-%d')

    user_info = get_user_info(user_id)
    simulator = simulate_future(user_info, sdate, edate)

    update_user_info(user_info, simulator, edate)
    update_trading_history(user_info, simulator)
    update_holdings(user_info, simulator)
