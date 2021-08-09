from datetime import datetime as dt

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Table, Column, Integer, Float, String, DateTime, Sequence, ForeignKey, Text
from sqlalchemy.orm import sessionmaker, relationship, backref

Base = declarative_base()
engine = create_engine('oracle://root:0000@localhost:1521/xe')
Session = sessionmaker(bind=engine)


class STKDB(Base):
    __table__ = Table('STKDB_REV5', Base.metadata, autoload=True, autoload_with=engine)


class KospiIndex(Base):
    __table__ = Table('KOSPI_INDEX', Base.metadata, autoload=True, autoload_with=engine)


class BacktestSummary(Base):
    __tablename__ = 'BACKTEST_SUMMARY'
    id_seq = Sequence('result_id_seq')
    id = Column(Integer, id_seq, primary_key=True)
    user_id = Column(String(10), nullable=False)
    backtest_time = Column(DateTime, nullable=False)
    market = Column(String(10), nullable=True)
    sector = Column(Text, nullable=True)
    take_profit = Column(Float, nullable=True)
    cut_loss = Column(Float, nullable=True)
    min_holding_period = Column(Integer, nullable=True)
    max_holding_count = Column(Integer, nullable=True)
    buy_condition = Column(String(100), nullable=True)
    sell_condition = Column(String(100), nullable=True)
    total_profit = Column(Float, nullable=False)
    html_path = Column(String(100), nullable=False)


class TransactionHistory(Base):
    __tablename__ = 'TRANSACTION_HISTORY'
    id_seq = Sequence('trading_id_seq')
    id = Column(Integer, id_seq, primary_key=True)
    user_id = Column(String(10), ForeignKey('USER_DETAIL.user_id'), nullable=False)
    trade_date = Column(DateTime, nullable=False)
    buy_or_sell = Column(String(10), nullable=False)
    isu_srt_cd = Column(String(10), nullable=False)
    isu_abbrv = Column(String(20), nullable=False)
    quantity = Column(Integer, nullable=False)
    buy_price = Column(Float, nullable=False)
    sell_price = Column(Float, nullable=True)

    user = relationship('UserDetail', backref=backref('transactions', order_by=id))


class HoldingStock(Base):
    __tablename__ = 'HOLDING_STOCK'
    id_seq = Sequence('holding_id_seq')
    id = Column(Integer, id_seq, primary_key=True)
    user_id = Column(String(10), ForeignKey('USER_DETAIL.user_id'), nullable=False)
    date = Column(DateTime, nullable=False)
    isu_srt_cd = Column(String(10), nullable=False)
    isu_abbrv = Column(String(20), nullable=False)
    quantity = Column(Integer, nullable=False)
    evaluation_price = Column(Integer, nullable=False)

    user = relationship('UserDetail', backref=backref('holdings', order_by=id))


class AssetHistory(Base):
    __tablename__ = 'ASSET_HISTORY'
    id_seq = Sequence('asset_id_seq')
    id = Column(Integer, id_seq, primary_key=True)
    user_id = Column(String(10), ForeignKey('USER_DETAIL.user_id'), nullable=False)
    date = Column(DateTime, nullable=False)
    total = Column(Integer, nullable=False)
    cash = Column(Integer, nullable=False)

    user = relationship('UserDetail', backref=backref('assets', order_by=id))


class UserDetail(Base):
    __tablename__ = 'USER_DETAIL'
    user_id = Column(String(10), primary_key=True)
    cash = Column(Integer, server_default='100000000')
    market = Column(String(10), server_default='ALL')
    sector = Column(Text, server_default='ALL')
    take_profit = Column(Float,  server_default='5.0')
    cut_loss = Column(Float,  server_default='3.0')
    min_holding_period = Column(Integer, server_default='7')
    max_holding_count = Column(Integer, server_default='10')
    buy_condition = Column(String(100), nullable=True)
    sell_condition = Column(String(100), nullable=True)
    last_updated = Column(DateTime, default=dt.strptime('20200701', '%Y%m%d'))


if __name__ == '__main__':
    Base.metadata.create_all(engine)
