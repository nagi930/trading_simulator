from collections import defaultdict

from flask import Flask, render_template, session, request, redirect, url_for
import sqlalchemy as sa

from simulation import *
from backtest import *
from database import *
from trading import *


import base64
import crypto
import sys
sys.modules['Crypto'] = crypto
from crypto import Random
from crypto.Cipher import AES

app = Flask(__name__, static_folder="./static")


class AES128Crypto:
    def __init__(self, encrypt_key):
        self.BS = AES.block_size
        self.encrypt_key = encrypt_key[:16].encode(encoding='utf-8', errors='strict')
        self.pad = lambda s: bytes(s + (self.BS - len(s) % self.BS) * chr(self.BS - len(s) % self.BS), 'utf-8')
        self.unpad = lambda s: s[0:-ord(s[-1:])]

    def encrypt(self, raw):
        raw = self.pad(raw)
        iv = Random.new().read(self.BS)
        cipher = AES.new(self.encrypt_key, AES.MODE_CBC, iv)
        return base64.b64encode(iv + cipher.encrypt(raw)).decode("utf-8")

    def decrypt(self, enc):
        enc = base64.b64decode(enc)
        iv = enc[:self.BS]
        encrypted_msg = enc[self.BS:]
        cipher = AES.new(self.encrypt_key, AES.MODE_CBC, iv)
        return self.unpad(cipher.decrypt(encrypted_msg)).decode('utf-8')


key = [0x10, 0x01, 0x15, 0x1B, 0xA1, 0x11, 0x57, 0x72, 0x6C, 0x21, 0x56, 0x57, 0x62, 0x16, 0x05, 0x3D,
       0xFF, 0xFE, 0x11, 0x1B, 0x21, 0x31, 0x57, 0x72, 0x6B, 0x21, 0xA6, 0xA7, 0x6E, 0xE6, 0xE5, 0x3F]

app.secret_key = 'asdjfkldjslkfj7ewr8qew668'

oracle_engine = sa.create_engine('oracle://root:0000@localhost:1521/xe')


@app.route("/auth_register")
def auth_register():
    return render_template('auth_register.html')


@app.route("/slider")
def slider():
    return render_template('ui_sliders.html')


@app.route("/myasset")
def myasset():
    user_id = session['SESS_USERID']
    user_info = get_user_info(user_id)

    now_date = get_recent_opening_date(user_info.last_updated)
    holdings = user_info.holdings
    assets = user_info.assets
    with Session() as sess:
        for holding in holdings:
            result = sess.query(STKDB).filter((STKDB.open_date == now_date)
                                              & (STKDB.isu_srt_cd == holding.isu_srt_cd)).first()
            holding.now_price = result.tdd_clsprc if result else holding.evaluation_price
        transactions = sess.query(TransactionHistory).filter(TransactionHistory.user_id == user_id).all()

    abbrvs = [holding.isu_abbrv for holding in holdings]
    amounts = [holding.evaluation_price * holding.quantity for holding in holdings]

    asset_info = defaultdict(list)
    for asset in assets:
        asset_info['date'].append(asset.date)
        asset_info['total'].append(asset.total)
        asset_info['cash'].append(asset.cash)

    abbrvs.append('현금')
    amounts.append(asset_info['cash'][-1]) if asset_info['cash'] else amounts.append(user_info.cash)

    fig1 = make_subplots()
    fig1.add_trace(go.Scatter(x=asset_info['date'], y=asset_info['total'], name='전체자산'))
    fig1.add_trace(go.Scatter(x=asset_info['date'], y=asset_info['cash'], name='현금'))
    fig1.update_layout(yaxis=dict(tickformat=',.0f'),
                       xaxis=dict(tickformat='%y/%m/%d'),
                       height=500, width=800,
                       title_text='자산변동',
                       template='ggplot2')
    fig2 = go.Figure(data=[go.Pie(
        labels=abbrvs, values=amounts,
        textinfo='label+percent', insidetextorientation='horizontal'
    )])
    fig2.update_layout(height=500, width=800,
                       title_text='자산비중',
                       template='ggplot2')

    graph1 = fig1.to_html(full_html=False)
    graph2 = fig2.to_html(full_html=False)

    return render_template(
        'myasset.html',
        holdings=holdings,
        transactions=transactions,
        now_date=now_date.strftime('%Y-%m-%d'),
        graph1=graph1,
        graph2=graph2,
    )


@app.route("/myasset_transaction_rest")
def myasset_transaction_rest():
    sdate = request.args.get('sdate', type=str, default='')
    edate = request.args.get('edate', type=str, default='')

    sdate = dt.strptime(sdate, '%Y-%m-%d')
    edate = dt.strptime(edate, '%Y-%m-%d')

    user_id = session['SESS_USERID']

    with Session() as sess:
        queryset = sess.query(TransactionHistory) \
            .filter(TransactionHistory.user_id == user_id) \
            .filter((TransactionHistory.trade_date >= sdate) \
                    & (TransactionHistory.trade_date <= edate))

    df = pd.read_sql(queryset.statement, queryset.session.bind)
    df = df.sort_values(by='trade_date', ascending=False)
    transactions = json.loads(df.to_json(orient='records', force_ascii=False, date_format='iso'))
    transactions_str = json.dumps(transactions)

    return transactions_str


@app.route("/auth_register_proc", methods=['POST'])
def auth_register_proc():
    userid = request.form['userid']
    usernm = request.form['usernm']
    userpw = request.form['userpw']
    email = request.form['email']
    with oracle_engine.connect() as conn:
        trans = conn.begin()
        try:
            enc_pw = AES128Crypto(str(key)).encrypt(userpw)
            sql = "insert into customer(userid, usernm, userpw, email) values (:1, :2, :3, :4)"
            conn.execute(sql, (userid, usernm, enc_pw, email))
            trans.commit()
        except Exception as e:
            trans.rollback()
            print(e)
            return render_template('error.html')
    with Session() as sess:
        user = UserDetail(user_id=userid,
                          buy_condition=None,
                          sell_condition=None)
        sess.add(user)
        sess.commit()

    return redirect(url_for('auth_login'))


@app.route("/auth_login")
def auth_login():
    return render_template('auth_login.html')


@app.route("/auth_login_proc", methods=['POST'])
def auth_login_proc():
    userid = request.form['userid']
    userpw = request.form['userpw']
    with oracle_engine.connect() as conn:
        try:
            sql = "select * from customer where userid=:1 "
            result = conn.execute(sql, (userid)).fetchone()

            if len(result['usernm']) > 0:
                dec_pw = AES128Crypto(str(key)).decrypt(result['userpw'])

                if userpw == dec_pw:
                    session['SESS_LOGIN_STATUS'] = True
                    session['SESS_USERNM'] = result['usernm']
                    session['SESS_USERID'] = userid
                else:
                    return render_template('error.html')
            else:
                return render_template('error.html')
        except Exception as e:
            print(e)
            return render_template('error.html')
    return redirect(url_for('index'))


@app.route("/auth_logout")
def auth_logout():
    session['SESS_LOGIN_STATUS'] = False
    session.clear()
    return redirect(url_for('index'))

# ----------------------------------------------------------
@app.route("/kakao")
def kakao():
    return render_template('kakao.html')


@app.route("/backtest_form_value", methods=['POST'])
def backtest_form_value():
    market = request.form['market']
    sectors_list = request.form.getlist('sector')
    se_date = request.form['se_date']
    start, end = se_date.split(' - ')
    start = dt.strftime(dt.strptime(start, '%m/%d/%Y'), '%Y-%m-%d')
    end = dt.strftime(dt.strptime(end, '%m/%d/%Y'), '%Y-%m-%d')

    take_profit = request.form['take_profit']
    cut_loss = request.form['cut_loss']
    cash = request.form['cash']
    min_holding_period = request.form['min_holding_period']
    max_holding_count = request.form['max_holding_count']

    buy_ = request.form['buy_']
    sell_ = request.form['sell_']

    buy_conditions = [(condition['buy_col'], condition['buy_op'], condition['buy_val']) for condition in json.loads(buy_)]
    sell_conditions = [(condition['sell_col'], condition['sell_op'], condition['sell_val']) for condition in json.loads(sell_)]
    buy_order_col_ = request.form['buy_order_col_']
    buy_order_ = request.form['buy_order_']

    sector = []
    for sectors in sectors_list:
        sector += sectors.split(',')

    if sector[0] == 'ALL':
        sector = 'ALL'

    dataframe_config = DataFrameConfig(market, sector, start, end)
    df = QueryDataFrame(dataframe_config).get_dataframe()
    customer_config = CustomerConfig(int(cash), int(min_holding_period), int(max_holding_count), float(take_profit), float(cut_loss))

    buy_algorithm = TradingAlgorithm()
    buy_order_ = bool(int(buy_order_))
    buy_algorithm.append(TradingCondition(buy_order_col_, 'sort', buy_order_))
    for col, op, val in buy_conditions:
        buy_algorithm.append(TradingCondition(col, op, int(val)))

    sell_algorithm = TradingAlgorithm()
    for col, op, val in sell_conditions:
        sell_algorithm.append(TradingCondition(col, op, int(val)))

    simulator = Simulator(dataframe=df, customer_config=customer_config,
                          sell_algorithm=sell_algorithm, buy_algorithm=buy_algorithm)
    simulator.run()
    user_id = session['SESS_USERID']

    rth = ResultToHtml(simulator)
    rth.setup(dataframe_config)
    rth.render_html()
    fname = rth.save_html(user_id=user_id)

    save_backtest_result(rth, user_id, market, sector)

    html_filename = 'html/' + fname + '.html'
    return render_template('result.html', KEY_HTML_REPORT=html_filename)


@app.route("/start", methods=['POST'])
def start():
    user_id = session['SESS_USERID']
    with Session.begin() as sess:
        sdate = sess.query(UserDetail).filter(UserDetail.user_id == user_id).first().last_updated
    months = 1
    sdate = dt.strftime(sdate, '%Y-%m-%d')
    edate = dt.strftime(dt.strptime(sdate, '%Y-%m-%d') + relativedelta(months=months), '%Y-%m-%d')

    user_info = get_user_info(user_id)
    simulator = simulate_future(user_info, sdate, edate)

    update_user_info(user_info, simulator, edate)
    update_trading_history(user_info, simulator)
    update_holdings(user_info, simulator)
    update_asset_change(user_info, simulator)
    return redirect(url_for('myasset'))


@app.route("/set_backtest_config", methods=['POST'])
def set_backtest_config():
    request_id = request.form['options']
    user_id = session['SESS_USERID']
    with Session() as sess:
        backtest = sess.query(BacktestSummary) \
            .filter((BacktestSummary.user_id == user_id) & (BacktestSummary.id == request_id)) \
            .first()
        sess.query(UserDetail) \
            .filter(UserDetail.user_id == user_id) \
            .update({
            'market': backtest.market,
            'sector': backtest.sector,
            'take_profit': backtest.take_profit,
            'cut_loss': backtest.cut_loss,
            'min_holding_period': backtest.min_holding_period,
            'max_holding_count': backtest.max_holding_count,
            'buy_condition': backtest.buy_condition,
            'sell_condition': backtest.sell_condition,
        })
        sess.commit()
    return redirect(url_for('backtests'))


@app.route("/rest_mystock_db")
def rest_stock_insert():
    code = request.args.get('code', type=str, default='')
    mode = request.args.get('mode', type=str, default='')

    if mode == "" or code == "" or session['SESS_USERID'] == "":
        return render_template('error.html')
    else:
        result_str = "ok"
        with oracle_engine.connect() as conn:
            trans = conn.begin()
            try:
                if mode == "insert":
                    sql = "insert into customer_stock(userid, code) values (:1, :2)"
                    conn.execute(sql, (session['SESS_USERID'], code))
                    trans.commit()

                elif mode == "delete":
                    sql = "delete from customer_stock where userid=:1 and code=:2"
                    conn.execute(sql, (session['SESS_USERID'], code))
                    trans.commit()

                    sql = f"""select distinct 종목명, 금일종가 as 종가, 등락률, 종목코드 as 티커,  nvl((Select Distinct Code From Customer_Stock Where Code = 종목코드 and userid='{session['SESS_USERID']}'),0)  As Selected
                                        from
                                        (
                                              select * from (
                                                  select tday.open_date 개장일, tday.isu_srt_cd 종목코드, tday.isu_abbrv 종목명, yday.tdd_clsprc 전일종가, tday.tdd_clsprc 금일종가, (tday.tdd_clsprc - yday.tdd_clsprc) / yday.tdd_clsprc * 100 등락률
                                                  from stkdb_rev3 tday,
                                                      (select isu_srt_cd, isu_abbrv, tdd_clsprc from stkdb_rev3  where open_date = '20210716') yday
                                                  where tday.open_date = '20210719'
                                                       and tday.isu_abbrv = yday.isu_abbrv
                                                       and tday.isu_srt_cd = yday.isu_srt_cd
                                                  Order By 등락률 Desc
                                              )
                                              where rownum <= 50
                                         ) S, customer_stock C
                                         where C.Code = S.종목코드
                    """
                    mystock_df = pd.read_sql_query(sql, conn).to_json(orient="values")
                    list_mystock = json.loads(mystock_df)
                    result_str = json.dumps(list_mystock)
                elif mode == "select":
                    sql = f"""select distinct 종목명, 금일종가 as 종가, 등락률, 종목코드 as 티커,  nvl((Select Distinct Code From Customer_Stock Where Code = 종목코드 and userid='{session['SESS_USERID']}'),0)  As Selected
                    from
                    (
                          select * from (
                              select tday.open_date 개장일, tday.isu_srt_cd 종목코드, tday.isu_abbrv 종목명, yday.tdd_clsprc 전일종가, tday.tdd_clsprc 금일종가, (tday.tdd_clsprc - yday.tdd_clsprc) / yday.tdd_clsprc * 100 등락률
                              from stkdb_rev3 tday,
                                  (select isu_srt_cd, isu_abbrv, tdd_clsprc from stkdb_rev3  where open_date = '20210716') yday
                              where tday.open_date = '20210719'
                                   and tday.isu_abbrv = yday.isu_abbrv
                                   and tday.isu_srt_cd = yday.isu_srt_cd
                              Order By 등락률 Desc
                          )
                          where rownum <= 50
                     ) S, customer_stock C
                     where C.Code = S.종목코드
                    """
                    mystock_df = pd.read_sql_query(sql, conn).to_json(orient="values")
                    list_mystock = json.loads(mystock_df)
                    result_str = json.dumps(list_mystock)
                else:
                    raise Exception('에러발생')

            except Exception as e:
                trans.rollback()
                print(e)
                return render_template('error.html')
    return result_str

@app.route("/rest_mystock_db")
def get_top50_by_db():
    with oracle_engine.connect() as conn:
        trans = conn.begin()
        try:
            if 'SESS_LOGIN_STATUS' in session:
                sql = f"""select 종목명, 금일종가 as 종가, 등락률, 종목코드 as 티커,  nvl((Select Distinct Code From Customer_Stock Where Code = 종목코드 and userid='{session['SESS_USERID']}'),0)  As Selected
                    from
                    (
                            select tday.open_date 개장일, tday.isu_srt_cd 종목코드, tday.isu_abbrv 종목명, yday.tdd_clsprc 전일종가, tday.tdd_clsprc 금일종가, (tday.tdd_clsprc - yday.tdd_clsprc) / yday.tdd_clsprc * 100 등락률
                            from stkdb_rev3 tday,
                                (select isu_srt_cd, isu_abbrv, tdd_clsprc from stkdb_rev3  where open_date = '20210716') yday
                            where tday.open_date = '20210719'
                                 and tday.isu_abbrv = yday.isu_abbrv
                                 and tday.isu_srt_cd = yday.isu_srt_cd
                            order by 등락률 desc
                     ) stkdb_rev3
                     where  rownum <= 50"""


            else:
                sql = """select 종목명, 금일종가 as 종가, 등락률, 종목코드 as 티커, '0' As Selected
                        from
                        (
                            select tday.open_date 개장일, tday.isu_srt_cd 종목코드, tday.isu_abbrv 종목명, yday.tdd_clsprc 전일종가, tday.tdd_clsprc 금일종가, (tday.tdd_clsprc - yday.tdd_clsprc) / yday.tdd_clsprc * 100 등락률
                            from stkdb_rev3 tday,
                                (select isu_srt_cd, isu_abbrv, tdd_clsprc from stkdb_rev3  where open_date = '20210716') yday
                            where tday.open_date = '20210719'
                                 and tday.isu_abbrv = yday.isu_abbrv
                                 and tday.isu_srt_cd = yday.isu_srt_cd
                            order by 등락률 desc
                         ) stkdb_rev3
                         where  rownum <= 50
                        """
            df_top50 = pd.read_sql_query(sql, conn)

        except Exception as e:
            trans.rollback()
            print(e)
            return render_template('error.html')
    return df_top50


def get_top50_by_db():
    with oracle_engine.connect() as conn:
        trans = conn.begin()
        try:
            if 'SESS_LOGIN_STATUS' in session:
                sql = f"""select 종목명, 금일종가 as 종가, 등락률, 종목코드 as 티커,  nvl((Select Distinct Code From Customer_Stock Where Code = 종목코드 and userid='{session['SESS_USERID']}'),0)  As Selected
                    from
                    (
                            select tday.open_date 개장일, tday.isu_srt_cd 종목코드, tday.isu_abbrv 종목명, yday.tdd_clsprc 전일종가, tday.tdd_clsprc 금일종가, (tday.tdd_clsprc - yday.tdd_clsprc) / yday.tdd_clsprc * 100 등락률
                            from stkdb_rev3 tday,
                                (select isu_srt_cd, isu_abbrv, tdd_clsprc from stkdb_rev3  where open_date = '20210716') yday
                            where tday.open_date = '20210719'
                                 and tday.isu_abbrv = yday.isu_abbrv
                                 and tday.isu_srt_cd = yday.isu_srt_cd
                            order by 등락률 desc
                     ) stkdb_rev3
                     where  rownum <= 50
                     """
            else:
                sql = """select 종목명, 금일종가 as 종가, 등락률, 종목코드 as 티커, '0' As Selected
                from

                        (
                            select tday.open_date 개장일, tday.isu_srt_cd 종목코드, tday.isu_abbrv 종목명, yday.tdd_clsprc 전일종가, tday.tdd_clsprc 금일종가, (tday.tdd_clsprc - yday.tdd_clsprc) / yday.tdd_clsprc * 100 등락률
                            from stkdb_rev3 tday,
                                (select isu_srt_cd, isu_abbrv, tdd_clsprc from stkdb_rev3  where open_date = '20210716') yday
                            where tday.open_date = '20210719'
                                 and tday.isu_abbrv = yday.isu_abbrv
                                 and tday.isu_srt_cd = yday.isu_srt_cd
                            order by 등락률 desc
                         ) stkdb_rev3
                         where  rownum <= 50
                        """

            df_top50 = pd.read_sql_query(sql, conn)
        except Exception as e:
            trans.rollback()
            print(e)
            return render_template('error.html')
    return df_top50


@app.route("/rest_top50_paging")
def rest_top50():
    page = request.args.get('page', type=int, default=1)
    page_per_count = 10
    if page > 1:
        start_page = (page - 1) * page_per_count
        end_page = page * page_per_count
    else:
        start_page = 0
        end_page = page * page_per_count
    df_top50 = get_top50_by_db()
    df_top50 = df_top50.iloc[start_page:end_page]

    list_top50 = df_top50.to_json(orient="values")
    list_top50 = json.loads(list_top50)
    list_top50_String = json.dumps(list_top50)
    return list_top50_String


@app.route("/rest_tap3")
def rest_tap3():
    dict_tab3 = {}
    code = request.args.get('code', type=str, default='005930')  # 종목코드

    with oracle_engine.connect() as conn:
        try:

            sql = "select info, invest from craw_info where code=:1"
            rows = conn.execute(sql, (code,))
            fetch_rows = rows.mappings().all()
            mystock_list = []
            for row in fetch_rows:
                mystock_list.append(row['info'])
                mystock_list.append(row['invest'])

            dict_tab3["html_tab1"] = mystock_list[0]
            dict_tab3["html_tab3"] = mystock_list[1]
            dict_chart = rest_tab3_chart(code)
            dict_tab3["html_tab3_chart"] = dict_chart

            sql = "select title,url,regdate from craw_naver_news where code=:1 order by regdate desc"
            rows = conn.execute(sql, (code,))
            fetch_rows = rows.mappings().all()

            html_tab2 = ""
            news_count = 0
            for row in fetch_rows:
                html_tab2 += "<tr><td>" + row['regdate'] + "</td>"
                html_tab2 += "<td><a href='" + row['url'] + "' target='_top'>" + row['title'] + "</a></td>"
                html_tab2 += "</tr>"
                news_count += 1
                if news_count > 10:
                    break

            dict_tab3["html_tab2"] = html_tab2
            dict_tab3["code"] = code
        except Exception as e:
            print(e)
            return "에러발생"
    return dict_tab3


def rest_tab3_chart(code='005930'):
    html_tab3_chart = ""
    html_tab3_chart += "<div class='tab-pane fade active show' id='subchart1'><img width='567' height='355' src='https://ssl.pstatic.net/imgfinance/chart/trader/month1/F_" + code + ".png' width='100%'></div>";
    html_tab3_chart += "<div class='tab-pane fade' id='subchart2'><img width='567' height='355'  src='https://ssl.pstatic.net/imgfinance/chart/trader/month3/F_" + code + ".png' width='100%'></div>";
    html_tab3_chart += "<div class='tab-pane fade' id='subchart3'><img width='567' height='355'  src='https://ssl.pstatic.net/imgfinance/chart/trader/month6/F_" + code + ".png' width='100%'></div>";
    html_tab3_chart += "<div class='tab-pane fade' id='subchart4'><img  width='567' height='355'  src='https://ssl.pstatic.net/imgfinance/chart/trader/year1/F_" + code + ".png' width='100%'></div>";
    html_tab3_chart += "<div class='tab-pane fade' id='subchart5'><img width='567' height='355' src='https://ssl.pstatic.net/imgfinance/chart/trader/month1/I_" + code + ".png' width='100%'></div>";
    html_tab3_chart += "<div class='tab-pane fade' id='subchart6'><img  width='567' height='355' src='https://ssl.pstatic.net/imgfinance/chart/trader/month3/I_" + code + ".png' width='100%'></div>";
    html_tab3_chart += "<div class='tab-pane fade' id='subchart7'><img  width='567' height='355'  src='https://ssl.pstatic.net/imgfinance/chart/trader/month6/I_" + code + ".png' width='100%'></div>";
    html_tab3_chart += "<div class='tab-pane fade' id='subchart8'><img  width='567' height='355'  src='https://ssl.pstatic.net/imgfinance/chart/trader/year1/I_" + code + ".png' width='100%'></div>";
    return html_tab3_chart


def index_data():
    today, yesterday, bf_yesterday = get_today_yesterday()
    idx_total_list = get_idx_total(yesterday, today)
    df_top50 = get_top50_by_db()
    return idx_total_list, df_top50


@app.route("/")
def index():
    idx_total_list, df_top50 = index_data()
    df_top50 = df_top50.iloc[:10]
    list_top50 = df_top50.to_json(orient="values")
    list_top50 = json.loads(list_top50)

    dict_tab3 = rest_tap3()
    html_tab3_chart = rest_tab3_chart()

    return render_template(
        'stock.html',
        KEY_TOTAL_IDX=idx_total_list,
        KEY_TOP50=list_top50,
        KEY_TAB3=dict_tab3,
        KEY_TAB3_CHART=html_tab3_chart
    )


@app.route("/main")
def main():
    return render_template('index.html')


@app.route("/result")
def result():
    return render_template('result.html')


@app.route("/backtests")
def backtests():
    with oracle_engine.connect() as conn:
        try:
            user_id = session['SESS_USERID']
            backtest_df = pd.read_sql_query(f"select * from BACKTEST_SUMMARY where USER_ID = '{user_id}'", conn).to_json(orient='values', date_format='iso')
            list_backtest_df = json.loads(backtest_df)

        except Exception as e:
            print(e)
            return render_template('error.html')

    return render_template('backresult.html',
                           KEY_BACKTEST=list_backtest_df,)

@app.route("/help")
def help():
    return render_template('help.html')


@app.route("/render_report")
def render_report():
    report_path = request.args.get('report_path', type=str, default="error.html")
    return render_template(report_path)


@app.route("/manager")
def manager():

    with oracle_engine.connect() as conn:
        try:
            user_df = pd.read_sql_query("select USERID,USERNM,USERPW,EMAIL from customer", conn).to_json(orient="values")
            list_user_df = json.loads(user_df)

            backtest_df = pd.read_sql_query("select * from BACKTEST_SUMMARY", conn).to_json(orient="values")
            list_backtest_df = json.loads(backtest_df)

            craw_info_df = pd.read_sql_query("select * from craw_info", conn).to_json(orient="values")
            list_craw_info_df = json.loads(craw_info_df)

            craw_news_df = pd.read_sql_query("select * from craw_naver_news", conn).to_json(orient="values")
            list_craw_news_df = json.loads(craw_news_df)

        except Exception as e:
            print(e)
            return render_template('error.html')
    return render_template('manager.html',
                           KEY_USER=list_user_df,
                           KEY_BACKTEST=list_backtest_df,
                           KEY_CRAW_INFO=list_craw_info_df,
                           KEY_CRAW_NEWS=list_craw_news_df,
                           )


if __name__ == '__main__':
    app.run(debug=True, port=8888)
