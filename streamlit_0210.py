# 초기 화면에서는 최근 6시간의 차트를 보여줍니다
# 하지만 24시간치 데이터를 모두 가지고 있어서 드래그하면 과거 데이터도 볼 수 있습니다
# 차트를 드래그하면 6시간 이전의 데이터도 자연스럽게 표시됩니다
# 거래현황을 표로 변경
# streamlit_app.py
import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pyupbit
import os
from dotenv import load_dotenv
import time

# .env 파일 로드
load_dotenv()

def get_connection():
    """데이터베이스 연결"""
    try:
        return sqlite3.connect('trading.db', timeout=30)  # 타임아웃 추가
    except Exception as e:
        st.error(f"데이터베이스 연결 오류: {e}")
        return None

def calculate_macd(df):
    """MACD(12,26,9) 계산"""
    try:
        if df is None or df.empty or len(df) < 26:
            return None, None

        # EMA 계산
        ema12 = df['close'].ewm(span=12, min_periods=1, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, min_periods=1, adjust=False).mean()
        
        # MACD 라인
        macd = ema12 - ema26
        
        # Signal 라인
        signal = macd.ewm(span=9, min_periods=1, adjust=False).mean()
        
        # NaN 값 처리
        macd = macd.fillna(0)
        signal = signal.fillna(0)
        
        return macd, signal
    except Exception as e:
        st.error(f"MACD 계산 오류: {e}")
        return None, None

def format_number(value):
    """숫자 포맷팅 함수"""
    try:
        if value is None:
            return "0"
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return "0"

def get_current_btc_price():
    """현재 BTC 가격 조회"""
    try:
        return pyupbit.get_current_price("KRW-BTC")
    except Exception as e:
        st.error(f"가격 조회 오류: {e}")
        return None

def format_price_with_color(price, direction=None):
    """가격에 색상을 적용하여 표시"""
    try:
        if price is None:
            return "0원"
            
        formatted_price = format_number(price)
        if direction == "up":
            return f"<span style='color:green'>▲ {formatted_price}원</span>"
        elif direction == "down":
            return f"<span style='color:red'>▼ {formatted_price}원</span>"
        else:
            return f"{formatted_price}원"
    except Exception:
        return "0원"

def format_datetime(dt):
    """날짜와 시간을 포맷팅"""
    try:
        return dt.strftime("%m-%d %H:%M:%S")
    except Exception:
        return "날짜 오류"

def get_current_balance(conn):
    """현재 잔고 계산"""
    try:
        last_trade = pd.read_sql("""
            SELECT type, price, amount, balance, profit
            FROM trades 
            ORDER BY timestamp DESC LIMIT 1
        """, conn)
        
        if not last_trade.empty:
            if last_trade.iloc[0]['type'] == 'BUY':
                return float(last_trade.iloc[0]['balance'])
            else:
                sell_balance = last_trade.iloc[0]['balance']
                profit = last_trade.iloc[0]['profit']
                return float(sell_balance + profit)
    except Exception as e:
        st.warning(f"잔고 조회 오류: {e}")
    return 10000000.0

def render_dashboard():
    """대시보드 렌더링"""
    try:
        current_price = get_current_btc_price()
        
        # 상단 정보 표시 영역
        col_info1, col_info2, col_info3 = st.columns([1, 1, 1])
        
        with get_connection() as conn:
            if conn is None:
                st.error("데이터베이스에 연결할 수 없습니다.")
                return
                
            # MACD 정보
            try:
                df_current = pd.read_sql("""
                    SELECT close FROM price_data 
                    ORDER BY timestamp DESC LIMIT 30
                """, conn)
                
                if not df_current.empty:
                    macd, signal = calculate_macd(df_current.iloc[::-1])
                    if macd is not None and signal is not None:
                        macd_value = float(macd.iloc[-1])
                        signal_value = float(signal.iloc[-1])
                        macd_diff = macd_value - signal_value
                        
                        with col_info1:
                            st.markdown("### MACD(12,26,9)")
                            st.metric("MACD", f"{macd_value:.2f}")
                            # st.metric("Signal", f"{signal_value:.2f}")
                            # st.metric("MACD-Signal", f"{macd_diff:.2f}")
                else:
                    with col_info1:
                        st.markdown("### MACD(12,26,9)")
                        st.info("가격 데이터가 충분하지 않습니다.")
            except Exception as e:
                with col_info1:
                    st.markdown("### MACD(12,26,9)")
                    st.error(f"MACD 계산 중 오류 발생: {str(e)}")

            # 가격 정보
            with col_info2:
                st.markdown("### BTC/KRW")
                if current_price:
                    if 'last_price' in st.session_state and st.session_state['last_price']:
                        price_direction = "up" if current_price > st.session_state['last_price'] else "down"
                    else:
                        price_direction = None
                    st.markdown(format_price_with_color(current_price, price_direction), unsafe_allow_html=True)
                    st.session_state['last_price'] = current_price
                else:
                    st.warning("현재가를 불러올 수 없습니다.")

            # 현재 포지션 정보
            with col_info3:
                try:
                    last_trade = pd.read_sql("""
                        SELECT * FROM trades 
                        ORDER BY timestamp DESC LIMIT 1
                    """, conn)

                    current_balance = get_current_balance(conn)

                    if not last_trade.empty:
                        trade_type = last_trade.iloc[0]['type']
                        
                        if trade_type == 'BUY':
                            entry_price = last_trade.iloc[0]['price']
                            entry_time = pd.to_datetime(last_trade.iloc[0]['timestamp'])
                            current_time = datetime.now()
                            hold_time = current_time - entry_time
                            hours = int(hold_time.total_seconds() // 3600)
                            minutes = int((hold_time.total_seconds() % 3600) // 60)
                            
                            amount = last_trade.iloc[0]['amount']
                            current_total = current_price * amount if current_price else 0
                            current_profit = ((current_price - entry_price) / entry_price * 100) if current_price else 0
                            
                            st.markdown("### 🟢 매수중")

                            # CSS로 테이블 스타일 적용
                            st.markdown("""
                                <style>
                                .custom-table {
                                    font-size: 1.1rem !important;
                                    width: 100% !important;
                                }
                                .custom-table th {
                                    background-color: #f0f2f6;
                                    padding: 12px !important;
                                }
                                .custom-table td {
                                    padding: 12px !important;
                                }
                                .highlight {
                                    font-weight: bold;
                                    font-size: 1.2rem;
                                }
                                </style>
                            """, unsafe_allow_html=True)

                            # 데이터프레임 생성
                            position_data = {
                                '진입시간': [format_datetime(entry_time)],
                                '진입가격': [f"{format_number(entry_price)}원"],
                                '현재가격': [f"{format_number(current_price)}원"],
                                '보유시간': [f"{hours}시간 {minutes}분"],
                                '수익률': [f"{current_profit:+.2f}%"],
                                '예상잔고': [f"{format_number(current_total)}원"],
                            }
                            df_position = pd.DataFrame(position_data)

                            # 스타일이 적용된 데이터프레임 표시
                            st.dataframe(
                                df_position,
                                hide_index=True,
                                use_container_width=True,  # 컨테이너 전체 너비 사용
                                column_config={
                                    "진입시간": st.column_config.Column(width=150),
                                    "진입가격": st.column_config.Column(width=150),
                                    "현재가격": st.column_config.Column(
                                        width=150,
                                        help="최신 거래 가격"
                                    ),
                                    "보유시간": st.column_config.Column(width=120),
                                    "수익률": st.column_config.Column(
                                        width=100,
                                        help="현재 수익률"
                                    ),
                                    "예상잔고": st.column_config.Column(
                                        width=150,
                                        help="현재가 기준 예상 잔고"
                                    )
                                }
                            )

                            # 수익률 강조 표시
                            col_profit1, col_profit2 = st.columns(2)
                            with col_profit1:
                                profit_color = "green" if current_profit >= 0 else "red"
                                st.markdown(f"""
                                    <div style='
                                        padding: 10px;
                                        border-radius: 5px;
                                        background-color: {'rgba(0,255,0,0.1)' if current_profit >= 0 else 'rgba(255,0,0,0.1)'};
                                        text-align: center;
                                    '>
                                        <span style='
                                            color: {profit_color};
                                            font-size: 1.3rem;
                                            font-weight: bold;
                                        '>
                                            수익률: {current_profit:+.2f}%
                                        </span>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with col_profit2:
                                st.markdown(f"""
                                    <div style='
                                        padding: 10px;
                                        border-radius: 5px;
                                        background-color: rgba(0,0,0,0.05);
                                        text-align: center;
                                    '>
                                        <span style='
                                            font-size: 1.3rem;
                                            font-weight: bold;
                                        '>
                                            현재 잔고: {format_number(current_balance)}원
                                        </span>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                        else:
                            st.markdown("### ⚪ 매수 대기")
                            st.markdown(f"""
                                <div style='
                                    padding: 10px;
                                    border-radius: 5px;
                                    background-color: rgba(0,0,0,0.05);
                                    text-align: center;
                                '>
                                    <span style='
                                        font-size: 1.3rem;
                                        font-weight: bold;
                                    '>
                                        현재 잔고: {format_number(current_balance)}원
                                    </span>
                                </div>
                            """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"포지션 정보 로드 중 오류 발생: {str(e)}")

        # 메인 차트와 거래 현황
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("MACD 차트")
            
            with get_connection() as conn:
                if conn is None:
                    st.error("데이터베이스에 연결할 수 없습니다.")
                    return
                    
                try:
                    df_prices = pd.read_sql("""
                        SELECT timestamp, close
                        FROM price_data 
                        ORDER BY timestamp DESC LIMIT 1440
                    """, conn)

                    if not df_prices.empty:
                        df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'])
                        now = datetime.now()
                        one_day_ago = now - timedelta(days=1)
                        six_hours_ago = now - timedelta(hours=6)
                        
                        # 24시간치 데이터 준비
                        df_prices = df_prices[df_prices['timestamp'] >= one_day_ago]
                        df_prices = df_prices.sort_values('timestamp')
                        
                        # 처음에 보여줄 범위 (6시간)
                        initial_range = [six_hours_ago, now]   
                        
                        if len(df_prices) > 0:
                            df_prices = df_prices.sort_values('timestamp')
                            macd, signal = calculate_macd(df_prices)
                            
                            if macd is not None and signal is not None:
                                df_prices['MACD'] = macd
                                df_prices['Signal'] = signal
                                df_prices['MACD_Hist'] = macd - signal

                                # 서브플롯 생성 (2개의 y축)
                                fig = make_subplots(specs=[[{"secondary_y": True}]])

                                # 코인 가격 라인 (오른쪽 y축)
                                fig.add_trace(
                                    go.Scatter(
                                        x=df_prices['timestamp'],
                                        y=df_prices['close'],
                                        name='BTC/KRW',
                                        line=dict(color='black', width=1),
                                        hovertemplate='%{y:,.0f}원'
                                    ),
                                    secondary_y=True
                                )

                                # MACD 히스토그램 (왼쪽 y축)
                                colors = ['red' if val < 0 else 'green' for val in df_prices['MACD_Hist']]
                                fig.add_trace(
                                    go.Bar(
                                        x=df_prices['timestamp'],
                                        y=df_prices['MACD_Hist'],
                                        name='MACD Hist',
                                        marker_color=colors,
                                        opacity=0.5
                                    ),
                                    secondary_y=False
                                )

                                # MACD 라인 (왼쪽 y축)
                                fig.add_trace(
                                    go.Scatter(
                                        x=df_prices['timestamp'],
                                        y=df_prices['MACD'],
                                        name='MACD',
                                        line=dict(color='blue', width=2)
                                    ),
                                    secondary_y=False
                                )

                                # Signal 라인 (왼쪽 y축)
                                fig.add_trace(
                                    go.Scatter(
                                        x=df_prices['timestamp'],
                                        y=df_prices['Signal'],
                                        name='Signal',
                                        line=dict(color='orange', width=2)
                                    ),
                                    secondary_y=False
                                )

                                # 0선 (왼쪽 y축)
                                fig.add_hline(y=0, line_dash="dot", line_color="gray", secondary_y=False)

                                # 매매 포인트 표시
                                trades = pd.read_sql("""
                                    SELECT 
                                        timestamp, 
                                        type,
                                        price,
                                        amount,
                                        profit_rate 
                                    FROM trades 
                                    WHERE timestamp >= ? AND type != 'INITIAL'
                                    ORDER BY timestamp DESC
                                """, conn, params=[one_day_ago])

                                if not trades.empty:
                                    trades['timestamp'] = pd.to_datetime(trades['timestamp'])

                                    # 매수 포인트
                                    buys = trades[trades['type'] == 'BUY']
                                    if not buys.empty:
                                        fig.add_trace(
                                            go.Scatter(
                                                x=buys['timestamp'],
                                                y=buys['price'],
                                                mode='markers+text',  # text 모드 추가
                                                marker=dict(
                                                    symbol='triangle-up',
                                                    size=15,
                                                    color='green'
                                                ),
                                                name='매수',
                                                text=[f"매수<br>{t.strftime('%H:%M')}<br>{format_number(p)}원<br>{format_number(p*a)}원" 
                                                     for t, p, a in zip(buys['timestamp'], buys['price'], buys['amount'])],
                                                textposition='top center',  # 텍스트 위치
                                                textfont=dict(size=10),  # 텍스트 크기
                                                hovertemplate='%{y:,.0f}원'
                                            ),
                                            secondary_y=True
                                        )

                                    # 매도 포인트
                                    sells = trades[trades['type'].str.startswith('SELL')]
                                    if not sells.empty:
                                        fig.add_trace(
                                            go.Scatter(
                                                x=sells['timestamp'],
                                                y=sells['price'],
                                                mode='markers+text',  # text 모드 추가
                                                marker=dict(
                                                    symbol='triangle-down',
                                                    size=15,
                                                    color='red'
                                                ),
                                                name='매도',
                                                text=[f"매도<br>{tm.strftime('%H:%M')}<br>{format_number(p)}원<br>{r:+.2f}%" 
                                                     for tm, p, r in zip(sells['timestamp'], sells['price'], sells['profit_rate'])],
                                                textposition='bottom center',  # 텍스트 위치
                                                textfont=dict(size=10),  # 텍스트 크기
                                                hovertemplate='매도가: %{y:,.0f}원<br>수익률: %{text}%'
                                            ),
                                            secondary_y=True
                                        )

                                # 레이아웃 업데이트
                                fig.update_layout(
                                    height=600,
                                    template='plotly_white',
                                    showlegend=True,
                                    legend=dict(
                                        yanchor="top",
                                        y=0.99,
                                        xanchor="left",
                                        x=0.01,
                                        bgcolor="rgba(255, 255, 255, 0.8)"
                                    ),
                                    xaxis_title="시간",
                                    hovermode='x unified',
                                    margin=dict(t=30, l=10, r=10, b=30),
                                    dragmode='pan',  # 드래그로 panning 가능하게 설정
                                    xaxis=dict(
                                    showgrid=True,
                                    gridcolor='rgba(128, 128, 128, 0.2)',
                                    showspikes=True,
                                    spikecolor='rgba(0, 0, 0, 0.5)',
                                    spikesnap='cursor',
                                    spikemode='across',
                                    spikethickness=2,
                                    type='date',
                                    range=initial_range  # 이 줄 추가
                                ),
                                    yaxis=dict(
                                        showgrid=True,
                                        gridcolor='rgba(128, 128, 128, 0.2)',
                                        showspikes=True,
                                        spikecolor='rgba(0, 0, 0, 0.5)',
                                        spikesnap='cursor',
                                        spikemode='across',
                                        spikethickness=2
                                    ),
                                    yaxis2=dict(
                                        showgrid=True,
                                        gridcolor='rgba(128, 128, 128, 0.2)',
                                        showspikes=True,
                                        spikecolor='rgba(0, 0, 0, 0.5)',
                                        spikesnap='cursor',
                                        spikemode='across',
                                        spikethickness=2
                                    )
                                )

                                # y축 제목 설정
                                fig.update_yaxes(title_text="MACD", secondary_y=False)
                                fig.update_yaxes(title_text="BTC/KRW", secondary_y=True)

                                current_time = datetime.now().strftime("%Y%m%d%H%M%S")
                                st.plotly_chart(fig, use_container_width=True, key=f"macd_chart_{current_time}")
                                
                            else:
                                st.info("MACD 계산을 위한 데이터가 부족합니다.")
                        else:
                            st.info("최근 6시간 동안의 가격 데이터가 없습니다.")
                    else:
                        st.info("가격 데이터가 없습니다. 데이터가 수집되면 차트가 표시됩니다.")
                except Exception as e:
                    st.error(f"차트 생성 중 오류 발생: {str(e)}")

        # 거래 현황
        with col2:
            st.subheader("거래 현황")
            
            with get_connection() as conn:
                if conn is None:
                    st.error("데이터베이스에 연결할 수 없습니다.")
                    return
                    
                try:
                    # 전략 수익률과 시장 수익률 계산을 위한 쿼리
                    stats_query = """
                        WITH initial_price AS (
                            SELECT close as first_price, timestamp as start_time
                            FROM price_data
                            ORDER BY timestamp ASC
                            LIMIT 1
                        ),
                        current_price AS (
                            SELECT close as last_price
                            FROM price_data
                            ORDER BY timestamp DESC
                            LIMIT 1
                        ),
                        trade_summary AS (
                            SELECT 
                                SUM(CASE WHEN type LIKE 'SELL%' THEN profit ELSE 0 END) as total_profit,
                                AVG(CASE WHEN type LIKE 'SELL%' THEN profit_rate END) as avg_profit_rate
                            FROM trades
                            WHERE type != 'INITIAL'
                        )
                        SELECT 
                            total_profit,
                            avg_profit_rate,
                            ((last_price - first_price) / first_price * 100) as market_return,
                            (total_profit / (SELECT balance FROM trades WHERE type = 'INITIAL') * 100) as strategy_return,
                            start_time
                        FROM trade_summary, initial_price, current_price
                    """
                    
                    stats = pd.read_sql(stats_query, conn)

                    col21, col22 = st.columns(2)
                    with col21:
                        strategy_return = stats['strategy_return'].iloc[0] if not stats['strategy_return'].isna().iloc[0] else 0
                        st.metric(
                            "Strategy Return", 
                            f"{strategy_return:+.2f}%",
                            delta=None,
                            delta_color="normal"
                        )
                        st.metric(
                            "총 수익", 
                            f"{format_number(stats['total_profit'].iloc[0]) if not stats['total_profit'].isna().iloc[0] else 0}원"
                        )
                    with col22:
                        market_return = stats['market_return'].iloc[0] if not stats['market_return'].isna().iloc[0] else 0
                        start_time = pd.to_datetime(stats['start_time'].iloc[0]).strftime('%m-%d %H:%M') if not stats['start_time'].isna().iloc[0] else ""
                        st.metric(
                            f"Market Return (since {start_time})", 
                            f"{market_return:+.2f}%",
                            delta=None,
                            delta_color="normal"
                        )
                        st.metric(
                            "평균 수익률", 
                            f"{stats['avg_profit_rate'].iloc[0]:.2f}%" if not stats['avg_profit_rate'].isna().iloc[0] else "0.00%"
                        )

                    # 최근 거래 내역
                    trades_query = """
                        WITH trade_pairs AS (
                            SELECT 
                                t1.timestamp as sell_time,
                                t1.price * t1.amount as sell_total,
                                t1.profit,
                                t1.profit_rate,
                                t2.timestamp as buy_time,
                                t2.price * t2.amount as buy_total,
                                t1.type as sell_type
                            FROM trades t1
                            LEFT JOIN trades t2 ON t2.id = (
                                SELECT MAX(id) FROM trades 
                                WHERE type = 'BUY' AND id < t1.id
                            )
                            WHERE t1.type LIKE 'SELL%'
                            ORDER BY t1.timestamp DESC
                        )
                        SELECT * FROM trade_pairs
                        LIMIT 10
                    """
                    
                    trades = pd.read_sql(trades_query, conn)

                    if not trades.empty:
                        st.markdown("### 최근 거래 내역")
                        trades['매수시간'] = pd.to_datetime(trades['buy_time']).dt.strftime('%m-%d %H:%M:%S')
                        trades['매도시간'] = pd.to_datetime(trades['sell_time']).dt.strftime('%m-%d %H:%M:%S')
                        trades['매수금액'] = trades['buy_total'].apply(format_number)
                        trades['매도금액'] = trades['sell_total'].apply(format_number)
                        trades['손익'] = trades['profit'].apply(format_number)
                        trades['수익률'] = trades['profit_rate'].apply(lambda x: f"{x:+.2f}%")

                        # 데이터프레임 표시
                        st.dataframe(
                            trades[['매수시간', '매도시간', '매수금액', '매도금액', '손익', '수익률']],
                            hide_index=True,
                            height=400,
                            use_container_width=True,
                            key=f"trades_df_{current_time}"  # 유니크한 key 추가
                        )

                        # CSS로 글자 크기 조정
                        st.markdown("""
                            <style>
                            .stDataFrame {
                                font-size: 1.2rem !important;
                            }
                            </style>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("아직 거래 내역이 없습니다.")
                except Exception as e:
                    st.error(f"거래 현황 로드 중 오류 발생: {str(e)}")
    except Exception as e:
        st.error(f"대시보드 렌더링 중 오류 발생: {str(e)}")

def main():
    """메인 함수"""
    st.set_page_config(
        page_title="BTC 모니터링 시스템", 
        page_icon="📈", 
        layout="wide"
    )
    
    # 초기 컨테이너 설정
    title_container = st.empty()
    main_container = st.empty()

    # 최초 렌더링
    title_text = "BTC 모니터링 시스템"
    credits_text = "<small>- Program by Lim</small>"
    title_container.markdown(f"""
        # {title_text}
        {credits_text}
    """, unsafe_allow_html=True)
    
    with main_container:
        render_dashboard()

    # 실시간 업데이트 루프
    while True:
        try:
            now = datetime.now()
            seconds_until_next_minute = 60 - now.second

            # 매 초마다 타이머 업데이트
            timer_text = f"다음 업데이트까지: **{seconds_until_next_minute}초**"
            title_container.markdown(f"""
                # {title_text}
                {timer_text}
                {credits_text}
            """, unsafe_allow_html=True)

            # 1분마다 대시보드 전체 업데이트
            if seconds_until_next_minute == 60:
                main_container.empty()
                with main_container:
                    render_dashboard()

            time.sleep(1)

        except Exception as e:
            st.error(f"오류 발생: {str(e)}")
            time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"프로그램 오류: {str(e)}")