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
        return sqlite3.connect('trading.db', timeout=30)
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
                return float(sell_balance)
    except Exception as e:
        st.warning(f"잔고 조회 오류: {e}")
    return 10000000.0

def render_dashboard():
    """대시보드 렌더링"""
    try:
        # 기존 차트 객체 정리
        if 'current_figure' in st.session_state:
            del st.session_state['current_figure']
            
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
                    ORDER BY timestamp DESC LIMIT 200
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
                                use_container_width=True,
                                column_config={
                                    "진입시간": st.column_config.Column(width=150),
                                    "진입가격": st.column_config.Column(width=150),
                                    "현재가격": st.column_config.Column(width=150),
                                    "보유시간": st.column_config.Column(width=120),
                                    "수익률": st.column_config.Column(width=100),
                                    "예상잔고": st.column_config.Column(width=150)
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
            st.subheader("MACD & 상승비율")
            
            with get_connection() as conn:
                if conn is None:
                    st.error("데이터베이스에 연결할 수 없습니다.")
                    return
                    
                try:
                    # 가격 데이터 조회 - 3일치
                    df_prices = pd.read_sql("""
                        SELECT datetime(timestamp) as timestamp, close
                        FROM price_data 
                        ORDER BY timestamp DESC LIMIT 14400
                    """, conn)
                    
                    # 마켓 데이터 조회
                    df_market = pd.read_sql("""
                        SELECT datetime(timestamp) as timestamp, rising_rate, rising_count, total_count
                        FROM market_data
                        ORDER BY timestamp DESC LIMIT 14400
                    """, conn)
                    
                    if not df_prices.empty and not df_market.empty:
                        df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'])
                        df_market['timestamp'] = pd.to_datetime(df_market['timestamp'])
                        
                        now = datetime.now()
                        three_days_ago = now - timedelta(days=3)
                        six_hours_ago = now - timedelta(hours=6)
                        
                        # 3일치 데이터 준비
                        df_prices = df_prices[df_prices['timestamp'] >= three_days_ago]
                        df_market = df_market[df_market['timestamp'] >= three_days_ago]
                        
                        df_prices = df_prices.sort_values('timestamp')
                        df_market = df_market.sort_values('timestamp')
                        
                        # 초기 표시 범위 (6시간)
                        initial_range = [six_hours_ago, now]
                        
                        if len(df_prices) > 0:
                            # MACD 계산
                            macd, signal = calculate_macd(df_prices)
                            
                            if macd is not None and signal is not None:
                                df_prices['MACD'] = macd
                                df_prices['Signal'] = signal
                                df_prices['MACD_Hist'] = macd - signal

                                # 2개의 서브플롯 생성
                                fig = make_subplots(
                                    rows=2, 
                                    cols=1,
                                    shared_xaxes=True,
                                    vertical_spacing=0.05,
                                    row_heights=[0.7, 0.3],
                                    specs=[[{"secondary_y": True}],
                                          [{"secondary_y": False}]]
                                )

                                # 코인 가격 라인
                                fig.add_trace(
                                    go.Scatter(
                                        x=df_prices['timestamp'],
                                        y=df_prices['close'],
                                        name='BTC/KRW',
                                        line=dict(color='black', width=1.5),
                                        hovertemplate='%{y:,.0f}원'
                                    ),
                                    row=1, col=1,
                                    secondary_y=True
                                )

                                # 현재가격 선 추가 (항상 표시)
                                current_price = df_prices['close'].iloc[-1]
                                latest_time = df_prices['timestamp'].iloc[-1]

                                # 현재가격 선
                                fig.add_hline(
                                    y=current_price,
                                    line_dash="dash",
                                    line_color="blue",
                                    line_width=1,
                                    row=1, col=1,
                                    secondary_y=True
                                )

                                # 현재가격 주석
                                fig.add_annotation(
                                    text=f"현재가격: {format_number(current_price)}원",
                                    x=latest_time,
                                    y=current_price,
                                    xref="x",
                                    yref="y2",
                                    showarrow=False,
                                    bgcolor="rgba(135, 206, 235, 0.8)",
                                    bordercolor="blue",
                                    borderwidth=1,
                                    font=dict(size=12, color="black"),
                                    xanchor="left",
                                    yanchor="middle"
                                )

                                # MACD 히스토그램
                                colors = ['red' if val < 0 else 'green' for val in df_prices['MACD_Hist']]
                                fig.add_trace(
                                    go.Bar(
                                        x=df_prices['timestamp'],
                                        y=df_prices['MACD_Hist'],
                                        name='MACD Hist',
                                        marker=dict(color=colors, opacity=0.6),
                                        width=50000,  # 막대 너비 1분으로 설정
                                    ),
                                    row=1, col=1,
                                    secondary_y=False
                                )

                                # MACD 라인과 Signal 라인
                                fig.add_trace(
                                    go.Scatter(
                                        x=df_prices['timestamp'],
                                        y=df_prices['MACD'],
                                        name='MACD',
                                        line=dict(color='blue', width=2)
                                    ),
                                    row=1, col=1,
                                    secondary_y=False
                                )

                                fig.add_trace(
                                    go.Scatter(
                                        x=df_prices['timestamp'],
                                        y=df_prices['Signal'],
                                        name='Signal',
                                        line=dict(color='orange', width=2)
                                    ),
                                    row=1, col=1,
                                    secondary_y=False
                                )

                                # 매매 포인트 표시
                                trades_query = """
                                    SELECT 
                                        datetime(timestamp) as timestamp, 
                                        type,
                                        price,
                                        amount,
                                        profit_rate 
                                    FROM trades 
                                    WHERE timestamp >= ? AND type != 'INITIAL'
                                    ORDER BY timestamp DESC
                                """
                                trades = pd.read_sql(trades_query, conn, params=[three_days_ago])

                                if not trades.empty:
                                    trades['timestamp'] = pd.to_datetime(trades['timestamp'])

                                    # 매수 포인트
                                    buys = trades[trades['type'] == 'BUY']
                                    if not buys.empty:
                                        fig.add_trace(
                                            go.Scatter(
                                                x=buys['timestamp'],
                                                y=buys['price'],
                                                mode='markers+text',
                                                marker=dict(
                                                    symbol='triangle-up',
                                                    size=15,
                                                    color='green',
                                                    opacity=0.6
                                                ),
                                                name='매수',
                                                text=[f"매수<br>{t.strftime('%H:%M')}<br>{format_number(p)}원" 
                                                     for t, p in zip(buys['timestamp'], buys['price'])],
                                                textposition='top center',
                                                hovertemplate='%{y:,.0f}원'
                                            ),
                                            row=1, col=1,
                                            secondary_y=True
                                        )

                                        # 현재 진입가격 확인 (가장 최근 매수 확인)
                                        if buys.iloc[0]['timestamp'] == trades.iloc[0]['timestamp']:
                                            # 현재 매수 포지션이 있는 경우, 진입가격 표시
                                            entry_price = buys.iloc[0]['price']
                                            profit_rate = ((current_price - entry_price) / entry_price) * 100
                                            profit_color = "green" if profit_rate >= 0 else "red"
                                            
                                            # 진입가격 선
                                            fig.add_hline(
                                                y=entry_price,
                                                line_dash="dash",
                                                line_color="red",
                                                line_width=1,
                                                row=1, col=1,
                                                secondary_y=True
                                            )
                                            
                                            # 진입가격 주석
                                            fig.add_annotation(
                                                text=f"진입가격: {format_number(entry_price)}원<br>수익률: {profit_rate:+.2f}%",
                                                x=latest_time,
                                                y=entry_price,
                                                xref="x",
                                                yref="y2",
                                                showarrow=False,
                                                bgcolor="rgba(255, 182, 193, 0.8)",
                                                bordercolor="red",
                                                borderwidth=1,
                                                font=dict(size=12, color=profit_color),
                                                xanchor="left",
                                                yanchor="middle"
                                            )

                                    # 매도 포인트
                                    sells = trades[trades['type'].isin(['SELL', 'STOP_LOSS'])]
                                    if not sells.empty:
                                        for idx, row in sells.iterrows():
                                            marker_color = 'red' if row['type'] == 'STOP_LOSS' else 'blue'
                                            trade_type = '손절매도' if row['type'] == 'STOP_LOSS' else '매도'
                                            show_legend = bool((row['type'] == 'STOP_LOSS' and idx == sells[sells['type'] == 'STOP_LOSS'].index[0]) or 
                                                             (row['type'] == 'SELL' and idx == sells[sells['type'] == 'SELL'].index[0]))
                                            
                                            fig.add_trace(
                                                go.Scatter(
                                                    x=[row['timestamp']],
                                                    y=[row['price']],
                                                    mode='markers+text',
                                                    marker=dict(
                                                        symbol='triangle-down',
                                                        size=15,
                                                        color=marker_color,
                                                        opacity=0.6
                                                    ),
                                                    name=trade_type,
                                                    text=f"{trade_type}<br>{row['timestamp'].strftime('%H:%M')}<br>{format_number(row['price'])}원<br>{row['profit_rate']:+.2f}%",
                                                    textposition='bottom center',
                                                    showlegend=show_legend
                                                ),
                                                row=1, col=1,
                                                secondary_y=True
                                            )

                                # 상승비율 차트
                                fig.add_trace(
                                    go.Scatter(
                                        x=df_market['timestamp'],
                                        y=df_market['rising_rate'],
                                        name='상승비율',
                                        line=dict(color='purple', width=2),
                                        fill='tozeroy',
                                        hovertemplate='시간: %{x|%Y-%m-%d %H:%M}<br>' +
                                                    '상승비율: %{y:.0f}%<br>' +
                                                    '전체종목수: %{customdata[0]:,d}<br>' +
                                                    '상승종목수: %{customdata[1]:,d}<br>' +
                                                    '하락종목수: %{customdata[2]:,d}<extra></extra>',
                                        customdata=np.column_stack((
                                            df_market['total_count'],
                                            df_market['rising_count'],
                                            df_market['total_count'] - df_market['rising_count']
                                        ))
                                    ),
                                    row=2, col=1
                                )

                                # 50% 기준선 추가
                                fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)

                                # 0선 추가 (MACD 차트)
                                fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1, secondary_y=False)

                                # 레이아웃 설정
                                fig.update_layout(
                                    height=900,
                                    template='plotly_white',
                                    showlegend=True,
                                    legend=dict(
                                        yanchor="top",
                                        y=0.99,
                                        xanchor="left",
                                        x=0.01,
                                        bgcolor="rgba(255, 255, 255, 0.8)"
                                    ),
                                    hovermode='x unified',
                                    margin=dict(t=30, l=10, r=120, b=30),
                                    dragmode='pan'
                                )

                                # x축 설정
                                fig.update_xaxes(
                                    range=initial_range,
                                    rangeslider=dict(visible=False),
                                    type='date',
                                    showspikes=True,
                                    spikemode='across',
                                    spikesnap='cursor',
                                    spikecolor='grey',
                                    spikethickness=1
                                )

                                # y축 설정
                                fig.update_yaxes(
                                    title_text="MACD",
                                    row=1, col=1,
                                    secondary_y=False,
                                    showspikes=True,
                                    spikemode='across',
                                    spikesnap='cursor',
                                    spikecolor='grey',
                                    spikethickness=1
                                )
                                fig.update_yaxes(
                                    title_text="BTC/KRW",
                                    row=1, col=1,
                                    secondary_y=True,
                                    showspikes=True,
                                    spikemode='across',
                                    spikesnap='cursor',
                                    spikecolor='grey',
                                    spikethickness=1
                                )
                                fig.update_yaxes(
                                    title_text="상승비율(%)",
                                    range=[0, 100],
                                    row=2, col=1,
                                    showspikes=True,
                                    spikemode='across',
                                    spikesnap='cursor',
                                    spikecolor='grey',
                                    spikethickness=1
                                )

                                st.plotly_chart(fig, use_container_width=True)
                                st.session_state['current_figure'] = fig

                            else:
                                st.error("MACD 계산에 실패했습니다.")
                        else:
                            st.error("가격 데이터가 충분하지 않습니다.")
                    else:
                        st.error("데이터를 불러올 수 없습니다.")
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
                    # 거래 통계 계산
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
                                SUM(CASE WHEN type LIKE 'SELL%' OR type = 'STOP_LOSS' THEN profit ELSE 0 END) as total_profit,
                                AVG(CASE WHEN type LIKE 'SELL%' OR type = 'STOP_LOSS' THEN profit_rate END) as avg_profit_rate
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
                                t1.type as sell_type,
                                t2.timestamp as buy_time,
                                t2.price * t2.amount as buy_total
                            FROM trades t1
                            LEFT JOIN trades t2 ON t2.id = (
                                SELECT MAX(id) FROM trades 
                                WHERE type = 'BUY' AND id < t1.id
                            )
                            WHERE t1.type LIKE 'SELL%' OR t1.type = 'STOP_LOSS'
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
                        trades['거래유형'] = trades['sell_type'].apply(lambda x: '손절매도' if x == 'STOP_LOSS' else '매도')

                        # 데이터프레임 표시
                        st.dataframe(
                            trades[['매수시간', '매도시간', '매수금액', '매도금액', '손익', '수익률', '거래유형']],
                            hide_index=True,
                            height=400,
                            use_container_width=True,
                            column_config={
                                "매수시간": st.column_config.Column(width=150),
                                "매도시간": st.column_config.Column(width=150),
                                "매수금액": st.column_config.Column(width=120),
                                "매도금액": st.column_config.Column(width=120),
                                "손익": st.column_config.Column(width=100),
                                "수익률": st.column_config.Column(width=100),
                                "거래유형": st.column_config.Column(width=100)
                            }
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