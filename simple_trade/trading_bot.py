import pyupbit
import pandas as pd
import numpy as np
import time
from datetime import datetime
import streamlit as st
import threading

class TradingBot:
    def __init__(self):
        st.write("비트코인 자동매매 시스템을 초기화합니다...")
        self.api_key = "YOUR_API_KEY"
        self.secret_key = "YOUR_SECRET_KEY"
        self.upbit = pyupbit.Upbit(self.api_key, self.secret_key)
        self.symbol = "KRW-BTC"
        self.position = "NONE"
        self.trades = []
        self.initial_balance = 10000000  # 1000만원 초기 자본
        self.current_balance = self.initial_balance
        
        # 초기 데이터 로딩
        st.write(f"초기 설정 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.load_initial_data()
        
    def load_initial_data(self):
        try:
            # 충분한 데이터 포인트 확보를 위해 200개의 1분봉 데이터 로드
            self.df = pyupbit.get_ohlcv(self.symbol, interval="minute1", count=200)
            if self.df is not None and not self.df.empty:
                self.current_price = self.df['close'].iloc[-1]
                self.df['macd'] = self.calculate_macd(self.df['close'])
                self.current_macd = self.df['macd'].iloc[-1]
                st.write("업비트 서버에 정상적으로 연결되었습니다.")
                st.write(f"트레이딩 시작 - 초기 잔고: {format_price(self.initial_balance)}")
            else:
                st.error("초기 데이터 로딩에 실패했습니다. 잠시 후 다시 시도합니다.")
        except Exception as e:
            st.error(f"초기화 중 오류 발생: {str(e)}")
            
    def calculate_macd(self, prices):
        if len(prices) < 26:
            return pd.Series(np.nan, index=prices.index)
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd - signal
        
    def check_candle_pattern(self, df, pattern_type):
        if len(df) < 5:
            return False
            
        last_5_candles = df[-5:].copy()
        last_5_candles['change'] = last_5_candles['close'] - last_5_candles['open']
        
        if pattern_type == 'BUY':
            pattern = [-1, -1, -1, 1, 1]
        else:  # SELL
            pattern = [1, 1, 1, -1, -1]
            
        actual_pattern = [1 if x > 0 else -1 for x in last_5_candles['change']]
        return actual_pattern == pattern
        
    def get_minute_data(self):
        try:
            df = pyupbit.get_ohlcv(self.symbol, interval="minute1", count=100)
            if df is not None and not df.empty:
                df['macd'] = self.calculate_macd(df['close'])
                return df
        except Exception as e:
            st.error(f"데이터 로딩 중 오류: {str(e)}")
        return pd.DataFrame()
        
    def check_trading_signals(self, df):
        if df.empty or len(df) < 26:  # MACD 계산에 필요한 최소 데이터 포인트
            return "HOLD"
            
        if pd.isna(df['macd'].iloc[-1]):
            return "HOLD"
            
        current_macd = df['macd'].iloc[-1]
        self.current_macd = current_macd
        
        if self.position == "NONE":
            if current_macd < 0 and self.check_candle_pattern(df, 'BUY'):
                return "BUY"
        elif self.position == "LONG":
            if current_macd > 0 and self.check_candle_pattern(df, 'SELL'):
                return "SELL"
                
        return "HOLD"
        
    def execute_trade(self, signal):
        try:
            current_time = datetime.now()
            current_price = pyupbit.get_current_price(self.symbol)
            if current_price is None:
                return
                
            self.current_price = current_price
            
            if signal == "BUY" and self.position == "NONE":
                self.position = "LONG"
                investment = self.current_balance
                quantity = investment / current_price
                self.trades.append({
                    'time': current_time,
                    'type': 'BUY',
                    'price': current_price,
                    'quantity': quantity,
                    'investment': investment,
                    'macd': self.current_macd
                })
                st.success(f"매수 신호: {format_price(current_price)}")
                
            elif signal == "SELL" and self.position == "LONG":
                self.position = "NONE"
                last_buy = next(trade for trade in reversed(self.trades) 
                              if trade['type'] == 'BUY')
                quantity = last_buy['quantity']
                current_value = quantity * current_price
                profit = ((current_value - last_buy['investment']) / 
                         last_buy['investment']) * 100
                self.current_balance = current_value
                
                self.trades.append({
                    'time': current_time,
                    'type': 'SELL',
                    'price': current_price,
                    'quantity': quantity,
                    'value': current_value,
                    'profit': profit,
                    'macd': self.current_macd
                })
                st.success(f"매도 신호: {format_price(current_price)} (수익률: {profit:.2f}%)")
                
        except Exception as e:
            st.error(f"거래 실행 중 오류: {str(e)}")
            
    def get_last_candles(self):
        if self.df is not None and not self.df.empty:
            return self.df[-5:]
        return pd.DataFrame()
            
    def run(self):
        while True:
            try:
                self.df = self.get_minute_data()
                if not self.df.empty:
                    signal = self.check_trading_signals(self.df)
                    self.execute_trade(signal)
                time.sleep(1)
            except Exception as e:
                st.error(f"실행 중 오류: {str(e)}")
                time.sleep(1)

def format_price(price):
    return f"₩{price:,.0f}" if price else "로딩 중..."

def create_streamlit_ui():
    st.set_page_config(
        page_title="BTC/KRW Trading Bot",
        layout="wide"
    )
    
    st.title("BTC/KRW Trading Bot")
    
    # Initialize session state
    if 'bot' not in st.session_state:
        st.session_state.bot = TradingBot()
        trading_thread = threading.Thread(target=st.session_state.bot.run)
        trading_thread.daemon = True
        trading_thread.start()
    
    # Main content area
    col1, col2 = st.columns([7, 3])
    
    with col1:
        st.subheader("시장 정보")
        
        # 현재 가격과 MACD 값을 메트릭으로 표시
        metrics_container = st.container()
        with metrics_container:
            price_col, macd_col, pos_col = st.columns(3)
            
            with price_col:
                st.metric(
                    label="현재 가격", 
                    value=format_price(st.session_state.bot.current_price)
                )
                
            with macd_col:
                st.metric(
                    label="MACD", 
                    value=f"{st.session_state.bot.current_macd:.2f}"
                )
                
            with pos_col:
                st.metric(
                    label="포지션", 
                    value=st.session_state.bot.position
                )
        
        # 잔고 정보
        st.metric(
            label="현재 잔고",
            value=format_price(st.session_state.bot.current_balance)
        )
        
        # 최근 캔들 정보
        st.subheader("최근 캔들 패턴")
        last_candles = st.session_state.bot.get_last_candles()
        if not last_candles.empty:
            formatted_candles = last_candles[['open', 'high', 'low', 'close', 'volume', 'macd']].copy()
            for col in ['open', 'high', 'low', 'close']:
                formatted_candles[col] = formatted_candles[col].apply(lambda x: f"₩{x:,.0f}")
            formatted_candles['volume'] = formatted_candles['volume'].apply(lambda x: f"{x:,.0f}")
            formatted_candles['macd'] = formatted_candles['macd'].apply(lambda x: f"{x:.2f}")
            st.dataframe(formatted_candles)
    
    # Sidebar for trade history
    with st.sidebar:
        st.header("거래 내역")
        if st.session_state.bot.trades:
            df_trades = pd.DataFrame(st.session_state.bot.trades)
            df_trades['time'] = df_trades['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df_trades['price'] = df_trades['price'].apply(lambda x: f"₩{x:,.0f}")
            if 'profit' in df_trades.columns:
                df_trades['profit'] = df_trades['profit'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
            if 'quantity' in df_trades.columns:
                df_trades['quantity'] = df_trades['quantity'].apply(lambda x: f"{x:.8f}")
            if 'investment' in df_trades.columns:
                df_trades['investment'] = df_trades['investment'].apply(lambda x: f"₩{x:,.0f}")
            if 'value' in df_trades.columns:
                df_trades['value'] = df_trades['value'].apply(lambda x: f"₩{x:,.0f}")
            st.dataframe(df_trades)
    
    # Auto-refresh
    time.sleep(1)
    st.experimental_rerun()

if __name__ == "__main__":
    create_streamlit_ui()