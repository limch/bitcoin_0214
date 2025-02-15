import sqlite3
import time
import numpy as np
from datetime import datetime
import pandas as pd
import pyupbit
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Upbit API 키 설정
access_key = os.getenv('UPBIT_ACCESS_KEY')
secret_key = os.getenv('UPBIT_SECRET_KEY')
upbit = pyupbit.Upbit(access_key, secret_key)

def init_db():
    """데이터베이스 초기화 및 테이블 생성"""
    try:
        conn = sqlite3.connect('trading.db')
        c = conn.cursor()
        
        # price_data 테이블 생성
        c.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                timestamp DATETIME PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )
        ''')
        
        # trades 테이블 생성
        c.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                type TEXT,
                price REAL,
                amount REAL,
                fee REAL,
                balance REAL,
                profit REAL,
                profit_rate REAL
            )
        ''')
        
        conn.commit()
        
        # 초기 데이터 확인
        c.execute('SELECT COUNT(*) FROM trades')
        trades_count = c.fetchone()[0]
        
        # trades 테이블이 비어있을 경우 초기 잔고 설정
        if trades_count == 0:
            c.execute('''
                INSERT INTO trades (
                    timestamp, type, price, amount, fee, balance, profit, profit_rate
                ) VALUES (?, 'INITIAL', 0, 0, 0, 10000000, 0, 0)
            ''', (datetime.now(),))
            conn.commit()
            
        return conn
    except Exception as e:
        print(f"데이터베이스 초기화 오류: {e}")
        raise

def fetch_price_data():
    """업비트에서 BTC/KRW 현재가 데이터 가져오기"""
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            btc_price = pyupbit.get_current_price("KRW-BTC")
            if btc_price is None:
                raise ValueError("현재가 데이터를 받아올 수 없습니다.")
                
            btc_minute_data = pyupbit.get_ohlcv("KRW-BTC", interval="minute1", count=1)
            if btc_minute_data is None or btc_minute_data.empty:
                raise ValueError("분봉 데이터를 받아올 수 없습니다.")
            
            now = datetime.now()
            return {
                'timestamp': now,
                'open': float(btc_minute_data['open'].iloc[-1]),
                'high': float(btc_minute_data['high'].iloc[-1]),
                'low': float(btc_minute_data['low'].iloc[-1]),
                'close': float(btc_price),
                'volume': float(btc_minute_data['volume'].iloc[-1])
            }
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"데이터 가져오기 재시도 ({attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)
            else:
                print(f"데이터 가져오기 최종 실패: {e}")
                return None

def calculate_macd(df):
    """MACD(12,26,9) 계산"""
    try:
        if df is None or len(df) < 26:
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
        print(f"MACD 계산 오류: {e}")
        return None, None

def check_buy_conditions(conn):
    """매수 조건 확인"""
    try:
        c = conn.cursor()
        
        # 이전 거래가 매도인지 확인
        c.execute('SELECT type FROM trades ORDER BY timestamp DESC LIMIT 1')
        last_trade = c.fetchone()
        if not last_trade:
            print("거래 내역이 없습니다.")
            return False
        if last_trade[0] != 'SELL' and last_trade[0] != 'INITIAL':
            return False
        
        # MACD 계산을 위한 데이터 조회
        c.execute('SELECT close FROM price_data ORDER BY timestamp DESC LIMIT 30')
        closes = [row[0] for row in c.fetchall()]
        
        if len(closes) < 30:
            print("MACD 계산을 위한 데이터가 부족합니다.")
            return False
        
        # MACD 계산
        df = pd.DataFrame({'close': closes[::-1]})  # 시간순 정렬
        macd, signal = calculate_macd(df)
        
        if macd is None or signal is None:
            return False
            
        # 최근 5개의 MACD 값
        recent_macd = macd.values[-5:]  # 하나 더 추가
        
        # 변곡점 확인 후 연속 상승 체크
        macd_pattern = (
            recent_macd[0] > recent_macd[1] and  # 첫 번째 값이 하락
            recent_macd[1] > recent_macd[2] and  # 계속 하락
            recent_macd[2] < recent_macd[3] and  # 첫 번째 상승 (변곡점)
            recent_macd[3] < recent_macd[4]      # 두 번째 상승
        )
        
        # 모든 조건을 한번에 확인
        should_buy = (
            macd.iloc[-1] < signal.iloc[-1] and  # MACD가 Signal보다 낮음
            macd.iloc[-1] < 0 and               # MACD가 0보다 작음
            macd_pattern                        # 하락 후 연속 상승 패턴
        )
        
        if should_buy:
            print(f"매수 신호 발생: MACD = {macd.iloc[-1]:.2f}, Signal = {signal.iloc[-1]:.2f}")
            
        return should_buy
        
    except Exception as e:
        print(f"매수 조건 확인 중 오류 발생: {e}")
        return False

def check_sell_conditions(conn, buy_price):
    """매도 조건 확인"""
    try:
        if buy_price is None:
            print("매수 가격 정보가 없습니다.")
            return False
            
        c = conn.cursor()
        
        # 이전 거래가 매수인지 확인
        c.execute('SELECT type FROM trades ORDER BY timestamp DESC LIMIT 1')
        last_trade = c.fetchone()
        if not last_trade or last_trade[0] != 'BUY':
            return False
        
        # MACD 계산을 위한 데이터 조회
        c.execute('SELECT close FROM price_data ORDER BY timestamp DESC LIMIT 30')
        closes = [row[0] for row in c.fetchall()]
        
        if len(closes) < 30:
            print("MACD 계산을 위한 데이터가 부족합니다.")
            return False
        
        current_price = closes[0]
        
        # 최소 수익률 확인 (0.05%)
        if current_price < buy_price * 1.0005:
            return False
        
        # MACD 계산
        df = pd.DataFrame({'close': closes[::-1]})
        macd, signal = calculate_macd(df)
        
        if macd is None or signal is None:
            return False
            
        # MACD가 0보다 큰지 확인
        if macd.iloc[-1] <= 0:
            return False
        
        # 최근 4개의 MACD 값
        recent_macd = macd.values[-4:]
        
        # MACD가 상향 후 2번 연속 하향 확인
        if (recent_macd[-4] < recent_macd[-3] and    # 상향
            recent_macd[-3] > recent_macd[-2] and    # 첫 번째 하향
            recent_macd[-2] > recent_macd[-1]):      # 두 번째 하향
            return True
        
        return False
    except Exception as e:
        print(f"매도 조건 확인 중 오류 발생: {e}")
        return False

def execute_trade(conn, trade_type, current_price, balance, buy_price=None):
    """거래 실행 및 기록"""
    c = conn.cursor()
    now = datetime.now()
    
    try:
        if trade_type == 'BUY':
            # 매매수수료 계산 (0.05%)
            fee = float(balance) * 0.0005
            invest_amount = float(balance) - fee  # 수수료 제외한 실제 매수금액
            actual_amount = invest_amount / float(current_price) if current_price else 0  # 실제 매수 수량
            new_balance = 0  # 매수 후 잔고는 0
            
            c.execute('''
                INSERT INTO trades 
                (timestamp, type, price, amount, fee, balance, profit, profit_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (now, 'BUY', float(current_price), float(actual_amount), float(fee), 
                  float(new_balance), 0.0, 0.0))
            
            print(f"매수: {float(current_price):,.0f}원 - {float(actual_amount):.8f}BTC (매수금액: {float(invest_amount):,.0f}원, 수수료: {float(fee):,.0f}원)")
            
        elif trade_type == 'SELL':
            # 이전 매수 정보 조회
            c.execute('''
                SELECT price, amount FROM trades 
                WHERE type = 'BUY' 
                ORDER BY timestamp DESC LIMIT 1
            ''')
            last_trade = c.fetchone()
            
            if last_trade:
                buy_price, amount = float(last_trade[0]), float(last_trade[1])
                sell_amount = amount * float(current_price)  # 매도 총액
                profit = sell_amount - (buy_price * amount)  # 순수익
                profit_rate = (profit / (buy_price * amount)) * 100 if buy_price and amount else 0
                
                # 새로운 잔고는 매도금액
                new_balance = sell_amount
                
                c.execute('''
                    INSERT INTO trades 
                    (timestamp, type, price, amount, fee, balance, profit, profit_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (now, 'SELL', float(current_price), float(amount), 0.0, 
                      float(sell_amount), float(profit), float(profit_rate)))
                
                print(f"매도: {float(current_price):,.0f}원 - 수익: {float(profit):,.0f}원 ({float(profit_rate):+.2f}%)")
                print(f"새로운 잔고: {float(new_balance):,.0f}원")
            
            else:
                new_balance = balance
                print("이전 매수 기록을 찾을 수 없습니다.")
                
        conn.commit()
        return float(new_balance) if new_balance is not None else float(balance)
        
    except Exception as e:
        print(f"거래 처리 중 오류 발생: {e}")
        conn.rollback()
        return float(balance)

def trading_loop():
    """메인 트레이딩 루프"""
    try:
        conn = init_db()
        c = conn.cursor()
        
        # 초기 잔고 확인
        c.execute('SELECT balance FROM trades ORDER BY timestamp DESC LIMIT 1')
        result = c.fetchone()
        balance = float(result[0]) if result else 10000000
        
        # 마지막 거래 타입 확인
        c.execute('SELECT type FROM trades ORDER BY timestamp DESC LIMIT 1')
        result = c.fetchone()
        last_trade_type = result[0] if result else None
        
        # 마지막 거래가 매수인 경우 매수 가격 확인
        buy_price = None
        if last_trade_type == 'BUY':
            c.execute('SELECT price FROM trades WHERE type = "BUY" ORDER BY timestamp DESC LIMIT 1')
            result = c.fetchone()
            buy_price = float(result[0]) if result else None
        
        print(f"트레이딩 시작 - 초기 잔고: {balance:,.0f}원")
        
        while True:
            try:
                # 현재 시간이 1분이 되기를 기다림
                now = datetime.now()
                seconds_until_next_minute = 60 - now.second
                if seconds_until_next_minute < 60:
                    time.sleep(seconds_until_next_minute)
                
                # 가격 데이터 가져오기
                price_data = fetch_price_data()
                if price_data:
                    current_price = price_data['close']
                    
                    # 가격 데이터 저장
                    try:
                        c.execute('''
                            INSERT INTO price_data VALUES (?, ?, ?, ?, ?, ?)
                        ''', (price_data['timestamp'], price_data['open'], 
                             price_data['high'], price_data['low'], 
                             price_data['close'], price_data['volume']))
                        conn.commit()
                    except sqlite3.IntegrityError:
                        # 동일한 timestamp가 이미 존재하는 경우 무시
                        pass
                    except Exception as e:
                        print(f"가격 데이터 저장 중 오류 발생: {e}")
                        continue
                    
                    # 매매 조건 확인 및 실행
                    try:
                        if (last_trade_type != 'BUY' and check_buy_conditions(conn)):
                            balance = execute_trade(conn, 'BUY', current_price, balance)
                            last_trade_type = 'BUY'
                            buy_price = current_price
                            print(f"매수 신호 발생 및 실행 완료 - 현재가: {current_price:,.0f}원")
                            
                        elif last_trade_type == 'BUY':
                            if check_sell_conditions(conn, buy_price):
                                balance = execute_trade(conn, 'SELL', current_price, balance, buy_price)
                                last_trade_type = 'SELL'
                                buy_price = None
                                print(f"매도 신호 발생 및 실행 완료 - 현재가: {current_price:,.0f}원")
                    except Exception as e:
                        print(f"매매 실행 중 오류 발생: {e}")
                        continue
                    
                else:
                    print("가격 데이터를 가져올 수 없습니다. 다음 주기를 기다립니다.")
            
            except Exception as e:
                print(f"거래 루프 중 오류 발생: {e}")
                time.sleep(5)
                continue
            
    except Exception as e:
        print(f"치명적인 오류 발생: {e}")
        if conn:
            conn.close()
        raise
    finally:
        if conn:
            conn.close()

def main():
    """메인 함수"""
    print("비트코인 자동매매 시스템을 시작합니다...")
    print(f"초기 설정 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 무한 재시도 루프
    while True:
        try:
            if not access_key or not secret_key:
                raise ValueError("업비트 API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
            
            # 업비트 연결 테스트
            balance = upbit.get_balance("KRW")
            if balance is None:
                raise ConnectionError("업비트 서버에 연결할 수 없습니다.")
                
            print("업비트 서버에 정상적으로 연결되었습니다.")
            print("자동매매를 시작합니다...")
            
            # 트레이딩 루프 시작
            trading_loop()
            
        except Exception as e:
            print(f"\n오류가 발생하여 프로그램을 재시작합니다: {e}")
            print("10초 후 재시작합니다...")
            time.sleep(10)
            continue

if __name__ == "__main__":
    main()