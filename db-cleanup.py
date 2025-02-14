import sqlite3
from datetime import datetime, timedelta
import pandas as pd

def optimize_database():
    """데이터베이스 최적화 및 10일 이상 데이터 정리"""
    try:
        conn = sqlite3.connect('trading.db')
        c = conn.cursor()
        
        # 10일 이상 된 데이터 삭제
        c.execute("""
            DELETE FROM price_data 
            WHERE timestamp < datetime('now', '-10 days')
        """)
        
        # 10일 이상 된 거래 기록 삭제 (초기 잔고 제외)
        c.execute("""
            DELETE FROM trades 
            WHERE timestamp < datetime('now', '-10 days')
            AND type != 'INITIAL'
        """)
        
        # 데이터베이스 최적화
        c.execute("VACUUM")
        
        # 인덱스 재생성
        c.execute("DROP INDEX IF EXISTS idx_timestamp")
        c.execute("CREATE INDEX idx_timestamp ON price_data (timestamp)")
        
        # 통계 업데이트
        c.execute("ANALYZE price_data")
        c.execute("ANALYZE trades")
        
        conn.commit()
        
        print("데이터베이스 정리 완료")
        
        return True
        
    except Exception as e:
        print(f"데이터베이스 최적화 중 오류 발생: {e}")
        return False
        
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    optimize_database()