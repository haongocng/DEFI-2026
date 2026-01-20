import requests
import pandas as pd
import ta
import time
from datetime import datetime, timedelta

# ==============================================================================
# 1. DATA FETCHER (LẤY DỮ LIỆU)
# ==============================================================================
def fetch_raw_candles(symbol: str, interval: str = '5min', days_back: int = 7) -> pd.DataFrame:
    """
    Lấy dữ liệu nến từ KuCoin.
    Return: DataFrame sạch (Open, High, Low, Close, Volume) hoặc DataFrame rỗng nếu lỗi.
    """
    base_url = 'https://api.kucoin.com/api/v1/market/candles'
    end_time = int(time.time())
    start_time = int((datetime.now() - timedelta(days=days_back)).timestamp())
    
    all_data = []
    current_end = end_time
    
    while True:
        params = {
            'symbol': symbol, 'type': interval, 
            'startAt': start_time, 'endAt': current_end
        }
        try:
            resp = requests.get(base_url, params=params, timeout=5)
            data = resp.json()
            if 'data' not in data or not data['data']: break
            
            candles = data['data']
            all_data.extend(candles)
            
            last_ts = int(candles[-1][0])
            current_end = last_ts
            if last_ts <= start_time: break
            time.sleep(0.05) # Rate limit protection
        except: break

    if not all_data: return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=['time', 'Open', 'Close', 'High', 'Low', 'Volume', 'Turnover'])
    df['time'] = pd.to_datetime(df['time'].astype(int), unit='s')
    df = df.set_index('time').sort_index()
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

# ==============================================================================
# 2. DATA PROCESSOR (XỬ LÝ CHỈ BÁO)
# ==============================================================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm các chỉ báo kỹ thuật vào DataFrame.
    Logic: EMA Trend, RSI, Bollinger Bands, ADX, Trend 4H.
    """
    if df.empty: return df
    
    # --- TREND ---
    df['EMA34'] = ta.trend.ema_indicator(df['Close'], window=34)
    df['EMA89'] = ta.trend.ema_indicator(df['Close'], window=89)
    
    # --- MOMENTUM & STRENGTH ---
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)

    # --- VOLATILITY (BOLLINGER) ---
    bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Mid'] = bb.bollinger_mavg() # SMA 20

    # --- MULTI-TIMEFRAME CONTEXT (4H Trend) ---
    # Resample ra khung 4H để xem xu hướng lớn
    df_4h = df.resample('4h').agg({'Close': 'last'}).dropna()
    ema34_4h = ta.trend.ema_indicator(df_4h['Close'], window=34)
    ema89_4h = ta.trend.ema_indicator(df_4h['Close'], window=89)
    
    # Map logic: True nếu Trend 4H là Tăng
    df_4h['Trend_4H_Up'] = ema34_4h > ema89_4h
    
    # Merge lại vào khung gốc (Forward Fill để tránh bias tương lai)
    df = df.join(df_4h[['Trend_4H_Up']].reindex(df.index, method='ffill'))

    return df.dropna()

# ==============================================================================
# 3. AGENT INTERFACE (OUTPUT JSON/DICT)
# ==============================================================================
def get_market_snapshot(symbol: str) -> dict:
    """
    Hàm này thiết kế riêng cho Agent Workflow.
    Nó gọi 2 hàm trên và trả về 1 Dictionary chứa trạng thái thị trường HIỆN TẠI.
    Agent có thể đọc dict này để đưa vào prompt.
    """
    # 1. Lấy dữ liệu (mặc định 5 ngày để đủ tính chỉ báo 4H)
    raw = fetch_raw_candles(symbol, interval='5min', days_back=5)
    if raw.empty: return {"error": "No data"}

    # 2. Xử lý
    processed = calculate_indicators(raw)
    if processed.empty: return {"error": "Not enough data for indicators"}

    # 3. Lấy cây nến vừa đóng (gần nhất)
    last = processed.iloc[-1]
    
    # 4. Đóng gói thông tin quan trọng cho AI
    # Tính toán khoảng cách tới Bollinger Bands (để AI biết giá rẻ hay đắt)
    dist_to_low_bb = (last['Close'] - last['BB_Low']) / last['BB_Low'] * 100
    
    snapshot = {
        "symbol": symbol,
        "timestamp": str(last.name),
        "price": {
            "current": last['Close'],
            "open": last['Open'],
            "high": last['High'],
            "low": last['Low']
        },
        "indicators": {
            "rsi": round(last['RSI'], 2),
            "adx": round(last['ADX'], 2),
            "trend_short": "UP" if last['EMA34'] > last['EMA89'] else "DOWN",
            "trend_long_4h": "UP" if last['Trend_4H_Up'] else "DOWN"
        },
        "volatility_context": {
            "bb_low_price": round(last['BB_Low'], 2),
            "bb_high_price": round(last['BB_High'], 2),
            "is_price_near_bottom": dist_to_low_bb < 0.2, # True nếu giá sát hoặc thủng đáy BB (biên độ 0.2%)
            "band_width_percent": round((last['BB_High'] - last['BB_Low']) / last['BB_Mid'] * 100, 2)
        }
    }
    return snapshot