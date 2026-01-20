import pandas as pd
from crypto.data_loader import fetch_raw_candles, calculate_indicators

class CryptoAdapter:
    def __init__(self, symbol='BTC-USDT', timeframe='15min', tp_pct=0.5, hold_candles=12):
        self.symbol = symbol
        self.timeframe = timeframe
        self.tp_pct = tp_pct
        self.hold_candles = hold_candles
        self.raw_df = pd.DataFrame()
    def load_and_label_data(self, days_back=30):

        print(f"📥 Đang tải dữ liệu {self.symbol} ({self.timeframe})...")
        raw_df = fetch_raw_candles(self.symbol, interval=self.timeframe, days_back=days_back)
        if raw_df.empty:
            raise Exception("Không tải được dữ liệu!")
            
        df = calculate_indicators(raw_df)
        self.raw_df = df.copy()
        records = df.reset_index().to_dict('records')
        labeled_data = []

        total_len = len(records)
        
        for i in range(total_len - self.hold_candles):
            row = records[i]
            entry_price = row['Close']
            
            hit_tp = False
            for j in range(1, self.hold_candles + 1):
                if records[i + j]['High'] >= entry_price * (1 + self.tp_pct/100):
                    hit_tp = True
                    break
            
            target_action = "BUY" if hit_tp else "WAIT"
            
            market_context = (
                f"Symbol: {self.symbol} | Time: {row['time']}\n"
                f"Price: {row['Close']} (BB Band Width: {self._get_bb_width(row)}%)\n"
                f"Indicators:\n"
                f"- RSI: {row['RSI']:.2f}\n"
                f"- ADX: {row['ADX']:.2f}\n"
                f"- Trend Short (EMA34/89): {'UP' if row['EMA34'] > row['EMA89'] else 'DOWN'}\n"
                f"- Trend Long (4H): {'UP' if row['Trend_4H_Up'] else 'DOWN'}\n"
                f"- BB Position: {'Near Low' if row['Close'] <= row['BB_Low']*1.002 else 'Normal'}"
            )

            labeled_data.append({
                "input": market_context,          
                "candidate_set": "Options: [BUY, WAIT]", 
                "target_text": target_action,    
                "target_id": 1 if target_action == "BUY" else 0,
                "raw_row": row               
            })
            
        return labeled_data

    def _get_bb_width(self, row):
        if row['BB_Mid'] == 0: return 0
        return round((row['BB_High'] - row['BB_Low']) / row['BB_Mid'] * 100, 2)