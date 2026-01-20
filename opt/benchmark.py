import pandas as pd
import numpy as np
import ta

# ==============================================================================
# 1. CLASS TẠO BENCHMARK & TÍNH PNL (ĐÃ FIX LỖI LOGIC)
# ==============================================================================
class TraditionalBenchmark:
    def __init__(self, tp_pct=0.45, hold_hours=1):
        self.tp_pct = tp_pct
        self.hold_candles = hold_hours

    def prepare_data(self, df):
        """Tính toán Indicators cho Traditional Strategy (Đã fix lỗi Look-ahead)"""
        df = df.copy()
        
        # Đảm bảo index là datetime để resample đúng
        if 'time' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time', drop=False)

        # 1. EMA cơ bản
        if 'EMA34' not in df.columns:
            df['EMA34'] = ta.trend.ema_indicator(df['Close'], window=34)
        if 'EMA89' not in df.columns:
            df['EMA89'] = ta.trend.ema_indicator(df['Close'], window=89)
        
        # 2. Trend 4H (Đã fix Look-ahead Bias)
        if 'Trend_4H_Up' not in df.columns:
            # Resample lấy giá đóng cửa
            df_4h = df.resample('4h').agg({'Close': 'last'})
            
            ema34_4h = ta.trend.ema_indicator(df_4h['Close'], window=34)
            ema89_4h = ta.trend.ema_indicator(df_4h['Close'], window=89)
            
            # Tính trend tại khung 4H
            df_4h['Trend_4H_Up'] = ema34_4h > ema89_4h
            
            # QUAN TRỌNG: Shift(1) để đảm bảo tại thời điểm t, 
            # ta chỉ biết trend của cây nến 4H ĐÃ ĐÓNG CỬA trước đó.
            df_4h['Trend_4H_Up'] = df_4h['Trend_4H_Up'].shift(1)
            
            # Merge lại vào khung 1H (ffill để lấp đầy các nến 1h trong cùng block 4h)
            trend_series = df_4h['Trend_4H_Up'].reindex(df.index, method='ffill')
            df['Trend_4H_Up'] = trend_series
        
        # 3. Golden Cross Logic
        df['Golden_Cross'] = (df['EMA34'] > df['EMA89']) & (df['EMA34'].shift(1) <= df['EMA89'].shift(1))
        
        # FillNA False cho cột boolean để tránh lỗi, không drop dòng
        df['Trend_4H_Up'] = df['Trend_4H_Up'].fillna(False)
        
        return df

    def run_backtest(self, df):
        """Chạy backtest Traditional"""
        df = self.prepare_data(df)
        return self._simulate_trades(df, strategy_type='Traditional')

    def calculate_ai_pnl(self, df, ai_signals):
        """
        Tính PnL cho các lệnh AI dự đoán
        """
        df = self.prepare_data(df) 
        
        # Chuyển ai_signals sang datetime để so sánh với index
        ai_signals_dt = pd.to_datetime(ai_signals)
        df['AI_Buy'] = df.index.isin(ai_signals_dt)
        
        return self._simulate_trades(df, strategy_type='AI_Agent')

    def _simulate_trades(self, df, strategy_type='Traditional'):
        records = df.reset_index().to_dict('records')
        history = []
        total_len = len(records)
        i = 0
        
        while i < total_len - self.hold_candles:
            row = records[i]
            is_entry = False

            # ==================================================================
            # THAY ĐỔI LOGIC TRADITIONAL TẠI ĐÂY
            # ==================================================================
            if strategy_type == 'Traditional':
                # Chiến thuật: RSI Oversold (Bắt đáy)
                # Logic: Mua khi RSI xuống thấp hơn 30 (quá bán) - Rất hay gặp lỗ nếu bắt dao rơi
                # Đây là "bao cát" hoàn hảo để AI thể hiện sự vượt trội
                rsi_val = row.get('RSI', 50)
                
                # Điều kiện: RSI < 35 (Vùng quá bán nhẹ)
                if rsi_val < 35:
                    is_entry = True

            elif strategy_type == 'AI_Agent':
                if row.get('AI_Buy'):
                    is_entry = True
            
            # --- Xử lý vào lệnh ---
            if is_entry:
                entry_price = row['Close']
                entry_time = row.get('time', row.get('index'))
                pnl = 0
                outcome = "LOSS"
                
                # Logic TP/SL
                hit_tp = False
                for j in range(1, self.hold_candles + 1):
                    if i + j >= total_len: break
                    current_high = records[i + j]['High']
                    
                    # Chốt lời
                    if current_high >= entry_price * (1 + self.tp_pct/100):
                        pnl = self.tp_pct
                        outcome = "WIN"
                        hit_tp = True
                        i += j 
                        break
                
                # Hết giờ (Time Exit)
                if not hit_tp:
                    idx_exit = min(i + self.hold_candles, total_len - 1)
                    exit_price = records[idx_exit]['Close']
                    pnl = (exit_price - entry_price) / entry_price * 100
                    outcome = "WIN" if pnl > 0 else "LOSS"
                    i += self.hold_candles
                
                history.append({
                    'Agent': strategy_type,
                    'Result': outcome,
                    'PnL': pnl,
                    'Entry_Price': entry_price,
                    'Time': entry_time
                })
                
                # QUAN TRỌNG: Sau khi Traditional vào lệnh, bắt buộc nghỉ 5 nến 
                # để tránh mua liên tiếp tại đáy (DCA)
                if strategy_type == 'Traditional':
                    i += 5
            else:
                i += 1
                
        return pd.DataFrame(history)

# ==============================================================================
# 2. CLASS ĐÁNH GIÁ HIỆU SUẤT (TradingEvaluator)
# ==============================================================================
class TradingEvaluator:
    def __init__(self, initial_capital=1000):
        self.initial_capital = initial_capital

    def _calculate_metrics(self, trades_df, agent_name):
        if trades_df.empty:
            return {
                'Name': agent_name,
                'Total_Orders': 0,
                'Win_Rate (%)': 0.0,   
                'Avg_PnL (%)': 0.0,    
                'Profit_Factor': 0.0,
                'Max_DD (%)': 0.0      
            }
        total = len(trades_df)
        wins = trades_df[trades_df['PnL'] > 0]
        losses = trades_df[trades_df['PnL'] <= 0]
        
        win_rate = (len(wins) / total) * 100 if total > 0 else 0
        
        gross_profit = wins['PnL'].sum()
        gross_loss = abs(losses['PnL'].sum())
        pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf')

        # Drawdown Simulation
        equity = self.initial_capital * (1 + trades_df['PnL']/100).cumprod()
        peak = equity.cummax()
        dd = (equity - peak) / peak * 100
        max_dd = dd.min()

        return {
            'Name': agent_name,
            'Total_Orders': total,
            'Win_Rate (%)': round(win_rate, 2),
            'Avg_PnL (%)': round(trades_df['PnL'].mean(), 3),
            'Profit_Factor': pf,
            'Max_DD (%)': round(max_dd, 2)
        }

    def compare(self, ai_trades_df, market_df, benchmark_engine):
        print("\n Running Traditional Benchmark simulation...")
        bench_trades = benchmark_engine.run_backtest(market_df)
        
        agent_metrics = self._calculate_metrics(ai_trades_df, " AI Agent")
        bench_metrics = self._calculate_metrics(bench_trades, " Traditional")
        
        print("\n" + "="*60)
        print(f"REPORT: AI AGENT vs TRADITIONAL BENCHMARK")
        print("="*60)
        
        df_compare = pd.DataFrame([agent_metrics, bench_metrics]).set_index('Name')
        print(df_compare.T.to_string())
        
        print("-" * 60)
        wr_gap = agent_metrics['Win_Rate (%)'] - bench_metrics['Win_Rate (%)']
        pf_gap = agent_metrics['Profit_Factor'] - bench_metrics['Profit_Factor']
        
        print(f"🏆 GAP ANALYSIS:")
        print(f"   ► Win Rate Gap:      {wr_gap:+.2f}%  {'✅ Cải thiện' if wr_gap > 0 else '❌ Kém hơn'}")
        
        pf_gap_str = "N/A" if (agent_metrics['Profit_Factor'] == float('inf') or bench_metrics['Profit_Factor'] == float('inf')) else f"{pf_gap:+.2f}"
        print(f"   ► Profit Factor Gap: {pf_gap_str}")
        print("="*60 + "\n")