import pandas as pd
import numpy as np

# ==============================================================================
# 1. BACKTEST ENGINE (LÕI GIẢ LẬP)
# ==============================================================================
def run_simulation(df: pd.DataFrame, rule_func, config: dict):
    """
    Chạy giả lập chiến thuật trên dữ liệu lịch sử.
    
    Tham số:
    - df: DataFrame chứa dữ liệu nến (cần có các cột chỉ báo đã tính).
    - rule_func: Hàm logic (nhận vào 1 dòng 'row', trả về True/False).
                 Đây là nơi bạn đưa Rule tự học của AI vào.
    - config: Cấu hình {'tp_pct': 0.5, 'hold_candles': 12}.
    
    Trả về:
    - List các lệnh đã thực hiện (Dictionary).
    """
    # Chuyển DataFrame sang list dictionary để loop cho nhanh (giống logic file gốc)
    records = df.reset_index().to_dict('records')
    total_len = len(records)
    
    tp_pct = config.get('tp_pct', 0.5)      # Mặc định TP 0.5%
    hold_candles = config.get('hold_candles', 12) # Mặc định giữ 12 nến
    
    history = []
    i = 0
    
    # [Logic gốc: Loop qua từng nến]
    while i < total_len - hold_candles:
        row = records[i]
        
        # 1. Kiểm tra Điều kiện vào lệnh (Gọi hàm Rule từ bên ngoài vào)
        # Hệ thống của bạn sẽ cập nhật logic bên trong hàm rule_func này
        is_buy = False
        try:
            is_buy = rule_func(row)
        except Exception:
            is_buy = False # Bỏ qua nếu lỗi dữ liệu
            
        if is_buy:
            entry_price = row['Close']
            entry_time = row['time'] if 'time' in row else i
            pnl = 0
            outcome = "LOSS"
            hit_tp = False
            
            # [Logic gốc: Giữ lệnh và check TP]
            for j in range(1, hold_candles + 1):
                # Kiểm tra giá High của các nến tương lai
                current_high = records[i + j]['High']
                
                if current_high >= entry_price * (1 + tp_pct/100):
                    pnl = tp_pct
                    outcome = "WIN"
                    hit_tp = True
                    i += j # Nhảy cóc qua giai đoạn giữ lệnh
                    break
            
            # [Logic gốc: Time Exit - Hết giờ thì bán]
            if not hit_tp:
                exit_price = records[i + hold_candles]['Close']
                pnl = (exit_price - entry_price) / entry_price * 100
                outcome = "WIN" if pnl > 0 else "LOSS"
                i += hold_candles # Nhảy tới nến bán
                
            # Ghi lại nhật ký
            history.append({
                'Entry_Time': entry_time,
                'Entry_Price': entry_price,
                'Result': outcome,
                'PnL': pnl,
                # Lưu thêm context để hệ thống học tiếp nếu cần
                'Context_RSI': row.get('RSI', 0),
                'Context_ADX': row.get('ADX', 0)
            })
            
            # Vào lệnh xong thì skip 1 nến tránh vào lệnh trùng
            if not hit_tp: i += 1 
        else:
            i += 1
            
    return history

# ==============================================================================
# 2. METRICS EVALUATOR (CHẤM ĐIỂM)
# ==============================================================================
def evaluate_metrics(trade_history: list) -> dict:
    """
    Tính toán các chỉ số đánh giá hiệu quả từ lịch sử giao dịch.
    """
    if not trade_history:
        return {
            "Total_Orders": 0,
            "Win_Rate": 0.0,
            "Avg_PnL": 0.0,
            "Total_Return": 0.0,
            "Evaluation": "NO_DATA"
        }
        
    df_res = pd.DataFrame(trade_history)
    
    # [Logic gốc: Tính Winrate và Avg PnL]
    total_orders = len(df_res)
    win_rate = (df_res['PnL'] > 0).mean() * 100
    avg_pnl = df_res['PnL'].mean()
    total_return = df_res['PnL'].sum()
    
    # Đánh giá sơ bộ (Rule-based evaluation)
    eval_score = "NEUTRAL"
    if win_rate > 65 and total_orders > 10:
        eval_score = "EXCELLENT"
    elif win_rate > 55:
        eval_score = "GOOD"
    elif win_rate < 40:
        eval_score = "BAD"
        
    return {
        "Total_Orders": total_orders,
        "Win_Rate": round(win_rate, 2),
        "Avg_PnL": round(avg_pnl, 4),
        "Total_Return": round(total_return, 2),
        "Evaluation": eval_score
    }