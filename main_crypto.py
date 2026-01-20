import asyncio
import time # Dùng để tính toán thời gian nếu cần
from tqdm.asyncio import tqdm_asyncio 
from opt.crypto_adapter import CryptoAdapter 
from opt.metrics import Metrics
from opt.memory_manager import KnowledgeBase
from opt.agent_retriever import AgentRetriever
from opt.agent_optimizer import AgentOptimizer
from opt.agent_analyst import AgentAnalyst
from opt.eval import Evaluator
from opt.request import TimelyClient
from opt.benchmark import TraditionalBenchmark, TradingEvaluator
# --- CẤU HÌNH RATE LIMIT & DEBUG ---
CONCURRENT_LIMIT = 3       # <--- Giảm xuống 3 để tránh spam API quá nhanh
DELAY_SECONDS = 4.0       # <--- Nghỉ 2 giây giữa mỗi lần xử lý xong 1 nến
DEBUG_MODE = True         
sem = asyncio.Semaphore(CONCURRENT_LIMIT)

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
CYAN = "\033[96m"

async def process_trade_signal(item, retriever, optimizer):
    """Xử lý 1 nến lịch sử để xem AI quyết định thế nào"""
    raw = item.get('raw_row', {})
    rsi = raw.get('RSI', 50) 
    
    if rsi > 45: 
        return {
            "target_id": item['target_id'],
            "predicted_ids": [], # Rỗng nghĩa là không mua
            "status": "SKIPPED_BY_FILTER" # Đánh dấu để biết
        }
    
    async with sem:
        try:
            market_context = item['input']
            candidate_str = item['candidate_set'] 
            
            # 1. Retrieve Rules
            relevant_rules = retriever.retrieve_rules(market_context,top_k=3)
            
            # 2. Optimizer (AI Trader ra quyết định)
            rec_ids, reason = await optimizer.get_recommendations_async(
                market_context, relevant_rules, candidate_str
            )
            
            # --- DEBUGGING: IN RA MÀN HÌNH ---
            if DEBUG_MODE:
                # Lấy ID dự đoán đầu tiên (hoặc -1 nếu rỗng)
                pred_id = rec_ids[0] if rec_ids else -1
                target_id = item['target_id']
                
                # So sánh kết quả để tô màu
                is_correct = str(pred_id) == str(target_id)
                color = GREEN if is_correct else RED
                icon = "✅" if is_correct else "❌"
                
                print(f"\n{color}{icon} Timestamp: {item['raw_row']['time']} | AI: {pred_id} vs Real: {target_id}{RESET}")
                print(f"   📝 {CYAN}AI Reason:{RESET} {reason}")
                print(f"   📚 Rules used: {len(relevant_rules)}")
                # print("-" * 50) 

            await asyncio.sleep(DELAY_SECONDS) 

            return {
                "target_id": item['target_id'],      
                "target_text": item['target_text'],  
                "predicted_ids": rec_ids,            
                "user_profile": market_context, 
                "timestamp": item['raw_row']['time'],     
                "applied_rules": relevant_rules
            }
        except Exception as e:
            print(f"{RED}Lỗi xử lý item: {e}{RESET}")
            return None

async def main_crypto_loop():

    client = TimelyClient(model_name="gpt-4o-mini") 
    print(f"🔌 Connected via Session ID: {client.session_id}")
    
    kb = KnowledgeBase(storage_path="./memory/crypto_rules.json") 
    metrics = Metrics()
    
    optimizer = AgentOptimizer(client)
    analyst = AgentAnalyst(client)
    retriever = AgentRetriever(client, kb)
    

    adapter = CryptoAdapter(symbol='ETH-USDT', timeframe='1hour', tp_pct=1.5, hold_candles=24)
    
    try:
        all_data = adapter.load_and_label_data(days_back=60)
        train_data = all_data[-50:] 
    except Exception as e:
        print(f"{RED}Không tải được dữ liệu: {e}{RESET}")
        return

    print(f"\n⚡ [PHASE 1] Backtest AI trên {len(train_data)} nến lịch sử...")
    
    tasks = [process_trade_signal(item, retriever, optimizer) for item in train_data]
    results = await tqdm_asyncio.gather(*tasks)
    valid_results = [r for r in results if r is not None]

    evaluator = Evaluator(retriever, optimizer, analyst, kb, metrics)
    report, error_list = evaluator.calculate_metrics_from_results(valid_results, k=1)
    
    print(f"\n📊 KẾT QUẢ AI: Accuracy={report.get('Hit_Rate', 0):.2f}")


    if error_list:
        print(f"\n🧠 [PHASE 2] Phát hiện {len(error_list)} quyết định sai lầm.")
        print(f"   🔎 Analyst đang phân tích nguyên nhân...")
        
        general_rules = await analyst.analyze_global_failures_async(error_list, sample_size=10)
        
        if general_rules:
            print(f"   -> Đề xuất {len(general_rules)} quy tắc mới. Đang kiểm tra mâu thuẫn...")
            existing_rules = kb.get_all_rules()
            
            for rule in general_rules:
                decision = await analyst.consolidate_rules_async(rule, existing_rules)
                action = decision.get('action', 'IGNORE')
                
                if action == 'ADD':
                    kb.add_rule(rule['condition_description'], rule['instruction'])
                    existing_rules.append(rule) 
                    print(f"   ✅ {GREEN}[ADD]{RESET} {rule['instruction']}")
                    
                elif action == 'REPLACE':
                    t_id = decision.get('target_id')
                    if t_id is not None:
                        kb.replace_rule(t_id, rule['condition_description'], rule['instruction'])
                        if 0 <= t_id < len(existing_rules):
                            existing_rules[t_id] = rule
                        print(f"   🔄 {CYAN}[REPLACE]{RESET} Rule #{t_id} -> {rule['instruction']}")
                    
                elif action == 'IGNORE':
                    print(f"   🚫 {RED}[IGNORE]{RESET} Rule bị trùng hoặc yếu hơn.")
            
            retriever.sync_kb_to_vector_db()
            print("💾 Đã đồng bộ hóa dữ liệu tri thức mới.")
        else:
            print("⚠️ Analyst không tìm ra quy tắc tổng quát nào.")
    else:
        print("🎉 Chúc mừng! Hệ thống hoạt động hoàn hảo (0 lỗi).")

    # ==============================================================================
    # [PHASE 3] BENCHMARK COMPARISON (SO SÁNH VỚI CHIẾN THUẬT CỔ ĐIỂN)
    # ==============================================================================
    print("\n  [PHASE 3] SO SÁNH HIỆU SUẤT VỚI TRADITIONAL STRATEGY...")

    # 1. Thu thập tín hiệu MUA của AI (Lọc từ kết quả chạy Phase 1)
    ai_buy_signals = []
    for res in valid_results:
        # Nếu AI dự đoán là 1 (BUY) -> Lưu lại timestamp
        if res.get('predicted_ids') and str(res['predicted_ids'][0]) == '1':
            # Lưu ý: Cần đảm bảo process_trade_signal trả về field 'timestamp'
            # Nếu chưa có, fallback lấy từ raw_row nếu có
            if 'timestamp' in res:
                ai_buy_signals.append(res['timestamp'])
            elif 'raw_row' in res and 'time' in res['raw_row']: # Fallback
                ai_buy_signals.append(res['raw_row']['time'])


    benchmark_engine = TraditionalBenchmark(tp_pct=1.5, hold_hours=24)
    evaluator_bench = TradingEvaluator(initial_capital=1000)

    if not adapter.raw_df.empty:
        # Bước A: Tính toán chỉ báo trên TOÀN BỘ dữ liệu gốc
        full_prepared_df = benchmark_engine.prepare_data(adapter.raw_df)
        
        bench_df = full_prepared_df.iloc[-50:].copy()
        
        # --- SỬA LOGIC SO SÁNH ---
        print(f"   Dữ liệu Benchmark: {len(bench_df)} nến. Số tín hiệu AI: {len(ai_buy_signals)}")
        
        # Tính PnL cho AI (bench_df đã có index là time nhờ prepare_data ở trên)
        ai_trades_df = benchmark_engine.calculate_ai_pnl(bench_df, ai_buy_signals)
        
        # So sánh (bench_df đã có sẵn chỉ báo, run_backtest sẽ dùng lại)
        evaluator_bench.compare(ai_trades_df, bench_df, benchmark_engine)
        
    else:
        print(f"{RED}❌ Không có dữ liệu gốc (adapter.raw_df) để chạy benchmark.{RESET}")
if __name__ == "__main__":
    asyncio.run(main_crypto_loop())