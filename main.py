# File: ai_agent/main.py
import asyncio
import re
import time
from tqdm.asyncio import tqdm_asyncio 
from opt.data_loader import DataLoader
from opt.metrics import Metrics
from opt.memory_manager import KnowledgeBase
from opt.agent_retriever import AgentRetriever
from opt.agent_optimizer import AgentOptimizer
from opt.agent_analyst import AgentAnalyst
from opt.eval import Evaluator

from opt.request import TimelyClient

CONCURRENT_LIMIT = 5 
sem = asyncio.Semaphore(CONCURRENT_LIMIT)

async def process_single_item(item, loader, retriever, optimizer):
    """Xử lý 1 dòng dữ liệu (Async)"""
    async with sem:
        try:
            raw_input = item['input']
            
            # 1. Parse dữ liệu
            clean_history = loader.parse_user_history(raw_input)
            formatted_context = loader.format_history_for_llm(clean_history)
            
            candidate_match = re.search(r'Candidate set: \[(.*?)\]', raw_input, re.DOTALL)
            candidate_str = candidate_match.group(1) if candidate_match else ""
            

            relevant_rules = retriever.retrieve_rules(formatted_context)
            

            rec_ids, reason = await optimizer.get_recommendations_async(
                formatted_context, relevant_rules, candidate_str
            )
            
            return {
                "target_id": item['target_index'],
                "target_text": item['target'],
                "predicted_ids": rec_ids,
                "user_profile": formatted_context, 
                "applied_rules": relevant_rules
            }
        except Exception as e:
            # print(f"Lỗi item: {e}") 
            return None

async def main_async():
    

    client = TimelyClient(model_name="gpt-4.1") 
    print(f"🔌 Connected via Session ID: {client.session_id}")
    
    kb = KnowledgeBase()
    metrics = Metrics()
    
    optimizer = AgentOptimizer(client)
    analyst = AgentAnalyst(client)
    retriever = AgentRetriever(client, kb)
    
    data_path = 'data/train_50.json' 
    try:
        loader = DataLoader(data_path)
        raw_data = loader.load_data()
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file dữ liệu tại: {data_path}")
        return

    print(f"\n⚡ [PHASE 1] Đang xử lý {len(raw_data)} mẫu dữ liệu song song...")
    
    tasks = [process_single_item(item, loader, retriever, optimizer) for item in raw_data]
    
    # Chạy progress bar
    results = await tqdm_asyncio.gather(*tasks)
    valid_results = [r for r in results if r is not None]
    
    evaluator = Evaluator(retriever, optimizer, analyst, kb, metrics)
    report, error_list = evaluator.calculate_metrics_from_results(valid_results, k=10)
    print(f"📊 KẾT QUẢ V1: Hit Rate={report.get('Hit_Rate', 0):.2f}, NDCG={report.get('NDCG', 0):.2f}, MRR={report.get('MRR', 0):.2f}")
    if error_list:
        print(f"\n🧠 [PHASE 2] Phát hiện {len(error_list)} lỗi.")
        
        general_rules = await analyst.analyze_global_failures_async(error_list, sample_size=15)
        
        print(f"   -> Analyst đề xuất {len(general_rules)} quy tắc tổng quát.")
        print("="*60)
        
        existing_rules = kb.get_all_rules()
        
        for rule in general_rules:
            decision = await analyst.consolidate_rules_async(rule, existing_rules)
            action = decision.get('action', 'IGNORE')
            
            if action == 'ADD':
                kb.add_rule(rule['condition_description'], rule['instruction'])
                existing_rules.append(rule)
                print(f"✅ [ADD] {rule['condition_description']}")
                
            elif action == 'REPLACE':
                t_id = decision.get('target_id')
                kb.replace_rule(t_id, rule['condition_description'], rule['instruction'])
                if 0 <= t_id < len(existing_rules): existing_rules[t_id] = rule
                print(f"🔄 [REPLACE] ID #{t_id} -> New Rule.")
            
            retriever.sync_kb_to_vector_db()


if __name__ == "__main__":
    asyncio.run(main_async())