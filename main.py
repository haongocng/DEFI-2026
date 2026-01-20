import asyncio
import re
import os
import sys
import ast
import json

# Thêm đường dẫn hiện tại vào sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tqdm.asyncio import tqdm_asyncio 
from opt.data_loader import DataLoader
from opt.metrics import Metrics
from opt.memory_manager import KnowledgeBase
from opt.agent_retriever import AgentRetriever
from opt.agent_optimizer import AgentOptimizer
from opt.agent_analyst import AgentAnalyst
from opt.eval import Evaluator
from opt.request import TimelyClient

# Giới hạn concurrency để tránh Rate Limit API
CONCURRENT_LIMIT = 2
sem = asyncio.Semaphore(CONCURRENT_LIMIT)

def parse_candidate_list(raw_text):
    """
    Parse danh sách ứng viên, xử lý đặc biệt cho format: [1."Name", 2."Name"]
    """
    candidates = []
    # 1. Tìm chuỗi nằm trong Candidate Set: [...]
    match = re.search(r'Candidate [Ss]et: (\[.*?\])', raw_text, re.DOTALL)
    
    if match:
        list_str = match.group(1)
        

        candidates = re.findall(r'"(.*?)"', list_str)
        
        # Nếu regex trả về rỗng (do format khác), mới thử dùng các cách cũ
        if not candidates:
            try:
                candidates = ast.literal_eval(list_str)
            except:
                try:
                    candidates = json.loads(list_str.replace("'", '"'))
                except:
                    # Fallback cuối cùng: Split thủ công và clean mạnh tay
                    content = list_str.strip('[]')
                    # Xóa cả số và dấu chấm đầu câu nếu có (VD: 1. Name -> Name)
                    candidates = [re.sub(r'^\d+\.\s*"?', '', x.strip()).strip('"') for x in content.split(',')]

    # Lọc phần tử rỗng và tạo chuỗi hiển thị
    candidates = [str(c) for c in candidates if c]
    
    formatted_str = ""
    for i, item in enumerate(candidates):
        formatted_str += f"{i}. {item}\n"
        
    return candidates, formatted_str

def find_real_target_index(target_text, candidates_list):
    """Tìm index của target trong list ứng viên."""
    if not target_text or not candidates_list:
        return -1
        
    # 1. Exact match
    if target_text in candidates_list:
        return candidates_list.index(target_text)
        
    # 2. Normalized match
    norm_target = str(target_text).lower().strip()
    norm_candidates = [str(c).lower().strip() for c in candidates_list]
    
    if norm_target in norm_candidates:
        return norm_candidates.index(norm_target)
        
    # 3. Substring match
    for i, cand in enumerate(norm_candidates):
        if norm_target in cand or cand in norm_target:
            return i
            
    return -1

async def process_single_item(item, loader, retriever, optimizer):
    async with sem:
        max_retries = 3  # Số lần thử lại tối đa
        retry_delay = 5  # Giờ nghỉ giữa các lần thử (giây)
        
        for attempt in range(max_retries):
            try:
                # Mỗi lần thử đều nghỉ một chút để giãn cách
                await asyncio.sleep(retry_delay * (attempt + 1)) 
                
                raw_input = item['input']
                
                # 1. Parse Data
                clean_history = loader.parse_user_history(raw_input)
                formatted_context = loader.format_history_for_llm(clean_history)
                candidates_list, candidate_str_for_llm = parse_candidate_list(raw_input)
                
                if not candidates_list:
                    return None

                # 2. RAG
                relevant_rules = retriever.retrieve_rules(formatted_context)
                
                # 3. Optimize
                rec_ids, reason = await optimizer.get_recommendations_async(
                    formatted_context, relevant_rules, candidate_str_for_llm
                )
                
                # 4. Kiểm tra kết quả
                target_text = item.get('target', 'Unknown')
                real_target_index = find_real_target_index(target_text, candidates_list)
                
                return {
                    "target_id": real_target_index,
                    "target_text": target_text,
                    "predicted_ids": rec_ids,
                    "user_profile": formatted_context, 
                    "applied_rules": relevant_rules
                }
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Lỗi API (Lần {attempt+1}): {e}. Đang thử lại sau {retry_delay * (attempt + 2)}s...")
                    continue
                else:
                    print(f"❌ Thất bại sau {max_retries} lần thử: {e}")
                    return None

async def run_learning_phase(analyst, kb, retriever, error_list):
    """
    Quy trình Tự học: Phân tích lỗi -> Tạo Rule -> Hợp nhất vào KB -> Sync Vector DB
    """
    print("\n" + "="*40)
    print("🧠 STARTING SELF-CORRECTION LOOP (LEARNING PHASE)")
    print("="*40)
    
    if not error_list:
        print("🎉 No errors found! Nothing to learn.")
        return

    # 1. Phân tích lỗi (Global Analysis)
    # Gom nhóm các lỗi và đề xuất Rules tổng quát
    print(f"🔍 Analyzing {len(error_list)} failures...")
    new_rule_proposals = await analyst.analyze_global_failures_async(error_list, sample_size=10)
    
    if not new_rule_proposals:
        print("⚠️ Analyst could not derive new rules.")
        return

    print(f"💡 Analyst proposed {len(new_rule_proposals)} new rules.")

    # 2. Hợp nhất Rules (Consolidate)
    # So sánh với Rules cũ để quyết định (ADD / REPLACE / IGNORE)
    existing_rules = kb.get_all_rules()
    
    for prop in new_rule_proposals:
        print(f"\n--- Processing Proposal: {prop.get('condition_description')} ---")
        
        decision = await analyst.consolidate_rules_async(prop, existing_rules)
        action = decision.get('action', 'IGNORE')
        
        if action == 'ADD':
            kb.add_rule(prop['condition_description'], prop['instruction'])
            
        elif action == 'REPLACE':
            target_id = decision.get('target_id')
            # Kiểm tra target_id hợp lệ
            if target_id is not None and isinstance(target_id, int):
                 kb.replace_rule(target_id, prop['condition_description'], prop['instruction'])
            else:
                print("⚠️ Replace action requested but target_id invalid. Adding as new instead.")
                kb.add_rule(prop['condition_description'], prop['instruction'])
                
        else:
            print("Action: IGNORE (Rule is duplicate or weak)")

    print("\n🔄 Syncing new Knowledge Base to Vector DB...")
    retriever.sync_kb_to_vector_db()
    print("✅ Learning Phase Complete!")

async def main_async():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'test_dataset.json') 
    print(f"📂 Reading Data: {data_path}")

    # 1. Setup Components
    try:
        client = TimelyClient(model_name="gpt-4o-mini") 
        print(f"🔌 API Connected.")
    except Exception as e:
        print(f"❌ API Error: {e}")
        return

    kb = KnowledgeBase() # Load rules.json
    metrics = Metrics()
    

    retriever = AgentRetriever(client, kb)
    print("📥 Initializing Knowledge Base...")
    retriever.sync_kb_to_vector_db()

    optimizer = AgentOptimizer(client)
    analyst = AgentAnalyst(client)
    loader = DataLoader(data_path)
    
    try:
        raw_data = loader.load_data()
    except:
        print("❌ File not found.")
        return

    print(f"✅ Loaded {len(raw_data)} items. Running Inference...")

    # 2. Inference Loop
    tasks = [process_single_item(item, loader, retriever, optimizer) for item in raw_data]
    results = await tqdm_asyncio.gather(*tasks)
    none_count = sum(1 for r in results if r is None)
    minus_one_count = sum(1 for r in results if r is not None and r['target_id'] == -1)
    print(f"--- DEBUG: {none_count} mẫu lỗi API, {minus_one_count} mẫu không khớp Target ---")
    valid_results = [r for r in results if r is not None and r['target_id'] != -1]
    
    # 3. Evaluation & Metrics
    print(f"\n📊 Calculating metrics on {len(valid_results)}/{len(results)} valid items...")
    
    evaluator = Evaluator(retriever, optimizer, analyst, kb, metrics)
    report, error_list = evaluator.calculate_metrics_from_results(valid_results, k=10)
    
    print("="*40)
    print(f"🎯 CURRENT PERFORMANCE:")
    for k in [1, 5, 10]:
        print(f"   --- Top-{k} ---")
        print(f"   Hit Rate @{k} : {report.get(f'Hit@{k}', 0):.4f}")
        print(f"   NDCG @{k}     : {report.get(f'NDCG@{k}', 0):.4f}")
        print(f"   MAP @{k}      : {report.get(f'MAP@{k}', 0):.4f}")
    print("="*40)


    await run_learning_phase(analyst, kb, retriever, error_list)

if __name__ == "__main__":
    asyncio.run(main_async())