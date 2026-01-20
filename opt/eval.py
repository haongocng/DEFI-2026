import json
import time
from tqdm import tqdm
from .metrics import Metrics
from .utils import Utils

class Evaluator:
    def __init__(self, retriever, optimizer, analyst, kb, metrics):
        self.retriever = retriever
        self.optimizer = optimizer
        self.analyst = analyst
        self.kb = kb
        self.metrics = metrics
        self.error_list = [] 

    def process_data(self, test_data, k=10, candidate_size=20):
        results = []
        self.error_list = []

        print(f"--- Đang chạy đánh giá trên {len(test_data)} mẫu ---")
        for item in tqdm(test_data):
            user_input = item['input']
            candidate_set = item.get('candidate_set', "") 
            target_id = item.get('target_id') 
            target_text = item.get('target_text', 'Không rõ')

            # 1. Retriever
            rules = self.retriever.retrieve_rules(user_input)

            # 2. Optimizer
            preds, reason = self.optimizer.get_recommendations(user_input, rules, candidate_set)

            # 3. Ghi nhận kết quả
            results.append({
                'target_id': target_id,
                'predicted_ids': preds
            })

            # 4. Kiểm tra lỗi (Dùng k để xác định xem có coi là lỗi hay không để feedback loop)
            if target_id not in preds[:k]:
                self.record_error(user_input, rules, preds, target_text)
                
            time.sleep(2) 

        # SỬ DỤNG LOGIC MỚI: Truyền candidate_size thay vì k đơn lẻ
        report = self.metrics.evaluate_batch(results, candidate_size=candidate_size)
        return report

    def record_error(self, user_input, rules, preds, target_text):
        self.error_list.append({
            "user_profile": user_input,
            "applied_rules": rules,
            "predicted_ids": preds,
            "target_text": target_text
        })

    def update_knowledge_base(self, top_n_errors=3):
        if not self.error_list:
            print("Hệ thống không có lỗi nào để phân tích.")
            return

        print(f"--- Đang kích hoạt Feedback Loop (Phân tích {top_n_errors} lỗi điển hình) ---")
        
        errors_to_analyze = self.error_list[:top_n_errors]
        
        for error in errors_to_analyze:
            print(f"\n🔍 Đang phân tích lỗi: User thích '{error['target_text']}' nhưng gợi ý sai...")
            
            new_rule = self.analyst.analyze_failure(
                error['user_profile'], 
                error['applied_rules'], 
                error['predicted_ids'], 
                error['target_text']
            )

            if new_rule:
                print(f"   -> Đề xuất Rule: {new_rule.get('condition_description')}")
                
                existing_rules = self.kb.get_all_rules()
                decision = self.analyst.consolidate_rules(new_rule, existing_rules)
                
                action = decision.get('action', 'IGNORE')
                
                if action == 'ADD':
                    self.kb.add_rule(new_rule['condition_description'], new_rule['instruction'])

                elif action == 'REPLACE':
                    t_id = decision.get('target_id')
                    print(f"   ♻️ Thay thế Rule cũ ID #{t_id}...")
                    self.kb.replace_rule(t_id, new_rule['condition_description'], new_rule['instruction'])
                
                elif action == 'IGNORE':
                    print("   🚫 Rule mới bị bỏ qua (Trùng lặp hoặc yếu hơn).")
            else:
                print("   ⚠️ Analyst không tìm ra nguyên nhân cụ thể.")

        self.kb.save_rules()

    def calculate_metrics_from_results(self, results, k=10, candidate_size=20):
        # SỬ DỤNG LOGIC MỚI
        report = self.metrics.evaluate_batch(results, candidate_size=candidate_size)
        
        self.error_list = []
        for res in results:
            target = str(res.get('target_id')) 
            preds = [str(p) for p in res.get('predicted_ids', [])]
            
            # Vẫn dùng k để lọc lỗi
            if target not in preds[:k]:
                if 'user_profile' in res:
                    self.record_error(
                        res['user_profile'], 
                        res.get('applied_rules', []), 
                        preds, 
                        res.get('target_text', 'Unknown')
                    )
        
        return report, self.error_list