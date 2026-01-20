import json
import random
import numpy as np
import os
from tqdm import tqdm
from metrics import Metric  # Class Metric bạn đã cập nhật
from request import TimelyClient # [SỬA] Import Client để gọi API

class ReferenceBenchmark:
    def __init__(self, data_path, model_name="gpt-4o-mini"):
        self.data_path = data_path
        self.model_name = model_name
        
        # [SỬA] Khởi tạo Client gọi API thay vì Utils
        try:
            self.client = TimelyClient(model_name=model_name)
        except Exception as e:
            print(f"⚠️ Không thể khởi tạo TimelyClient: {e}")
            self.client = None

        # Giả lập "Best Prompt" từ Stage 2 của họ
        self.best_prompt_template = """
Bạn là một trợ lý gợi ý sản phẩm thông minh.
Dựa trên lịch sử tương tác của người dùng và danh sách các ứng viên, hãy sắp xếp và gợi ý Top 10 sản phẩm phù hợp nhất.

LỊCH SỬ NGƯỜI DÙNG:
{history}

DANH SÁCH ỨNG VIÊN (Candidate Set):
{candidates}

YÊU CẦU:
- Chỉ trả về danh sách ID của sản phẩm, ngăn cách bởi dấu phẩy.
- Không giải thích gì thêm.
- Ví dụ output: 101, 205, 300
"""

    def load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def stage1_rerank_and_inject(self, item, candidate_size=20):
        """
        Mô phỏng Giai đoạn 1: Reranking + Target Injection (Đã sửa lỗi crash)
        """
        full_candidates = item.get('candidate_set', []) 
        target_id = item.get('target_id')
        
        if full_candidates is None:
            full_candidates = []

        # --- MÔ PHỎNG SCORING ---
        if len(full_candidates) > candidate_size:
            top_candidates = full_candidates[:candidate_size]
        else:
            top_candidates = list(full_candidates) 

        # --- TARGET INJECTION ---
        if target_id is not None and target_id not in top_candidates:
            if len(top_candidates) >= candidate_size and len(top_candidates) > 0:
                remove_idx = random.randint(0, len(top_candidates) - 1)
                top_candidates.pop(remove_idx)
            
            top_candidates.append(target_id)
            
        return top_candidates

    def stage3_testing(self, test_data, top_k=10):
        results = []
        
        print(f"--- Bắt đầu Benchmark theo Reference Flow ({len(test_data)} mẫu) ---")
        
        for item in tqdm(test_data):
            target_id = item.get('target_id')
            history_text = item.get('input', '') 
            
            processed_candidates = self.stage1_rerank_and_inject(item, candidate_size=20)
            
            candidate_text = ", ".join([str(c) for c in processed_candidates])
            
            prompt = self.best_prompt_template.format(
                history=history_text,
                candidates=candidate_text
            )
            
            if self.client:
                llm_output = self.client.call_api(prompt)
            else:
                llm_output = "" # Trả về rỗng nếu client lỗi
            
            # BƯỚC 3: Trích xuất kết quả
            predicted_ids = []
            if llm_output:
                try:
                    # Lấy các cụm số (giả sử output là "101, 102")
                    parts = llm_output.replace('[', '').replace(']', '').split(',')
                    predicted_ids = [p.strip() for p in parts if p.strip()]
                    
                    # Ép kiểu nếu target là int
                    if isinstance(target_id, int):
                        predicted_ids = [int(p) for p in predicted_ids if p.isdigit()]
                except Exception as e:
                    pass

            results.append({
                'target_id': target_id,
                'predicted_ids': predicted_ids
            })
            
        return results

    def calculate_metrics(self, results, candidate_size=20):
        rank_list = []
        
        for item in results:
            target = item['target_id']
            preds = item['predicted_ids']
            
            if target in preds:
                rank = preds.index(target) + 1
            else:
                rank = candidate_size + 1
            
            rank_list.append(rank)
            
        conf = {'candidate_size': candidate_size}
        metric_tool = Metric(rank_list, conf)
        final_df = metric_tool.run()
        
        return final_df

# --- HÀM MAIN ĐỂ CHẠY ---
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_root = os.path.dirname(current_dir)              
    data_file = os.path.join(project_root, "data", "train_50.json") 
    
    print(f"Đang tìm dữ liệu tại: {data_file}")

    if os.path.exists(data_file):
        bencher = ReferenceBenchmark(data_file)
        data = bencher.load_data()
        
        # Chạy thử trên 5 mẫu đầu tiên
        results = bencher.stage3_testing(data[:5])
        
        # Tính metrics
        df_metrics = bencher.calculate_metrics(results)
        
        print("\n=== KẾT QUẢ BENCHMARK (REFERENCE STYLE) ===")
        print(df_metrics)
    else:
        print(f"❌ Vẫn không tìm thấy file! Hãy kiểm tra lại đường dẫn.")