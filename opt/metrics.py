import numpy as np
import pandas as pd
import csv
from datetime import datetime
import os

# --- CORE METRIC CLASS ---
class Metric():
    def __init__(self, rank_list, conf) -> None:
        self.rank_list = rank_list
        self.conf = conf
     
    def ndcg(self, N):
        res = []
        for rank in self.rank_list:
            if rank > N:
                res.append(0)
            else:
                res.append((1 / np.log2(rank + 1)))
        return np.mean(res)
     
    def hit(self, N):
        res = []
        for rank in self.rank_list:
            if rank > N:
                res.append(0)
            else:
                res.append(1)
        return np.mean(res)
     
    def map(self, N):
        res = []
        for rank in self.rank_list:
            if rank > N:
                res.append(0)
            else:
                res.append((1 / rank))
        return np.mean(res)
     
    def run(self):
        res = pd.DataFrame({'KPI@K': ['NDCG', 'HIT', 'MAP']})
        
        size = self.conf.get('candidate_size', 20)
        if size == 10:
            topk_list = [1, 5, 10]
        elif size == 20:
            topk_list = [1, 5, 10, 20]
        else:
            topk_list = [1, 5, 10]
 
        for topk in topk_list:
            metric_res = []
            metric_res.append(self.ndcg(topk))
            metric_res.append(self.hit(topk))
            metric_res.append(self.map(topk))
            res[topk] = np.array(metric_res)
         
        count = sum(1 for r in self.rank_list if r <= size)
        res['#valid_data'] = np.array([count, 0, 0])
         
        return res

class Metrics:
    def __init__(self):
        pass

    def evaluate_batch(self, results, candidate_size=20):
        rank_list = []
        
        print(f"\n🔍 [METRICS DEBUG] Kiểm tra 3 mẫu đầu tiên:")
        
        for i, item in enumerate(results):
            # 1. Trích xuất dữ liệu
            if isinstance(item, dict):
                raw_target = item.get('target_id')
                raw_preds = item.get('predicted_ids', [])
            else:
                raw_target, raw_preds = item
            
            try:
                target = int(raw_target)
                if isinstance(raw_preds, list):
                    preds = [int(p) for p in raw_preds]
                else:
                    preds = [int(raw_preds)]
            except (ValueError, TypeError):
                target = -1
                preds = []

            if target in preds:
                rank = preds.index(target) + 1
            else:
                rank = candidate_size + 1
            
            rank_list.append(rank)

            if i < 50:
                status = "✅ HIT" if rank <= candidate_size else "❌ MISS"
                print(f"   Sample {i}: Target={target} (type {type(target).__name__}) | "
                      f"Preds={preds} (type list) | Match? {target in preds} -> {status}")

        print(f"   ... Tổng {len(rank_list)} mẫu đã xử lý.\n")

        # 5. Chạy tính toán
        conf = {'candidate_size': candidate_size}
        metric_tool = Metric(rank_list, conf)
        df_result = metric_tool.run()

        report = {}
        k_columns = [col for col in df_result.columns if isinstance(col, int)]
        
        for k in k_columns:
            values = df_result[k]
            report[f"NDCG@{k}"] = float(values[0])
            report[f"Hit@{k}"] = float(values[1])
            report[f"MAP@{k}"] = float(values[2])
            
        report["valid_data_count"] = int(df_result['#valid_data'][0])
        
        return report

    @staticmethod
    def log_to_csv(report, file_path='logs/performance.csv'):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_exists = os.path.isfile(file_path)
        row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        row.update(report)
        try:
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            print(f"⚠️ Không thể ghi log CSV: {e}")