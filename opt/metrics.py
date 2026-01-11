import numpy as np
import csv
from datetime import datetime
import os

class Metrics:
    def __init__(self):
        pass

    @staticmethod
    def get_hit_rate(target_id, recommended_ids, k=10):

        top_k = recommended_ids[:k]
        return 1.0 if target_id in top_k else 0.0

    @staticmethod
    def get_ndcg(target_id, recommended_ids, k=10):

        top_k = recommended_ids[:k]
        if target_id in top_k:
            index = top_k.index(target_id)
            # Công thức: 1 / log2(vị trí + 2). Vị trí bắt đầu từ 0 -> +2 để log cơ số 2 của (rank+1) không bị lỗi
            return 1.0 / np.log2(index + 2)
        return 0.0

    @staticmethod
    def get_mrr(target_id, recommended_ids, k=10):

        top_k = recommended_ids[:k]
        if target_id in top_k:
            index = top_k.index(target_id)
            return 1.0 / (index + 1)
        return 0.0

    def evaluate_batch(self, results, k=10):

        hr_list = []
        ndcg_list = []
        mrr_list = []

        for item in results:
            if isinstance(item, dict):
                target = item.get('target_id')
                preds = item.get('predicted_ids', [])
            else:
                target, preds = item

            if target is not None:
                hr_list.append(self.get_hit_rate(target, preds, k))
                ndcg_list.append(self.get_ndcg(target, preds, k))
                mrr_list.append(self.get_mrr(target, preds, k))

        avg_hr = np.mean(hr_list) if hr_list else 0.0
        avg_ndcg = np.mean(ndcg_list) if ndcg_list else 0.0
        avg_mrr = np.mean(mrr_list) if mrr_list else 0.0

        return {
            f"Hit@{k}": avg_hr,
            f"NDCG@{k}": avg_ndcg,
            f"MRR@{k}": avg_mrr,
            
            "Hit_Rate": avg_hr,
            "NDCG": avg_ndcg,
            "MRR": avg_mrr
        }

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