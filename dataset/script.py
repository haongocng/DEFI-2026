import math
import ast
import re

def calculate_ndcg(rank, k):
    """Tính NDCG@K: 1/log2(rank+1) nếu rank <= k, ngược lại là 0."""
    if 0 < rank <= k:
        return 1 / math.log2(rank + 1)
    return 0

def calculate_ap(rank, k):
    """Tính Average Precision@K: 1/rank nếu rank <= k, ngược lại là 0."""
    if 0 < rank <= k:
        return 1 / rank
    return 0

def process_metrics(gt_file, result_file, k_list=[1, 5, 10]):
    print(f"\n{'='*50}")
    print(f"ĐANG XỬ LÝ: {result_file}")
    print(f"{'='*50}")
    
    # 1. Đọc Ground Truth
    try:
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_lines = f.readlines()[1:] 
            targets = [int(line.split(',')[1].strip()) for line in gt_lines]
    except FileNotFoundError:
        print(f"  [Lỗi] Không tìm thấy file Ground Truth: {gt_file}")
        return

    # 2. Đọc và lọc kết quả từ LLM
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            raw_lines = f.readlines()
    except FileNotFoundError:
        print(f"  [Lỗi] Không tìm thấy file kết quả: {result_file}")
        return

    clean_rankings = []
    for line in raw_lines:
        line = line.strip()
        if not line: continue
        match = re.search(r'\[.*?\]', line)
        if match:
            try:
                ranking = ast.literal_eval(match.group(0))
                if isinstance(ranking, list) and len(ranking) >= 20:
                    clean_rankings.append(ranking)
            except:
                continue

    # 3. Tính toán cho từng mốc K
    valid_samples = min(len(clean_rankings), len(targets))
    if valid_samples == 0:
        print("  [Cảnh báo] Không có dữ liệu hợp lệ để tính toán.")
        return

    # In tiêu đề bảng
    print(f"{'Metric':<10} | {'@1':<10} | {'@5':<10} | {'@10':<10}")
    print("-" * 50)

    results = {"HIT": [], "NDCG": [], "MAP": []}

    for k in k_list:
        total_hit = 0
        total_ndcg = 0
        total_map = 0
        
        for i in range(valid_samples):
            target = targets[i]
            ranking = clean_rankings[i]
            
            if target in ranking:
                rank = ranking.index(target) + 1
                
                if rank <= k:
                    total_hit += 1
                    total_ndcg += calculate_ndcg(rank, k)
                    total_map += calculate_ap(rank, k)
        
        results["HIT"].append(f"{total_hit / valid_samples:.4f}")
        results["NDCG"].append(f"{total_ndcg / valid_samples:.4f}")
        results["MAP"].append(f"{total_map / valid_samples:.4f}")

    # In kết quả theo hàng
    for metric in ["HIT", "NDCG", "MAP"]:
        vals = results[metric]
        print(f"{metric:<10} | {vals[0]:<10} | {vals[1]:<10} | {vals[2]:<10}")
    
    print(f"\nTổng số mẫu khớp: {valid_samples}/{len(targets)}")

# --- Cấu hình các file ---
evaluation_tasks = [
    {'name': 'Games', 'gt': 'games_ground_truth.csv', 'res': 'games_output.txt'},
    {'name': 'Bundle', 'gt': 'bundle_ground_truth.csv', 'res': 'bundle_output.txt'},
    {'name': 'ML-1M', 'gt': 'ml1m_ground_truth.csv', 'res': 'ml1m_output.txt'}
]


for task in evaluation_tasks:
    process_metrics(task['gt'], task['res'])