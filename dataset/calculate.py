import math
import ast

def calculate_ndcg(rank):
    """Tính NDCG dựa trên vị trí (rank) của sản phẩm đúng."""
    if rank > 0:
        return 1 / math.log2(rank + 1)
    return 0

def process_metrics(gt_file, result_file, k=10):
    print(f"\n--- Đang xử lý: {result_file} ---")
    
    # 1. Đọc Ground Truth (Bỏ dòng tiêu đề)
    with open(gt_file, 'r') as f:
        gt_lines = f.readlines()[1:] 
        targets = [int(line.split(',')[1].strip()) for line in gt_lines]

    # 2. Đọc kết quả từ LLM
    with open(result_file, 'r') as f:
        results = f.readlines()

    total_hit = 0
    total_ndcg = 0
    count = 0

    for i, res_str in enumerate(results):
        if not res_str.strip(): continue
        
        try:
            # Chuyển chuỗi "[1, 2, ...]" thành list Python
            ranking = ast.literal_eval(res_str.strip())
            target = targets[i]
            
            # Tìm vị trí của target trong bảng xếp hạng (bắt đầu từ 1)
            if target in ranking:
                rank = ranking.index(target) + 1
                
                # Tính HIT@K
                if rank <= k:
                    total_hit += 1
                
                # Tính NDCG
                total_ndcg += calculate_ndcg(rank)
            
            count += 1
        except Exception as e:
            print(f"Lỗi ở dòng {i+1}: {e}")

    # 3. Tính trung bình
    if count == 0: return
    
    avg_hit = total_hit / count
    avg_ndcg = total_ndcg / count

    print(f"Số lượng mẫu: {count}")
    print(f"HIT@{k}: {avg_hit:.4f}")
    print(f"NDCG:    {avg_ndcg:.4f}")

# Cấu hình các file tương ứng
evaluation_tasks = [
    {
        'name': 'Games',
        'gt': 'games_ground_truth.csv',
        'res': 'games_results.txt'
    },
    {
        'name': 'Bundle',
        'gt': 'bundle_ground_truth.csv',
        'res': 'bundle_results.txt'
    },
    {
        'name': 'ML-1M',
        'gt': 'ml1m_ground_truth.csv',
        'res': 'ml1m_results.txt'
    }
]

for task in evaluation_tasks:
    try:
        process_metrics(task['gt'], task['res'])
    except FileNotFoundError as e:
        print(f"Bỏ qua {task['name']}: Thiếu file {e.filename}")