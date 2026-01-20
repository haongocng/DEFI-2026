import pandas as pd
import json

def convert_parquet_to_recommendation_json(parquet_path, output_json, num_samples=50):
    print(f"--- Đang xử lý file: {parquet_path} ---")
    
    # 1. Đọc file Parquet
    df = pd.read_parquet(parquet_path)
    
    # 2. Lấy một số lượng mẫu nhất định để thử nghiệm
    df_sample = df.head(num_samples)
    
    formatted_data = []
    
    for _, row in df_sample.iterrows():
        # Lưu ý: Tên cột (history, candidates, target) có thể thay đổi tùy theo dataset
        # Bạn cần kiểm tra df.columns nếu bị lỗi tên cột
        history = row.get('history', [])
        candidates = row.get('candidates', [])
        target = row.get('target', '')
        
        # Tìm vị trí của target trong candidates (từ 1-20)
        try:
            # Chuyển về list nếu nó đang ở dạng numpy array
            cand_list = list(candidates)
            target_idx = cand_list.index(target) + 1
        except (ValueError, TypeError):
            target_idx = 1 # Mặc định nếu không tìm thấy
            
        # Tạo chuỗi input chuẩn để dán vào Chat
        input_text = f"Current session interactions: {history}\nCandidate Set: {candidates}"
        
        formatted_data.append({
            "input": input_text,
            "target_index": target_idx
        })
    
    # 3. Lưu thành file JSON để dùng cho các bước sau
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=4, ensure_ascii=False)
        
    print(f"=> Đã tạo xong file JSON: {output_json}")

# --- Thực thi ---
# Thay 'full-00000-of-00002.parquet' bằng đường dẫn thực tế của bạn
convert_parquet_to_recommendation_json('full-00000-of-00002.parquet', 'amazon_cellphones_test.json')