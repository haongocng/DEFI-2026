import pandas as pd
import random
import json
import os

def generate_datasets(review_path, meta_path, n_train=50, n_test=1000, n_negatives=19):
    # 1. Đọc và chuẩn bị dữ liệu
    print(f"Đang đọc dữ liệu từ: {review_path}...")
    if not os.path.exists(review_path) or not os.path.exists(meta_path):
        print(f"❌ Lỗi: Không tìm thấy file đầu vào. Vui lòng kiểm tra lại đường dẫn.")
        return

    df_review = pd.read_json(review_path, lines=True)
    df_meta = pd.read_json(meta_path, lines=True)

    # Gộp dữ liệu Review và Meta
    df = pd.merge(
        df_review[['user_id', 'parent_asin', 'timestamp']], 
        df_meta[['parent_asin', 'title']], 
        on='parent_asin', 
        how='left'
    )

    # Làm sạch dữ liệu
    df = df.dropna(subset=['title'])
    # Lọc User có ít nhất 2 tương tác (để có ít nhất 1 session và 1 target)
    user_counts = df['user_id'].value_counts()
    df = df[df['user_id'].isin(user_counts[user_counts >= 2].index)]
    df = df.sort_values(['user_id', 'timestamp'])

    all_titles = df['title'].unique().tolist()
    all_samples = []

    print("Đang xử lý cấu trúc dữ liệu...")
    # 2. Duyệt qua từng User để tạo mẫu dữ liệu
    for user_id, group in df.groupby('user_id'):
        titles = group['title'].tolist()
        
        target_title = titles[-1] # Món cuối là target
        session_titles = titles[:-1] # Các món trước là history
        
        # Lấy mẫu âm (19 món user chưa từng mua)
        user_bought = set(titles)
        potential_negatives = [t for t in all_titles if t not in user_bought]
        
        if len(potential_negatives) < n_negatives:
            continue
            
        negatives = random.sample(potential_negatives, n_negatives)
        candidates = [target_title] + negatives
        random.shuffle(candidates)
        
        target_index = candidates.index(target_title) + 1
        
        # Format chuỗi với ký tự thoát (escaped quotes) và đánh số
        session_str = ", ".join([f'{i+1}.\\"{t}\\"' for i, t in enumerate(session_titles)])
        candidate_str = ", ".join([f'{i+1}.\\"{t}\\"' for i, t in enumerate(candidates)])
        
        # Tạo chuỗi input chuẩn
        input_text = f"Current session interactions: [{session_str}]\nCandidate Set: [{candidate_str}]"
        
        all_samples.append({
            "target": target_title,
            "target_index": target_index,
            "input": input_text
        })

    # Đảm bảo trộn ngẫu nhiên toàn bộ mẫu trước khi chia file
    random.shuffle(all_samples)

    # Đảm bảo thư mục data tồn tại
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)

    # 3. Tạo file TEST (1000 mẫu)
    if len(all_samples) >= n_test:
        test_data = all_samples[:n_test]
        test_path = os.path.join(output_dir, 'test_dataset.json')
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        print(f"--- ✅ Đã tạo xong file TEST tại: {test_path} ({len(test_data)} mẫu) ---")
    else:
        print(f"⚠️ Cảnh báo: Chỉ có {len(all_samples)} mẫu, không đủ 1000 cho file test.")

    # 4. Tạo file TRAIN (50 mẫu)
    # Lấy 50 mẫu tiếp theo (không trùng với file test)
    if len(all_samples) >= (n_test + n_train):
        train_samples = all_samples[n_test : n_test + n_train]
        # SỬA ĐỔI CHÍNH Ở ĐÂY: Giữ lại toàn bộ thông tin thay vì chỉ giữ "input"
        train_data = train_samples 
        
        train_path = os.path.join(output_dir, 'train_dataset.json')
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        print(f"--- ✅ Đã tạo xong file TRAIN tại: {train_path} ({len(train_data)} mẫu) ---")
    else:
        print("⚠️ Cảnh báo: Không đủ mẫu để tạo file train riêng biệt.")

if __name__ == "__main__":
    base_path = "normal_dataset/amazon"
    rev = os.path.join(os.path.dirname(__file__), base_path, 'Gift_Cards.jsonl')
    meta = os.path.join(os.path.dirname(__file__), base_path, 'meta_Gift_Cards.jsonl')
    
    generate_datasets(rev, meta)