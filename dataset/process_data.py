import pandas as pd
import random
import json

def generate_test_data(review_path, meta_path, output_file, n_negatives=19):
    # 1. Đọc dữ liệu
    print("Đang đọc dữ liệu...")
    df_review = pd.read_json(review_path, lines=True)
    df_meta = pd.read_json(meta_path, lines=True)

    # Gộp để lấy Title (Tên sản phẩm)
    df = pd.merge(
        df_review[['user_id', 'parent_asin', 'timestamp']], 
        df_meta[['parent_asin', 'title']], 
        on='parent_asin', 
        how='left'
    )

    # Làm sạch: Bỏ các dòng thiếu Title và lọc User có ít nhất 2 tương tác
    df = df.dropna(subset=['title'])
    user_counts = df['user_id'].value_counts()
    df = df[df['user_id'].isin(user_counts[user_counts >= 2].index)]
    df = df.sort_values(['user_id', 'timestamp'])

    all_titles = df['title'].unique().tolist()
    test_cases = []

    print("Đang tạo format JSON...")
    # 2. Xử lý từng User
    for user_id, group in df.groupby('user_id'):
        interactions = group['title'].tolist()
        
        # Target là món cuối cùng
        target_title = interactions[-1]
        # Session là các món trước đó
        session_titles = interactions[:-1]
        
        # 3. Tạo Candidate Set (1 món thật + n món giả)
        user_bought = set(interactions)
        potential_negatives = [t for t in all_titles if t not in user_bought]
        
        if len(potential_negatives) < n_negatives:
            continue
            
        negatives = random.sample(potential_negatives, n_negatives)
        candidates = [target_title] + negatives
        random.shuffle(candidates)
        
        target_index = candidates.index(target_title) + 1
        
        session_str = ", ".join([f'{i+1}."{title}"' for i, title in enumerate(session_titles)])
        candidate_str = ", ".join([f'{i+1}."{title}"' for i, title in enumerate(candidates)])
        
        input_string = f"Current session interactions: [{session_str}]\nCandidate Set: [{candidate_str}]"
        
        test_case = {
            "target": target_title,
            "target_index": target_index,
            "input": input_string
        }
        test_cases.append(test_case)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_cases[:100], f, ensure_ascii=False, indent=2) # Lưu 100 mẫu để test
    
    print(f"Đã tạo xong file: {output_file}")

generate_test_data('Gift_Cards.jsonl', 'meta_Gift_Cards.jsonl', 'test_data_format.json')