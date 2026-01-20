import pandas as pd
import random
import os

def load_and_preprocess(review_path, meta_path):
    print("--- Bước 1: Đang đọc và gộp dữ liệu ---")
    # Đọc file JSONL (JSON Lines)
    df_review = pd.read_json(review_path, lines=True)
    df_meta = pd.read_json(meta_path, lines=True)

    # Gộp dữ liệu Review với Meta để lấy thông tin Title sản phẩm
    # Chúng ta sử dụng 'parent_asin' làm khóa chính
    df = pd.merge(
        df_review[['user_id', 'parent_asin', 'timestamp', 'rating']], 
        df_meta[['parent_asin', 'title']], 
        on='parent_asin', 
        how='left'
    )

    # --- Bước 2: Làm sạch và Lọc dữ liệu ---
    # Loại bỏ các dòng thiếu thông tin quan trọng
    df = df.dropna(subset=['user_id', 'parent_asin'])

    # Lọc User: Chỉ giữ những người có ít nhất 3 tương tác 
    # (Để có: 2 cái làm lịch sử/train, 1 cái làm Ground Truth để test)
    user_counts = df['user_id'].value_counts()
    df = df[df['user_id'].isin(user_counts[user_counts >= 3].index)]

    # Sắp xếp theo User và Thời gian (rất quan trọng trong RecSys)
    df = df.sort_values(by=['user_id', 'timestamp'])
    
    print(f"Số lượng tương tác sau khi lọc: {len(df)}")
    print(f"Số lượng User: {df['user_id'].nunique()}")
    print(f"Số lượng Sản phẩm: {df['parent_asin'].nunique()}")
    
    return df

def split_leave_one_out(df):
    print("\n--- Bước 2: Tách Ground Truth (Leave-one-out) ---")
    # Lấy tương tác cuối cùng (mới nhất) của mỗi user làm Ground Truth
    test_gt = df.groupby('user_id').tail(1).copy()
    
    # Tất cả các tương tác trước đó được coi là lịch sử (History/Train)
    train_history = df.drop(test_gt.index)
    
    return train_history, test_gt

def get_test_candidates(user_id, train_history, test_gt, all_items, n_negatives=99):
    """
    Tạo danh sách gồm 1 sản phẩm thật (GT) và n sản phẩm nhiễu (Negatives)
    """
    # 1. Lấy sản phẩm thật
    real_item = test_gt[test_gt['user_id'] == user_id]['parent_asin'].values[0]
    
    # 2. Lấy các sản phẩm user đã từng tương tác để tránh đưa vào phần nhiễu
    interacted_items = set(train_history[train_history['user_id'] == user_id]['parent_asin'].tolist())
    interacted_items.add(real_item)
    
    # 3. Lấy mẫu âm (Negative Sampling)
    # Lấy những sản phẩm có trong hệ thống nhưng user chưa từng tương tác
    negative_candidates = list(set(all_items) - interacted_items)
    negatives = random.sample(negative_candidates, min(len(negative_candidates), n_negatives))
    
    # 4. Tạo danh sách cuối cùng để đưa vào model rank
    test_list = [real_item] + negatives
    random.shuffle(test_list)
    
    return test_list, real_item

# ==========================================
# CHƯƠNG TRÌNH CHÍNH
# ==========================================
if __name__ == "__main__":
    # 1. Load dữ liệu
    # Thay đổi tên file nếu bạn lưu tên khác
    df_cleaned = load_and_preprocess('Gift_Cards.jsonl', 'meta_Gift_Cards.jsonl')

    # 2. Chia tập dữ liệu
    train_hist, test_gt = split_leave_one_out(df_cleaned)
    all_item_ids = df_cleaned['parent_asin'].unique().tolist()

    # 3. Demo tạo dữ liệu test cho 3 User đầu tiên
    print("\n--- Bước 3: Demo tạo Input cho Model Ranking ---")
    sample_users = test_gt['user_id'].iloc[:3]

    for u_id in sample_users:
        input_list, ground_truth = get_test_candidates(u_id, train_hist, test_gt, all_item_ids)
        
        print(f"\n[User ID]: {u_id}")
        print(f"Lịch sử đã mua: {len(train_hist[train_hist['user_id'] == u_id])} sản phẩm")
        print(f"Đáp án đúng (GT): {ground_truth}")
        print(f"Số lượng sản phẩm đưa vào Model để Rank: {len(input_list)}")
        print(f"Gợi ý 5 mã đầu trong danh sách Input: {input_list[:5]}")
