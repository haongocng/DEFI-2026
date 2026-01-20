import json
import re
import os
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, data_path):

        self.data_path = data_path
        self.raw_data = []
        
    def load_data(self):
        """Đọc file JSON."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Không tìm thấy file tại {self.data_path}")
            
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        print(f"✅ Đã load {len(self.raw_data)} dòng dữ liệu từ {self.data_path}")
        return self.raw_data

    def clean_item_text(self, item_str):

        match = re.search(r'\d+\.\"(.*?)\"', item_str)
        if match:
            return match.group(1)
        return item_str 
    def parse_user_history(self, input_string):
 
        history_match = re.search(r'Current session interactions: \[(.*?)\]', input_string)
        
        if history_match:
            raw_items = history_match.group(1).split(',')
            clean_items = [self.clean_item_text(item.strip()) for item in raw_items]
            return clean_items
        return []

    def format_history_for_llm(self, clean_items):

        if not clean_items:
            return "Người dùng chưa có lịch sử tương tác."
        
        history_str = ", ".join(clean_items)
        return f"Người dùng đã xem các nội dung sau theo thứ tự thời gian: {history_str}."

    def split_train_test(self, test_size=0.2, random_state=42):

        if not self.raw_data:
            self.load_data()
            
        train_data, test_data = train_test_split(
            self.raw_data, test_size=test_size, random_state=random_state
        )
        print(f"✅ Đã chia dữ liệu: Train ({len(train_data)}), Test ({len(test_data)})")
        return train_data, test_data

