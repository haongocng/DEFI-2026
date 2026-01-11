import json
import os
from datetime import datetime

class KnowledgeBase:
    def __init__(self, storage_path='memory/rules.json'):

        self.storage_path = storage_path
        self.rules = self._load_rules()

    def _load_rules(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Lỗi đọc file rules: {e}")
                return []
        return []

    def save_rules(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.rules, f, ensure_ascii=False, indent=4)
            print(f"✅ Đã lưu {len(self.rules)} rules vào bộ nhớ.")
        except Exception as e:
            print(f"❌ Lỗi lưu file rules: {e}")

    def add_rule(self, condition, instruction):
        new_rule = {
            "condition_description": condition,
            "instruction": instruction,
            "created_at": datetime.now().isoformat(),
            "performance_score": 0.5
        }
        self.rules.append(new_rule)
        self.save_rules()
        print(f"➕ Đã THÊM rule mới: {condition[:30]}...")
        return new_rule

    def replace_rule(self, index, condition, instruction):

        if 0 <= index < len(self.rules):
            old_rule = self.rules[index]
            self.rules[index] = {
                "condition_description": condition,
                "instruction": instruction,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(), # Tracking
                "performance_score": old_rule.get("performance_score", 0.5) # Giữ lại score cũ hoặc reset tùy logic
            }
            self.save_rules()
            print(f"🔄 Đã THAY THẾ rule số {index}.")
            return True
        else:
            print(f"⚠️ Index {index} không hợp lệ. Không thể thay thế.")
            return False

    def get_all_rules(self):
        return self.rules