import re
import json
import numpy as np
import os

class Utils:
    @staticmethod
    def extract_item_ids(response_text):

        try:
            # Tìm tất cả các cụm số nằm trong dấu ngoặc vuông [ ]
            match = re.search(r'\[\s*([\d\s,]+)\s*\]', response_text)
            if match:
                ids_str = match.group(1)
                # Chuyển chuỗi "101, 202" thành list [101, 202]
                return [int(id_strip.strip()) for id_strip in ids_str.split(',') if id_strip.strip()]
        except Exception as e:
            print(f"⚠️ Lỗi trích xuất ID: {e}")
        return []
    @staticmethod
    def parse_json_list(response_text):

        try:
            match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            return []
        except Exception as e:
            print(f"⚠️ Lỗi parse JSON List: {e}")
            return []
    @staticmethod
    def parse_json_rule(response_text):

        try:
            json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            return json.loads(response_text)
        except Exception as e:
            print(f"⚠️ Lỗi parse JSON: {e}")
            return None

    @staticmethod
    def calculate_cosine_similarity(vec1, vec2):

        if not vec1 or not vec2:
            return 0.0
        
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        return dot_product / (norm_v1 * norm_v2)

class EmbeddingCache:

    def __init__(self, cache_file='memory/embedding_cache.json'):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_cache(self):
        # Đảm bảo thư mục memory/ tồn tại
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False)

    def get_embedding(self, text, client_genai):

        if text in self.cache:
            return self.cache[text]
        
        # Gọi hàm get_embedding từ GeminiClient (trong request.py)
        embedding = client_genai.get_embedding(text)
        
        if embedding:
            self.cache[text] = embedding
            self.save_cache()
            
        return embedding

