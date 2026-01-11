import json
from .utils import Utils

class AgentAnalyst:
    def __init__(self, gemini_client):

        self.client = gemini_client

    def analyze_failure(self, user_profile, applied_rules, predicted_ids, target_id_text):
        role_prompt = (
            "Bạn là một chuyên gia phân tích dữ liệu hệ thống gợi ý. "
            "Nhiệm vụ của bạn là tìm ra lý do tại sao hệ thống gợi ý sai và tạo ra quy tắc (Rule) mới để sửa lỗi.\n\n"
        )
        
        context_prompt = (
            f"### NGỮ CẢNH:\n"
            f"- Hồ sơ User: {user_profile}\n"
            f"- Các quy tắc đã áp dụng: {applied_rules if applied_rules else 'Không có'}\n"
            f"- Gợi ý của AI: {predicted_ids}\n"
            f"- Kết quả đúng thực tế: {target_id_text}\n\n"
        )
        
        logic_instruction = (
            "### CHỈ DẪN PHÂN TÍCH:\n"
            "1. Nếu không có quy tắc nào được áp dụng: Hãy tạo một quy tắc mới dựa trên điểm chung giữa hồ sơ user và kết quả đúng.\n"
            "2. Nếu đã có quy tắc nhưng vẫn sai: Hãy sửa lại quy tắc đó để cụ thể và chính xác hơn.\n\n"
        )
        
        output_format = (
            "### YÊU CẦU ĐẦU RA (JSON):\n"
            "Trả về JSON theo cấu trúc sau:\n"
            "{\n"
            "  \"condition_description\": \"Mô tả ngắn gọn ngữ cảnh của user (VD: Người dùng thích phim hành động sci-fi thập niên 80)\",\n"
            "  \"instruction\": \"Chỉ dẫn cụ thể (VD: Ưu tiên gợi ý các phim có yếu tố du hành thời gian hoặc robot)\"\n"
            "}\n"
        )

        full_prompt = role_prompt + context_prompt + logic_instruction + output_format
        
        response_text = self.client.call_api(full_prompt)
        
        if response_text:
            new_rule = Utils.parse_json_rule(response_text)
            return new_rule
        
        return None
    def consolidate_rules(self, new_rule, existing_rules):
        """
        [Checklist 4.2] Kiểm tra mâu thuẫn và bao trùm.
        Trả về: 'ADD', 'REPLACE', 'MERGE', hoặc 'IGNORE'.
        """
        if not existing_rules: return "ADD", None

        prompt = f"""
        Bạn là chuyên gia quản lý tri thức. Hãy so sánh Quy tắc Mới với Danh sách Quy tắc Cũ.
        
        QUY TẮC MỚI: 
        - Ngữ cảnh: {new_rule['condition_description']}
        - Chỉ dẫn: {new_rule['instruction']}
        
        DANH SÁCH CŨ:
        {json.dumps([{'id': i, 'desc': r['condition_description']} for i, r in enumerate(existing_rules)], ensure_ascii=False)}
        
        NHIỆM VỤ:
        1. Nếu Quy tắc Mới trùng lặp hoặc mâu thuẫn nhưng yếu hơn quy tắc cũ: Trả về {{"action": "IGNORE"}}.
        2. Nếu Quy tắc Mới bao trùm hoặc tốt hơn quy tắc cũ: Trả về {{"action": "REPLACE", "target_id": index_của_quy_tắc_cũ}}.
        3. Nếu Quy tắc Mới hoàn toàn khác: Trả về {{"action": "ADD"}}.
        
        Trả về JSON duy nhất.
        """
        
        res = self.client.call_api(prompt)
        return Utils.parse_json_rule(res)
    async def analyze_batch_failures_async(self, error_batch):
        """
        [MỚI] Phân tích danh sách nhiều lỗi cùng lúc.
        input: error_batch = list các dict lỗi
        """
        if not error_batch:
            return []

        cases_text = ""
        for idx, err in enumerate(error_batch):
            cases_text += (
                f"\n--- TRƯỜNG HỢP {idx+1} ---\n"
                f"User Profile: {err['user_profile']}\n"
                f"AI Gợi ý sai: {err['predicted_ids']}\n"
                f"Đáp án đúng: {err['target_text']}\n"
            )

        prompt = f"""
        Bạn là chuyên gia phân tích lỗi hệ thống RecSys. Dưới đây là {len(error_batch)} trường hợp hệ thống gợi ý sai.
        
        {cases_text}
        
        NHIỆM VỤ:
        Hãy phân tích tổng quát các trường hợp trên và rút ra 01 QUY TẮC (RULE) quan trọng nhất có thể khắc phục được nhiều lỗi nhất.
        
        TRẢ VỀ JSON DUY NHẤT:
        {{
            "condition_description": "Mô tả ngữ cảnh chung (VD: User thích phim hành động nhưng hệ thống gợi ý phim tình cảm)",
            "instruction": "Chỉ dẫn cụ thể để sửa lỗi"
        }}
        """

        response_text = await self.client.call_api_async(prompt)
        if response_text:
             return Utils.parse_json_rule(response_text)
        return None
    async def analyze_global_failures_async(self, error_list, sample_size=10):

        if not error_list: return []

        sample_errors = error_list[:sample_size]

        cases_text = ""
        for idx, err in enumerate(sample_errors):
            cases_text += (
                f"\n--- Case #{idx+1} ---\n"
                f"User History: {err['user_profile'][:300]}...\n" # Cắt ngắn
                f"AI Gợi ý (SAI): {err['predicted_ids']}\n"
                f"Đáp án (ĐÚNG): {err['target_text']}\n"
            )

        prompt = f"""
        Bạn là Chuyên gia Phân tích Hệ thống RecSys. Dưới đây là {len(sample_errors)} trường hợp hệ thống gợi ý thất bại.
        
        DỮ LIỆU LỖI:
        {cases_text}
        
        NHIỆM VỤ:
        1. "Clustering": Hãy gom nhóm các lỗi trên thành các nguyên nhân phổ biến (VD: Sai về thể loại, Sai về độ tuổi, Bỏ qua phần tiếp theo của series...).
        2. "Generalization": Với mỗi nhóm nguyên nhân, hãy đề xuất 01 QUY TẮC (RULE) tổng quát nhất để khắc phục triệt để.
        3. Giới hạn: Đề xuất tối đa 3 Rules quan trọng nhất.
        
        YÊU CẦU OUTPUT (JSON ARRAY):
        Trả về một danh sách JSON (List of Objects) theo định dạng:
        [
            {{
                "condition_description": "Mô tả nhóm nguyên nhân (VD: Khi user đang xem chuỗi phim nhiều phần)",
                "instruction": "Chỉ dẫn tổng quát (VD: Ưu tiên gợi ý phần tiếp theo theo thứ tự phát hành)"
            }},
            ...
        ]
        """
        
        response_text = await self.client.call_api_async(prompt)
        
        if response_text:
            rules = Utils.parse_json_list(response_text)
            return rules
        return []
    
    async def consolidate_rules_async(self, new_rule, existing_rules):

        if not existing_rules: return {"action": "ADD"}

        prompt = f"""
        Bạn là chuyên gia quản lý tri thức RecSys. Hãy so sánh Quy tắc Mới với Danh sách Quy tắc Cũ.
        
        QUY TẮC MỚI: 
        - Ngữ cảnh: {new_rule['condition_description']}
        - Chỉ dẫn: {new_rule['instruction']}
        
        DANH SÁCH CŨ:
        {json.dumps([{'id': i, 'desc': r['condition_description']} for i, r in enumerate(existing_rules)], ensure_ascii=False)}
        
        NHIỆM VỤ:
        1. Nếu Quy tắc Mới trùng lặp hoặc yếu hơn quy tắc cũ: Trả về {{"action": "IGNORE"}}.
        2. Nếu Quy tắc Mới bao trùm/tốt hơn quy tắc cũ: Trả về {{"action": "REPLACE", "target_id": index_của_quy_tắc_cũ}}.
        3. Nếu Quy tắc Mới khác biệt và hữu ích: Trả về {{"action": "ADD"}}.
        
        Trả về JSON duy nhất.
        """
        
        res = await self.client.call_api_async(prompt)
        return Utils.parse_json_rule(res)
    