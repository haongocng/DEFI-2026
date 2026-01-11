import json
from .utils import Utils

class AgentOptimizer:
    def __init__(self, gemini_client):
        self.client = gemini_client

    def build_prompt(self, user_profile, retrieved_rules, candidate_set):
        role_prompt = (
            "Bạn là một hệ thống gợi ý thông minh. Nhiệm vụ của bạn là chọn ra item phù hợp nhất "
            "từ danh sách ứng viên (Candidate set) dựa trên lịch sử của người dùng.\n\n"
        )
        
        data_prompt = (
            f"### LỊCH SỬ NGƯỜI DÙNG:\n{user_profile}\n\n"
            f"### DANH SÁCH ỨNG VIÊN (CANDIDATE SET):\n{candidate_set}\n\n"
        )
        
        rules_section = "### CÁC QUY TẮC PHẢI TUÂN THỦ:\n"
        if retrieved_rules:
            for r in retrieved_rules:
                rules_section += f"- {r}\n"
        else:
            rules_section += "- Ưu tiên item có cùng thể loại hoặc series với lịch sử người dùng.\n"
        
        output_format = (
            "\n### YÊU CẦU ĐẦU RA:\n"
            "Trả về JSON duy nhất với định dạng:\n"
            "{\"recommended_ids\": [số_thứ_tự_trong_candidate_set], \"reason\": \"giải thích\"}\n"
            "Ví dụ: Nếu chọn item '4.\"StarWars:Battlefront\"', trả về {\"recommended_ids\": [4], ...}"
        )
    
        return role_prompt + data_prompt + rules_section + output_format


    async def get_recommendations_async(self, user_profile, retrieved_rules, candidate_set):
        prompt = self.build_prompt(user_profile, retrieved_rules, candidate_set)
        
        response_text = await self.client.call_api_async(prompt)
        
        print(f"\n🤖 LLM Raw Response:\n{response_text}\n{'-'*20}")

        if response_text:
            data = Utils.parse_json_rule(response_text)
            if data and 'recommended_ids' in data:
                return data['recommended_ids'], data.get('reason', '')
        
        return [], f"Parse Error. Raw text: {response_text[:100]}..."
