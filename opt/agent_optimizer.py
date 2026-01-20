import json
from .utils import Utils

class AgentOptimizer:
    def __init__(self, gemini_client):
        self.client = gemini_client

    def build_prompt(self, context, retrieved_rules, candidate_set):
        # 1. Detect Context: Crypto vs Recommendation
        # Giữ lại check này để code tương thích ngược, nhưng tối ưu phần else (Recommendation)
        is_crypto = "Symbol:" in str(context) and "RSI:" in str(context)

        if is_crypto:
            # --- PROMPT FOR CRYPTO TRADING (Simplified for brevity) ---
            role_prompt = (
                "You are a professional Crypto Trader. Analyze the market and decide: BUY or WAIT.\n"
                "Strictly follow the Trading Rules below.\n\n"
            )
            data_prompt = f"### MARKET CONTEXT:\n{context}\n\n### ACTIONS:\n{candidate_set}\n\n"
            rules_section = "### TRADING RULES:\n" + ("\n".join([f"- {r}" for r in retrieved_rules]) if retrieved_rules else "- Follow Trend.")
            output_example = "\nExample: {\"recommended_ids\": [1], \"reason\": \"Trend is UP.\"}"

        else:
            # --- PROMPT FOR ITEM RECOMMENDATION (OPTIMIZED) ---
            role_prompt = (
                "You are an expert Recommendation System. Your goal is to select the BEST item "
                "from the Candidate Set that matches the User's Profile and History.\n"
                "You must strictly follow the Learned Rules to avoid previous mistakes.\n\n"
            )
            
            data_prompt = (
                f"### 1. USER PROFILE & HISTORY:\n{context}\n\n"
                f"### 2. CANDIDATE SET (Items to choose from):\n{candidate_set}\n\n"
            )
            
            rules_section = "### 3. LEARNED RULES (KNOWLEDGE BASE):\n"
            if retrieved_rules:
                rules_section += "WARNING: You MUST apply these rules to filter or rank candidates:\n"
                for idx, r in enumerate(retrieved_rules):
                    # Handle if r is dict or string
                    rule_text = r['instruction'] if isinstance(r, dict) and 'instruction' in r else str(r)
                    rules_section += f"   - Rule {idx+1}: {rule_text}\n"
            else:
                rules_section += "   - No specific rules yet. Prioritize Genre match and High Rating.\n"
            
            steps_prompt = (
                "\n### 4. REASONING STEPS:\n"
                "1. Analyze the User's core interests (Genres, Themes, Decades).\n"
                "2. Filter Candidates: Which items match the core interests?\n"
                "3. APPLY RULES: Check if any candidate violates or satisfies the Learned Rules.\n"
                "4. Rank the Top 10 best items in descending order of relevance.\n" # <-- Sửa ở đây
            )

            output_example = (
                "\n### 5. OUTPUT FORMAT:\n"
                "Return a SINGLE valid JSON object:\n"
                "{\n"
                "  \"recommended_ids\": [ id1, id2, id3, ..., id10 ],\n" # <-- Yêu cầu 1 list 10 cái
                "  \"reason\": \"Explain your top choice and how rules were applied.\"\n"
                "}\n"
            )

        full_prompt = role_prompt + data_prompt + rules_section + steps_prompt + output_example
        return full_prompt


    async def get_recommendations_async(self, user_profile, retrieved_rules, candidate_set):
        prompt = self.build_prompt(user_profile, retrieved_rules, candidate_set)
        
        # Gọi API
        response_text = await self.client.call_api_async(prompt)
        
        # Fallback logging or debugging could go here
        # print(f"DEBUG LLM Response: {response_text}")

        if response_text:
            data = Utils.parse_json_rule(response_text)
            if data and 'recommended_ids' in data:
                return data['recommended_ids'], data.get('reason', '')
        
        return [], f"Error parsing response. Raw: {response_text[:50]}..."