import json
from .utils import Utils

class AgentAnalyst:
    def __init__(self, gemini_client):
        self.client = gemini_client

    def analyze_failure(self, user_profile, applied_rules, predicted_ids, target_id_text):
        """
        Phân tích thất bại cụ thể trong ngữ cảnh Recommendation.
        """
        role_prompt = (
            "You are a Senior Data Analyst for a Recommendation System. "
            "Your task is to diagnose why the AI failed to recommend the correct item and generate a strict Rule to fix it.\n\n"
        )
        
        context_prompt = (
            f"### CONTEXT DATA:\n"
            f"1. USER PROFILE (History/Interests): {user_profile}\n"
            f"2. APPLIED RULES (Previously learned): {applied_rules if applied_rules else 'None'}\n"
            f"3. AI PREDICTION (Incorrect): {predicted_ids}\n"
            f"4. GROUND TRUTH (What User actually wanted): {target_id_text}\n\n"
        )
        
        logic_instruction = (
            "### ANALYSIS STEPS:\n"
            "1. COMPARE the 'Ground Truth' item attributes (Genre, Year, Director, Style) against the 'User Profile'.\n"
            "2. IDENTIFY the gap: Why did the AI miss this? (e.g., 'User likes 90s Action, but AI ignored the era', 'AI over-prioritized popularity over genre').\n"
            "3. IF NO RULES applied: Create a new rule capturing this user preference pattern.\n"
            "4. IF RULES applied but failed: The existing rule is too weak or vague. Refine it.\n"
            "5. RULE LOGIC: The rule must be a conditional statement focusing on Item Attributes vs User History.\n\n"
        )
        
        output_format = (
            "### OUTPUT FORMAT (JSON):\n"
            "Return a SINGLE JSON object. The 'instruction' must be actionable for the Recommender.\n"
            "{\n"
            "  \"condition_description\": \"Brief context (e.g., 'User history is dominated by Horror, but Candidates include Sci-Fi')\",\n"
            "  \"instruction\": \"The specific rule (e.g., 'Prioritize items sharing at least 2 genres with the user's last watched item')\"\n"
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
        [Checklist 4.2] Conflict and Redundancy Check.
        """
        if not existing_rules: 
           return {"action": "ADD", "target_id": None}
        prompt = f"""
        You are a Knowledge Base Manager for a Recommender System. 
        Compare the NEW Rule with the EXISTING Rules List to maintain a clean rule set.
        
        NEW RULE: 
        - Condition: {new_rule['condition_description']}
        - Instruction: {new_rule['instruction']}
        
        EXISTING RULES:
        {json.dumps([{'id': i, 'desc': r['condition_description'], 'instr': r['instruction']} for i, r in enumerate(existing_rules)], ensure_ascii=False)}
        
        TASK:
        1. IGNORE: If New Rule is semantically identical or weaker than an existing one.
        2. REPLACE: If New Rule covers the same logic but is clearer, stricter, or better generalized than an existing one.
        3. ADD: If New Rule provides completely new logic (different genre aspect, different user behavior pattern).
        
        RETURN SINGLE JSON: {{"action": "ADD" | "IGNORE" | "REPLACE", "target_id": <int or null>}}
        """
        
        res = self.client.call_api(prompt)
        return Utils.parse_json_rule(res)

    async def analyze_batch_failures_async(self, error_batch):
        """
        Analyze a batch of errors to find a common pattern.
        """
        if not error_batch:
            return []

        cases_text = ""
        for idx, err in enumerate(error_batch):
            cases_text += (
                f"\n--- CASE {idx+1} ---\n"
                f"User History: {str(err['user_profile'])[:200]}...\n"
                f"Prediction: {err['predicted_ids']}\n"
                f"Correct Item: {err['target_text']}\n"
            )

        prompt = f"""
        You are a Lead Analyst for a Recommender System. Below are {len(error_batch)} failed recommendation cases.
        
        FAILED CASES:
        {cases_text}
        
        TASK:
        Analyze these cases to find the MOST COMMON failure pattern (e.g., "System consistently ignores the release year preference" or "System fails to detect niche genres").
        Derive 01 GENERAL RULE that would have prevented most of these errors.
        
        RETURN SINGLE JSON:
        {{
            "condition_description": "Describe the common recurring scenario",
            "instruction": "General directive to fix this pattern"
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
                f"User Profile: {str(err['user_profile'])[:300]}...\n"
                f"Bad Prediction: {err['predicted_ids']}\n"
                f"Correct Item: {err['target_text']}\n"
            )

        prompt = f"""
        You are a Strategic Analyst. Below are {len(sample_errors)} examples where the Recommendation Agent failed.
        
        FAILURE DATA:
        {cases_text}
        
        TASK:
        1. CLUSTER the errors into common causes (e.g., 'Ignored Recency', 'Genre Mismatch', 'Sequel Detection Failed').
        2. GENERALIZE: For each cause, propose 01 GENERAL RULE to fix it fundamentally.
        3. LIMIT: Propose a maximum of 3 key rules.
        
        OUTPUT FORMAT (JSON ARRAY):
        [
            {{
                "condition_description": "Description of the error group",
                "instruction": "The correcting rule"
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

        return self.consolidate_rules(new_rule, existing_rules)