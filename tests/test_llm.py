import ssl
import json
import unittest
from unittest.mock import patch

from drap.llm import TimelyLLMClient


class TimelyClientTests(unittest.TestCase):
    def test_legacy_key_alias_and_local_model_are_supported(self) -> None:
        local = {
            "TIMELYGPT_API_KEY": "fixture-key",
            "TIMELY_BASE_URL": "https://example.invalid/api",
            "MODEL_NAME": "fixture-model",
        }
        with patch("drap.llm._read_local_environment", return_value=local):
            with patch.dict("os.environ", {}, clear=True):
                client = TimelyLLMClient(
                    ssl_context=ssl.create_default_context()
                )

        self.assertEqual(client.api_key, "fixture-key")
        self.assertEqual(client.base_url, "https://example.invalid/api")
        self.assertEqual(client.model, "fixture-model")

    def test_reasoner_uses_integer_json_schema_and_accepts_parsed_response(self) -> None:
        client = TimelyLLMClient(
            api_key="fixture-key",
            base_url="https://example.invalid/api",
            model="fixture-model",
            ssl_context=ssl.create_default_context(),
        )
        client._access_token = "fixture-token"
        client._expires_at = float("inf")
        requests: list[dict] = []

        def request(url, method, headers, payload=None):
            requests.append(payload)
            return {
                "type": "final_response",
                "message": "",
                "parsed": {
                    "selected_intent": "fixture intent",
                    "ranked_candidate_ids": [1, 2],
                },
            }

        client._request = request
        prompt = """You are the Reasoner in a fixture prompt
[CANDIDATE SET]
1. first item
2. second item
"""
        response = client.complete(prompt)
        self.assertEqual(json.loads(response)["ranked_candidate_ids"], [1, 2])
        node = requests[0]["chat_model_node"]
        self.assertEqual(node["output_type"], "JSON")
        ranked_ids = node["output_schema"]["properties"]["ranked_candidate_ids"]
        self.assertEqual(ranked_ids["items"]["type"], "integer")
        self.assertEqual(ranked_ids["items"]["enum"], [1, 2])
        self.assertEqual(ranked_ids["minItems"], 2)
        self.assertEqual(ranked_ids["maxItems"], 2)
        self.assertTrue(ranked_ids["uniqueItems"])

    def test_reasoner_repair_prompt_keeps_candidate_constraints(self) -> None:
        client = TimelyLLMClient(
            api_key="fixture-key",
            base_url="https://example.invalid/api",
            ssl_context=ssl.create_default_context(),
        )
        client._access_token = "fixture-token"
        client._expires_at = float("inf")
        requests: list[dict] = []

        def request(url, method, headers, payload=None):
            requests.append(payload)
            return {"type": "final_response", "message": "{}"}

        client._request = request
        client.complete(
            """You are the Reasoner
[CANDIDATE SET]
4. fourth item
7. seventh item

[OUTPUT REPAIR]
Previous response: {"ranked_candidate_ids": [4, 4, 7]}
"""
        )
        ranked_ids = requests[0]["chat_model_node"]["output_schema"][
            "properties"
        ]["ranked_candidate_ids"]
        self.assertEqual(ranked_ids["items"]["enum"], [4, 7])
        self.assertEqual(ranked_ids["minItems"], 2)
        self.assertEqual(ranked_ids["maxItems"], 2)
        self.assertTrue(ranked_ids["uniqueItems"])

    def test_analyst_uses_three_required_string_fields(self) -> None:
        client = TimelyLLMClient(
            api_key="fixture-key",
            base_url="https://example.invalid/api",
            model="fixture-model",
            ssl_context=ssl.create_default_context(),
        )
        client._access_token = "fixture-token"
        client._expires_at = float("inf")
        requests: list[dict] = []

        def request(url, method, headers, payload=None):
            requests.append(payload)
            return {"type": "final_response", "message": "{}"}

        client._request = request
        client.complete("You are the Analyst in a fixture prompt")
        schema = requests[0]["chat_model_node"]["output_schema"]
        self.assertEqual(
            schema["required"], ["diagnosis", "condition", "instruction"]
        )
        self.assertTrue(
            all(field["type"] == "string" for field in schema["properties"].values())
        )


if __name__ == "__main__":
    unittest.main()
