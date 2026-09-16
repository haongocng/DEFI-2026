from __future__ import annotations

import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol
from uuid import uuid4


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


class HeuristicLLMClient:
    """Deterministic, API-free smoke-test client; never use for final results."""

    _candidate = re.compile(r"(?m)^(\d+)\.\s+(.+)$")

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.casefold()))

    def complete(self, prompt: str) -> str:
        if prompt.startswith("You are the Reasoner"):
            history_start = prompt.find("[CURRENT SESSION INTERACTIONS")
            candidate_start = prompt.find("[CANDIDATE SET]")
            if candidate_start < 0:
                raise ValueError("Heuristic client could not find candidate block")
            history = prompt[history_start:candidate_start]
            candidate_block = prompt[candidate_start + len("[CANDIDATE SET]") :]
            history_tokens = self._tokens(history)
            candidates = [
                (int(match.group(1)), match.group(2).strip())
                for match in self._candidate.finditer(candidate_block)
            ]
            ranked = sorted(
                candidates,
                key=lambda item: (
                    -len(history_tokens & self._tokens(item[1])),
                    item[0],
                ),
            )
            return json.dumps(
                {
                    "selected_intent": "continue the most coherent recent item pattern",
                    "ranked_candidate_ids": [item[0] for item in ranked],
                }
            )
        if prompt.startswith("You are the Analyst"):
            return json.dumps(
                {
                    "condition": "the session contains a coherent recent product pattern",
                    "instruction": "prioritize candidates that directly continue the most recent coherent pattern",
                    "diagnosis": "the smoke-test ranker underweighted the recent pattern",
                }
            )
        raise ValueError("Heuristic client received an unknown prompt type")


def _read_local_environment() -> dict[str, str]:
    """Read MODEL/.env without overriding process-level configuration."""
    project_file = Path(__file__).resolve().parents[2] / ".env"
    candidates = (Path.cwd() / ".env", project_file)
    values: dict[str, str] = {}
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values.setdefault(
                key.strip(), value.strip().strip('"').strip("'")
            )
    return values


def _verified_ssl_context() -> ssl.SSLContext:
    configured = os.getenv("SSL_CERT_FILE")
    if configured:
        return ssl.create_default_context(cafile=configured)
    system_bundle = Path("/etc/ssl/cert.pem")
    if system_bundle.is_file():
        return ssl.create_default_context(cafile=str(system_bundle))
    return ssl.create_default_context()


def _http_error(exc: urllib.error.HTTPError) -> RuntimeError:
    try:
        detail = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        detail = ""
    suffix = f": {detail[:1000]}" if detail else ""
    return RuntimeError(f"Timely API returned HTTP {exc.code}{suffix}")


def _structured_output_schema(prompt: str) -> dict | None:
    """Return a small stage-specific schema for Timely-supported JSON output."""

    if prompt.startswith("You are the Reasoner"):
        candidate_ids: list[int] = []
        if "[CANDIDATE SET]" in prompt:
            candidate_block = prompt.split("[CANDIDATE SET]", 1)[1]
            candidate_block = candidate_block.split("[OUTPUT REPAIR]", 1)[0]
            candidate_ids = list(
                dict.fromkeys(
                    int(match.group(1))
                    for match in re.finditer(
                        r"(?m)^(\d+)\.\s", candidate_block
                    )
                )
            )
        ranked_ids: dict = {
            "type": "array",
            "items": {"type": "integer"},
        }
        if candidate_ids:
            # Timely/Gemini can constrain array length and allowed integer values.
            # Exact uniqueness/full-set equality is still enforced locally.
            ranked_ids.update(
                {
                    "minItems": len(candidate_ids),
                    "maxItems": len(candidate_ids),
                    "uniqueItems": True,
                }
            )
            ranked_ids["items"]["enum"] = candidate_ids
        return {
            "type": "object",
            "properties": {
                "selected_intent": {"type": "string"},
                "ranked_candidate_ids": ranked_ids,
            },
            "required": ["selected_intent", "ranked_candidate_ids"],
        }
    if prompt.startswith("You are the Analyst"):
        return {
            "type": "object",
            "properties": {
                "diagnosis": {"type": "string"},
                "condition": {"type": "string"},
                "instruction": {"type": "string"},
            },
            "required": ["diagnosis", "condition", "instruction"],
        }
    return None


class TimelyLLMClient:
    """Small synchronous adapter for the API used by the legacy repository."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: int = 120,
        ssl_context: ssl.SSLContext | None = None,
        structured_output: bool = True,
    ):
        local = _read_local_environment()
        self.api_key = (
            api_key
            or os.getenv("TIMELY_API_KEY")
            or os.getenv("TIMELYGPT_API_KEY")
            or local.get("TIMELY_API_KEY")
            or local.get("TIMELYGPT_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "TIMELY_API_KEY (or legacy TIMELYGPT_API_KEY) is required"
            )
        self.base_url = (
            base_url
            or os.getenv("TIMELY_BASE_URL")
            or local.get("TIMELY_BASE_URL")
            or "https://hello.timelygpt.co.kr/api/v2/chat"
        ).rstrip("/")
        self.model = (
            model
            or os.getenv("MODEL_NAME")
            or local.get("MODEL_NAME")
            or "gemini-2.5-flash"
        )
        self.timeout_seconds = timeout_seconds
        self.ssl_context = ssl_context or _verified_ssl_context()
        self.structured_output = structured_output
        self._access_token: str | None = None
        self._expires_at = 0.0

    def _request(
        self,
        url: str,
        method: str,
        headers: dict[str, str],
        payload: dict | None = None,
    ) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(
            request,
            timeout=self.timeout_seconds,
            context=self.ssl_context,
        ) as response:
            return json.loads(response.read().decode("utf-8"))

    def _authenticate(self) -> None:
        if self._access_token and time.time() < self._expires_at:
            return
        response = self._request(
            f"{self.base_url}/sdk-auth/authenticate",
            "GET",
            {"Content-Type": "application/json", "X-Timely-API": self.api_key},
        )
        self._access_token = response.get("data", {}).get("access_token")
        if not self._access_token:
            raise RuntimeError(f"Authentication failed: {response}")
        self._expires_at = time.time() + 55 * 60

    def complete(self, prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                self._authenticate()
                chat_model_node: dict = {
                    "model": self.model,
                    "temperature": 0,
                }
                output_schema = (
                    _structured_output_schema(prompt)
                    if self.structured_output
                    else None
                )
                if output_schema is not None:
                    chat_model_node.update(
                        {
                            "output_type": "JSON",
                            "output_schema": output_schema,
                        }
                    )
                response = self._request(
                    f"{self.base_url}/llm-completion",
                    "POST",
                    {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self._access_token}",
                    },
                    {
                        # Every DRAP call is independent; no hidden chat history may
                        # leak between samples or experimental seeds.
                        "session_id": f"drap_call_{uuid4()}",
                        "messages": [{"role": "user", "content": prompt}],
                        "chat_model_node": chat_model_node,
                        "chat_type": "DYNAMIC_CHAT",
                        "stream": False,
                        "locale": "en",
                    },
                )
                if response.get("type") == "final_response":
                    message = response.get("message")
                    if isinstance(message, str) and message.strip():
                        return message
                    parsed = response.get("parsed")
                    if parsed is not None:
                        return json.dumps(parsed, ensure_ascii=False)
                    return ""
                raise RuntimeError(f"Unexpected LLM response: {response}")
            except urllib.error.HTTPError as exc:
                last_error = _http_error(exc)
                if exc.code == 401:
                    self._access_token = None
                retryable = exc.code in (401, 429, 500, 502, 503, 504)
                if not retryable or attempt == 2:
                    raise last_error from exc
                time.sleep(2**attempt)
            except (TimeoutError, urllib.error.URLError) as exc:
                last_error = exc
                if attempt == 2:
                    raise RuntimeError("Timely API request failed") from exc
                time.sleep(2**attempt)
        raise RuntimeError("LLM request failed") from last_error


def parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
