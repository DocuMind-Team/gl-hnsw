from __future__ import annotations

import json
import re
import time

from hnsw_logic.core.utils import append_jsonl, utc_now


class OpenAIProviderTransportMixin:
    def _trace_remote(self, stage: str, status: str, detail: str = "") -> None:
        if self.trace_path is None:
            return
        append_jsonl(
            self.trace_path,
            [
                {
                    "timestamp": utc_now(),
                    "stage": stage,
                    "status": status,
                    "detail": detail[:240],
                    "model": self.config.chat_model,
                }
            ],
        )

    def _handle_remote_failure(self, stage: str, exc: Exception) -> None:
        self._trace_remote(stage, "remote_failure", str(exc))
        if self.require_remote:
            raise RuntimeError(f"Remote provider call failed during {stage}: {exc}") from exc

    def _is_content_filter_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "content_filter" in message or "moderation block" in message or "'code': '421'" in message

    def _is_response_parse_error(self, exc: Exception) -> bool:
        if isinstance(exc, json.JSONDecodeError):
            return True
        message = str(exc).lower()
        return (
            "jsondecodeerror" in message
            or "expecting ',' delimiter" in message
            or "unterminated string" in message
            or "extra data" in message
            or "empty response body" in message
        )

    def _is_output_limit_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "router_output_limitation" in message or "output token rate limit exceeded" in message

    def _is_connection_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "connection error" in message
            or "apiconnectionerror" in message
            or "connecterror" in message
            or "unexpected eof while reading" in message
            or "ssl:" in message
            or "timed out" in message
            or "timeout" in message
            or "remoteprotocolerror" in message
        )

    def _parse_json(self, content: str):
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(1))

    def _invoke_json(self, system_prompt: str, user_prompt: str, *, thinking: bool = False, stage: str = "generic"):
        from langchain_core.messages import HumanMessage, SystemMessage

        thinking_enabled = thinking
        current_user_prompt = user_prompt
        last_exc: Exception | None = None
        max_attempts = 5
        for attempt in range(max_attempts):
            kwargs = {
                "extra_body": {
                    "thinking": {"type": "enabled" if thinking_enabled else "disabled"}
                }
            }
            try:
                response = self._chat.invoke([SystemMessage(content=system_prompt), HumanMessage(content=current_user_prompt)], **kwargs)
                content = response.content if isinstance(response.content, str) else "".join(part.get("text", "") for part in response.content)
                if not str(content).strip():
                    raise ValueError("empty response body")
                payload = self._parse_json(content)
                self._trace_remote(stage, "success", f"attempt={attempt + 1}")
                return payload
            except Exception as exc:
                last_exc = exc
                self._trace_remote(stage, "error", f"attempt={attempt + 1}: {exc}")
                if attempt == 0 and (self._is_response_parse_error(exc) or "empty response body" in str(exc).lower()):
                    current_user_prompt = (
                        f"{user_prompt}\n\n"
                        "Previous response was not valid JSON. Return exactly one JSON object or JSON array matching the schema. "
                        "Do not include markdown fences, prose, or explanations outside JSON."
                    )
                    continue
                if thinking_enabled and self._is_output_limit_error(exc):
                    thinking_enabled = False
                    current_user_prompt = (
                        f"{user_prompt}\n\n"
                        "Keep the response concise. Return only the required JSON fields. "
                        "Use short evidence spans and a brief decision_reason."
                    )
                    continue
                if self._is_connection_error(exc) and attempt < max_attempts - 1:
                    backoff_seconds = min(8.0, 1.5 * (2**attempt))
                    self._trace_remote(stage, "retry", f"sleep={backoff_seconds:.1f}s after attempt={attempt + 1}")
                    time.sleep(backoff_seconds)
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Remote provider call failed during {stage}: unknown error")
