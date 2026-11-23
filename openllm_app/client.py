"""Streaming chat against llama-server's OpenAI-compatible /v1/chat/completions.

Field names (reasoning_content, usage, timings.predicted_per_second) follow upstream
llama-server; confirmed against the PrismML build b10809 in Task 5 step 5. That build sends no `usage` in the
stream, so usage is derived from timings (prompt_n + cache_n, predicted_n) when absent.
"""
import json
from dataclasses import dataclass
from typing import AsyncIterator, Literal

Kind = Literal["reasoning", "content", "usage", "timings", "done", "error"]


@dataclass
class ChatEvent:
    kind: Kind
    text: str = ""
    data: dict | None = None


async def stream_chat(base_url: str, api_key: str | None, model_id: str,
                      messages: list[dict], max_tokens: int) -> AsyncIterator[ChatEvent]:
    import httpx  # imported on first use: it is a fifth of the UI's start-up import time

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    body = {"model": model_id, "messages": messages, "max_tokens": max_tokens, "stream": True}
    finish, saw_content, finished, saw_usage = None, False, False, False
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None)) as client:
            async with client.stream("POST", f"{base_url}/chat/completions", json=body, headers=headers) as r:
                if r.status_code != 200:
                    detail = (await r.aread()).decode(errors="replace")[:200]
                    yield ChatEvent("error", f"HTTP {r.status_code}: {detail}")
                    return
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        finished = True
                        break
                    try:
                        obj = json.loads(payload)
                    except ValueError:
                        continue
                    for choice in obj.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("reasoning_content"):
                            yield ChatEvent("reasoning", delta["reasoning_content"])
                        if delta.get("content"):
                            saw_content = True
                            yield ChatEvent("content", delta["content"])
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
                    if obj.get("usage"):
                        saw_usage = True
                        u = obj["usage"]
                        yield ChatEvent("usage", data={"prompt_tokens": u.get("prompt_tokens", 0),
                                                       "completion_tokens": u.get("completion_tokens", 0)})
                    if obj.get("timings"):
                        t = obj["timings"]
                        yield ChatEvent("timings", data=t)
                        if not saw_usage and "predicted_n" in t:
                            # PrismML llama-server b10809 streams timings but no usage object.
                            yield ChatEvent("usage", data={
                                "prompt_tokens": t.get("prompt_n", 0) + t.get("cache_n", 0),
                                "completion_tokens": t["predicted_n"]})
    except httpx.HTTPError as e:
        yield ChatEvent("error", f"{type(e).__name__}: {e}" if str(e) else type(e).__name__)
        return
    if not finished:
        yield ChatEvent("error", "connection closed before the reply finished")
        return
    yield ChatEvent("done", data={"finish_reason": finish, "empty_answer": not saw_content})
