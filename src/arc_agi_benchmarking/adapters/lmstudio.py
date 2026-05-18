import logging
import os
import time
from typing import Any, Dict, List

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice as OpenAIChoice
from openai.types import CompletionUsage

from .openai_base import OpenAIBaseAdapter, _filter_api_kwargs

logger = logging.getLogger(__name__)


class LMStudioAdapter(OpenAIBaseAdapter):
    """Adapter for LM Studio's local OpenAI-compatible server."""

    def init_client(self):
        """Initialize an OpenAI client pointed at the local LM Studio server.

        LM Studio exposes an OpenAI-compatible API (default http://localhost:1234/v1).
        No API key is required, but the OpenAI SDK still wants a non-empty string.
        Override the endpoint with LMSTUDIO_BASE_URL if running on a different host/port.
        """
        base_url = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:4444/v1")
        api_key = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")

        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=1800,
        )

    def _chat_completion(self, messages: List[Dict[str, str]]) -> Any:
        if self.model_config.kwargs.get("reserved_final_grid_tokens"):
            return self._chat_completion_with_reserved_tokens(messages)

        api_kwargs = _filter_api_kwargs(self.model_config.kwargs)
        prompt_chars = sum(len(m.get("content") or "") for m in messages)
        logger.info(
            f"[lmstudio] -> {self.model_config.model_name} "
            f"msgs={len(messages)} prompt_chars={prompt_chars} kwargs={api_kwargs}"
        )
        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                model=self.model_config.model_name, messages=messages, **api_kwargs
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error(
                f"[lmstudio] <- ERROR after {elapsed:.1f}s: {type(e).__name__}: {e}"
            )
            raise
        elapsed = time.perf_counter() - t0
        usage = getattr(resp, "usage", None)
        if usage is not None:
            logger.info(
                f"[lmstudio] <- {self.model_config.model_name} {elapsed:.1f}s "
                f"prompt_tok={getattr(usage, 'prompt_tokens', '?')} "
                f"completion_tok={getattr(usage, 'completion_tokens', '?')} "
                f"total_tok={getattr(usage, 'total_tokens', '?')}"
            )
        else:
            logger.info(f"[lmstudio] <- {self.model_config.model_name} {elapsed:.1f}s (no usage)")
        return resp

    def _chat_completion_with_reserved_tokens(self, messages: List[Dict[str, str]]) -> ChatCompletion:
        """Stream the response, and if the model is still reasoning when it
        nears max_tokens, abort and resend with an assistant prefill that closes
        the thinking block, so the reserved tokens are spent on the final answer.
        """
        api_kwargs = _filter_api_kwargs(self.model_config.kwargs)
        api_kwargs.pop("stream", None)

        budget = api_kwargs.get("max_tokens") or 32768
        reserved = self.model_config.kwargs["reserved_final_grid_tokens"]
        close_marker = self.model_config.kwargs.get(
            "reserved_close_marker", "</think>"
        )
        nudge = self.model_config.kwargs.get(
            "reserved_nudge",
            "\n\nI must stop reasoning now. Final answer grid:\n",
        )
        soft_limit = max(1, budget - reserved)

        prompt_chars = sum(len(m.get("content") or "") for m in messages)
        logger.info(
            f"[lmstudio:reserved] -> {self.model_config.model_name} "
            f"msgs={len(messages)} prompt_chars={prompt_chars} "
            f"budget={budget} soft_limit={soft_limit} reserved={reserved}"
        )

        reasoning_chunks: List[str] = []
        content_chunks: List[str] = []
        chunks_seen = 0
        finish_reason = "stop"
        pass1_usage = None
        last_chunk = None
        truncated = False

        t0 = time.perf_counter()
        stream = self.client.chat.completions.create(
            model=self.model_config.model_name,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **api_kwargs,
        )

        try:
            for chunk in stream:
                last_chunk = chunk
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    rc = getattr(delta, "reasoning_content", None) or ""
                    c = getattr(delta, "content", None) or ""
                    if rc:
                        reasoning_chunks.append(rc)
                    if c:
                        content_chunks.append(c)
                    chunks_seen += 1
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason
                if hasattr(chunk, "usage") and chunk.usage:
                    pass1_usage = chunk.usage

                if chunks_seen >= soft_limit and not content_chunks:
                    truncated = True
                    logger.info(
                        f"[lmstudio:reserved] aborting pass-1 at chunk {chunks_seen} "
                        f"(reasoning_chars={sum(len(r) for r in reasoning_chunks)}) "
                        f"— still in reasoning, forcing close"
                    )
                    break
        finally:
            try:
                stream.close()
            except Exception:
                pass

        pass1_elapsed = time.perf_counter() - t0

        if not truncated:
            final_content = "".join(content_chunks)
            usage_data = pass1_usage or CompletionUsage(
                prompt_tokens=0, completion_tokens=0, total_tokens=0
            )
            logger.info(
                f"[lmstudio:reserved] <- {self.model_config.model_name} "
                f"{pass1_elapsed:.1f}s no-force chunks={chunks_seen} "
                f"content_chars={len(final_content)}"
            )
            return ChatCompletion(
                id=(last_chunk.id if last_chunk else f"stream-{int(time.time())}"),
                choices=[
                    OpenAIChoice(
                        finish_reason=finish_reason,
                        index=0,
                        message=ChatCompletionMessage(
                            content=final_content, role="assistant"
                        ),
                        logprobs=None,
                    )
                ],
                created=int(time.time()),
                model=self.model_config.model_name,
                object="chat.completion",
                usage=usage_data,
            )

        # Pass 2: force the model out of thinking via assistant prefill.
        partial_reasoning = "".join(reasoning_chunks)
        prefill = f"<think>\n{partial_reasoning}\n{close_marker}{nudge}"
        follow_messages = list(messages) + [
            {"role": "assistant", "content": prefill}
        ]

        follow_kwargs = dict(api_kwargs)
        follow_kwargs["max_tokens"] = reserved
        # Stop as soon as the model closes the grid so it can't ramble past it.
        existing_stop = follow_kwargs.get("stop")
        grid_stops = ["]]\n", "]]\r\n", "]]\n\n"]
        if existing_stop is None:
            follow_kwargs["stop"] = grid_stops
        elif isinstance(existing_stop, str):
            follow_kwargs["stop"] = [existing_stop, *grid_stops]
        else:
            follow_kwargs["stop"] = [*existing_stop, *grid_stops]

        t1 = time.perf_counter()
        follow_stream = self.client.chat.completions.create(
            model=self.model_config.model_name,
            messages=follow_messages,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={
                "continue_final_message": True,
                "add_generation_prompt": False,
            },
            **follow_kwargs,
        )

        pass2_chunks: List[str] = []
        pass2_usage = None
        pass2_finish = "stop"
        pass2_last = None
        pass2_seen = 0
        try:
            for chunk in follow_stream:
                pass2_last = chunk
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    c = getattr(delta, "content", None) or ""
                    if c:
                        pass2_chunks.append(c)
                    pass2_seen += 1
                    if chunk.choices[0].finish_reason:
                        pass2_finish = chunk.choices[0].finish_reason
                if hasattr(chunk, "usage") and chunk.usage:
                    pass2_usage = chunk.usage
        finally:
            try:
                follow_stream.close()
            except Exception:
                pass

        pass2_elapsed = time.perf_counter() - t1
        final_content = "".join(pass2_chunks)
        # OpenAI-compatible servers strip the stop string. If we stopped on a
        # grid close, reattach `]]` so the parser sees a complete 2D array.
        if pass2_finish == "stop" and final_content.rstrip().endswith("]"):
            stripped = final_content.rstrip()
            if not stripped.endswith("]]"):
                final_content = stripped + "]"
        elif pass2_finish == "stop" and "[[" in final_content and "]]" not in final_content:
            final_content = final_content.rstrip() + "]]"

        # Pass 1 reasoning tokens are real spend even though we aborted the stream:
        # track them on prompt-side accounting is wrong; treat them as reasoning tokens
        # by surfacing them in completion_tokens. Pass 2 prompt re-bills the original
        # prompt + the prefill, which is how LM Studio actually charges (locally free,
        # but useful for pricing math).
        p1_pt = getattr(pass1_usage, "prompt_tokens", 0) if pass1_usage else 0
        p1_ct = getattr(pass1_usage, "completion_tokens", 0) if pass1_usage else chunks_seen
        p2_pt = getattr(pass2_usage, "prompt_tokens", 0) if pass2_usage else 0
        p2_ct = getattr(pass2_usage, "completion_tokens", 0) if pass2_usage else pass2_seen

        combined_usage = CompletionUsage(
            prompt_tokens=p1_pt + p2_pt,
            completion_tokens=p1_ct + p2_ct,
            total_tokens=p1_pt + p1_ct + p2_pt + p2_ct,
        )

        logger.info(
            f"[lmstudio:reserved] <- {self.model_config.model_name} "
            f"forced pass1={pass1_elapsed:.1f}s pass2={pass2_elapsed:.1f}s "
            f"reasoning_chars={len(partial_reasoning)} answer_chars={len(final_content)} "
            f"pass2_finish={pass2_finish}"
        )

        return ChatCompletion(
            id=(pass2_last.id if pass2_last else f"stream-{int(time.time())}"),
            choices=[
                OpenAIChoice(
                    finish_reason=pass2_finish,
                    index=0,
                    message=ChatCompletionMessage(
                        content=final_content, role="assistant"
                    ),
                    logprobs=None,
                )
            ],
            created=int(time.time()),
            model=self.model_config.model_name,
            object="chat.completion",
            usage=combined_usage,
        )
