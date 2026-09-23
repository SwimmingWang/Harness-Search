"""Agent implementations and OpenAI orchestration helpers."""

import json
import os
import json_repair
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast
from openai import OpenAI
import anthropic
import requests
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.responses import Response
from openai.types.responses.response_function_tool_call import (
    ResponseFunctionToolCall,
)
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_refusal import ResponseOutputRefusal
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from openai_harmony import (
    HarmonyEncoding,
    HarmonyEncodingName,
    Message,
    RenderConversationConfig,
    load_harmony_encoding,
)
import tinker
from datagen.search_dataset import SearchDataset, get_dataset
import structlog
from harness_search.prompts import get_retrieval_subagent_budget_exhausted_message
from harness_search.rerank import VLLMQwen3Reranker
from harness_search.tasks import DOC_ID_PATTERN, SearchTaskOutput, SearchTaskEvaluationOutput
from harness_search.tools import (
    GrepCorpusToolCallMetadata,
    PruneChunksTool,
    ReadTool,
    SearchTool,
    SearchToolCallMetadata,
    Tool,
    ToolCallMetadata,
    ToolSet,
    UserTextTool,
)
from harness_search.trajectory import (
    Action,
    ActionBuilder,
    Observation,
    ObservationBuilder,
    Trajectory,
    TrajectoryBuilder,
)
from harness_search.utils import ProviderFormat
from concurrent.futures import Future
import tenacity


logger = structlog.get_logger("search_agent.agent")

# ============================================================================
# Agent Base Class
# ============================================================================


@dataclass
class InferenceContext:
    """Context for inference, including trajectory and any constraints.

    Used when driving the agent manually (e.g., in RL) to get both the
    trajectory prepared for inference and any constraints that should apply.

    The previous_response_id is used for OpenAI Responses API session continuity.
    Inference models should read from and write to this field, unless
    skip_response_id_update is True (indicating the trajectory was modified
    and the response chain should not be updated).
    """

    trajectory: Trajectory
    toolset: ToolSet
    max_tokens: Optional[int] = None
    previous_response_id: Optional[str] = None
    skip_response_id_update: bool = False


class AgentResult(ABC):
    """A base class for the final result of an agent."""


class AgentInferenceModel(ABC):
    """A base class for all inference models."""

    @abstractmethod
    def __call__(self, context: InferenceContext) -> Optional[Action]:
        """Given an inference context, return the next action."""
        pass


class OpenAIAgentInferenceModel(AgentInferenceModel):
    """Inference model that uses OpenAI Responses or Chat Completions API."""

    def __init__(
        self,
        openai_client: OpenAI,
        model: str = "gpt-5",
        max_output_tokens: int = 4096,
        temperature: float = 1.0,
        reasoning_effort: Optional[str] = None,
        api_style: str = "responses",
    ):
        self.openai_client = openai_client
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.api_style = api_style

    def __call__(self, context: InferenceContext) -> Optional[Action]:
        if self.api_style == "chat_completions":
            return self._call_chat_completions(context)
        if self.api_style not in {"responses", "auto"}:
            raise ValueError(f"Unsupported OpenAI api_style: {self.api_style}")

        if self.api_style == "auto":
            try:
                return self._call_responses_api(context)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Responses API failed, falling back to chat completions",
                    model=self.model,
                    error=str(exc),
                )
                context.previous_response_id = None
                return self._call_chat_completions(context)

        return self._call_responses_api(context)

    def _call_responses_api(self, context: InferenceContext) -> Optional[Action]:
        trajectory = context.trajectory
        toolset = context.toolset
        max_tokens = context.max_tokens

        if context.previous_response_id:
            entries_to_send: List[Observation] = []
            encountered_action = False
            has_user_observation = False
            for entry in reversed(trajectory.actions_and_observations):
                if isinstance(entry, Observation):
                    if any(source == "user" for source in entry.sources):
                        has_user_observation = True
                    if any(source != "user" for source in entry.sources):
                        entries_to_send.append(entry)
                    continue
                if isinstance(entry, Action):
                    encountered_action = True
                    break
            if has_user_observation or not encountered_action or not entries_to_send:
                context.previous_response_id = None
                request_input = trajectory.to_provider_format(
                    ProviderFormat.OPENAI_RESPONSES
                )
            else:
                request_input = trajectory.to_openai_responses_subset(
                    reversed(entries_to_send)
                )
        else:
            request_input = trajectory.to_provider_format(
                ProviderFormat.OPENAI_RESPONSES
            )
        request_tools = toolset.get_formats(ProviderFormat.OPENAI)
        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": request_input,
            "tools": request_tools,  # type: ignore[arg-type]
            "parallel_tool_calls": True,
            "temperature": self.temperature,
            "max_output_tokens": max_tokens or self.max_output_tokens,
        }
        if context.previous_response_id:
            request_kwargs["previous_response_id"] = context.previous_response_id
        if self.reasoning_effort:
            request_kwargs["reasoning"] = {"effort": self.reasoning_effort}

        response = self.openai_client.responses.create(**request_kwargs)

        # Only update response ID if not skipping
        if not context.skip_response_id_update:
            context.previous_response_id = response.id
        return self._response_to_action(response, toolset)

    def _call_chat_completions(self, context: InferenceContext) -> Optional[Action]:
        trajectory = context.trajectory
        toolset = context.toolset
        max_tokens = context.max_tokens

        request_messages = trajectory.to_provider_format(ProviderFormat.OPENAI)
        request_tools = toolset.get_formats(ProviderFormat.OPENAI)
        response: ChatCompletion = self.openai_client.chat.completions.create(
            messages=request_messages,
            tools=request_tools,  # type: ignore[arg-type]
            parallel_tool_calls=True,
            model=self.model,
            temperature=self.temperature,
            max_completion_tokens=max_tokens or self.max_output_tokens,
        )
        if not response.choices:
            raise RuntimeError("No response choices received from OpenAI")

        choice = response.choices[0]
        message: ChatCompletionMessage = choice.message
        action_builder = ActionBuilder()

        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content:
            action_builder.add_reasoning(reasoning_content)

        if choice.finish_reason == "stop":
            text = self._extract_chat_message_text(message)
            if text:
                action_builder.add_tool_call(UserTextTool(), {"text": text}, "agent")
        elif (
            choice.finish_reason == "tool_calls"
            and message.tool_calls
            and len(message.tool_calls) > 0
        ):
            for tool_call in message.tool_calls:
                if not hasattr(tool_call, "function"):
                    raise ValueError("Tool call is missing function payload")
                tool = toolset.get_tool(tool_call.function.name)
                if tool is None:
                    raise ValueError(
                        "Model requested unknown tool or tool not in toolset: "
                        f"{tool_call.function.name}"
                    )
                try:
                    parsed_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON arguments for tool {tool_call.function.name}"
                    ) from exc
                action_builder.add_tool_call(tool, parsed_args, tool_call.id)
        else:
            text = self._extract_chat_message_text(message)
            if text:
                action_builder.add_tool_call(UserTextTool(), {"text": text}, "agent")

        return action_builder.build()

    def _response_to_action(self, response: Response, toolset: ToolSet) -> Action:
        if not response.output:
            raise RuntimeError("No response output received from OpenAI Responses API")
        action_builder = ActionBuilder()
        reasoning_chunks: List[str] = []

        for output in response.output or []:
            if isinstance(output, ResponseReasoningItem):
                reasoning_chunks.extend(
                    summary.text for summary in output.summary if summary.text
                )
            elif isinstance(output, ResponseOutputMessage):
                message_text = self._extract_message_text(output)
                if message_text:
                    action_builder.add_tool_call(
                        UserTextTool(), {"text": message_text}, "agent"
                    )
            elif isinstance(output, ResponseFunctionToolCall):
                tool = toolset.get_tool(output.name)
                if tool is None:
                    raise ValueError(
                        "Model requested unknown tool or tool not in toolset: "
                        f"{output.name}"
                    )
                if output.call_id is None:
                    raise ValueError(
                        f"Function call for tool {output.name} missing call_id"
                    )
                args_str = output.arguments or "{}"
                try:
                    params = json.loads(args_str)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON arguments for tool {output.name}"
                    ) from exc
                action_builder.add_tool_call(tool, params, output.call_id)
            else:
                logger.warning(
                    "Unsupported response output type from OpenAI Responses API",
                    output_type=getattr(output, "type", type(output).__name__),
                )

        if reasoning_chunks:
            action_builder.add_reasoning("\n\n".join(reasoning_chunks))

        return action_builder.build()

    def _extract_message_text(self, message: ResponseOutputMessage) -> str:
        text_parts: List[str] = []
        for content in message.content:
            if isinstance(content, ResponseOutputText):
                text_parts.append(content.text)
            elif isinstance(content, ResponseOutputRefusal):
                text_parts.append(content.refusal)
            else:
                logger.warning(
                    "Unsupported message content type from OpenAI Responses API",
                    content_type=getattr(content, "type", type(content).__name__),
                )
        return "".join(text_parts).strip()

    def _extract_chat_message_text(self, message: ChatCompletionMessage) -> str:
        content = message.content
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
            return "".join(text_parts).strip()
        return str(content).strip()


class AnthropicAgentInferenceModel(AgentInferenceModel):

    def __init__(
        self,
        anthropic_client: anthropic.Anthropic,
        model: str = "claude-opus-4-5-20251101",  # "claude-sonnet-4-5-20250929" for sonnet
        max_tokens: int = 4096,
        temperature: float = 1.0,
        thinking_budget: int = 6000,
        betas: List[str] = ["interleaved-thinking-2025-05-14"],
    ):
        self.anthropic_client = anthropic_client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.betas = betas

    def __call__(self, context: InferenceContext) -> Optional[Action]:
        trajectory = context.trajectory
        toolset = context.toolset
        max_tokens = context.max_tokens

        with self.anthropic_client.beta.messages.stream(
            model=self.model,
            tools=toolset.get_formats(ProviderFormat.ANTHROPIC),  # type: ignore
            messages=trajectory.to_provider_format(ProviderFormat.ANTHROPIC),  # type: ignore
            betas=self.betas,
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature,
            thinking={"type": "enabled", "budget_tokens": self.thinking_budget},
        ) as stream:
            chunks = [chunk for chunk in stream]
        if not chunks:
            raise RuntimeError("No response chunks received from Anthropic stream")

        last_chunk = chunks[-1]
        if last_chunk.type != "message_stop":
            raise ValueError("Last chunk is not a message stop, cannot proceed.")

        assistant_content_out = ""
        thinking_content_out = ""
        thinking_signature_out = ""

        for chunk in chunks:
            if chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                assistant_content_out += chunk.delta.text
            if (
                chunk.type == "content_block_delta"
                and chunk.delta.type == "thinking_delta"
            ):
                thinking_content_out += chunk.delta.thinking
            if (
                chunk.type == "content_block_delta"
                and chunk.delta.type == "signature_delta"
            ):
                thinking_signature_out += chunk.delta.signature
            if (
                chunk.type == "content_block_delta"
                and chunk.delta.type == "redacted_thinking_delta"
            ):
                raise ValueError("Redacted thinking delta encountered, cannot proceed.")
            if chunk.type == "redacted_thinking":
                raise ValueError("Redacted thinking encountered, cannot proceed.")

        action_builder = ActionBuilder()

        if thinking_content_out and len(thinking_content_out) > 0:
            action_builder.add_reasoning(thinking_content_out, thinking_signature_out)
        if assistant_content_out and len(assistant_content_out) > 0:
            action_builder.add_tool_call(
                UserTextTool(), {"text": assistant_content_out}, "agent"
            )

        if last_chunk.type == "message_stop":
            for block in last_chunk.message.content:
                if block.type != "tool_use":
                    continue
                tool = toolset.get_tool(block.name)
                if tool is None:
                    raise ValueError(
                        "Model requested unknown tool or tool not in toolset: "
                        f"{block.name}"
                    )
                action_builder.add_tool_call(tool, block.input, block.id)

        return action_builder.build()


class TinkerAgentInferenceModel(AgentInferenceModel):

    client: tinker.SamplingClient
    sampling_params: tinker.types.SamplingParams
    enc: HarmonyEncoding
    model: str
    sampling_timeout_secs: int
    strict_mode: bool
    context_window: int

    def __init__(
        self,
        tinker_sampling_client: tinker.SamplingClient,
        model: str = "Qwen/Qwen3.5-27B",
        max_completion_tokens: int = 4096,
        temperature: float = 1.0,
        sampling_timeout_secs: int = 500,
        strict_mode: bool = True,
        context_window: int = 32768,
    ):
        self.client = tinker_sampling_client
        self.strict_mode = strict_mode
        self.context_window = context_window
        if model in ("openai/gpt-oss-20b", "openai/gpt-oss-120b"):
            self.enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            self.sampling_params = tinker.types.SamplingParams(
                temperature=temperature,
                max_tokens=max_completion_tokens,
                top_p=0.9,
                stop=[200002, 200012],
            )
        else:
            raise ValueError(f"Unsupported model: {model}")
        self.model = model
        self.sampling_timeout_secs = sampling_timeout_secs

    @staticmethod
    def _extract_first_json_object(s: str) -> Optional[str]:
        """Return the substring for the first balanced top-level JSON object/array.

        Walks the string tracking brace/bracket depth and string quoting so
        that trailing garbage (extra text, duplicate objects, ``[END]``
        markers, etc.) is silently discarded.  Returns ``None`` when no
        balanced object is found.
        """
        # Find the opening delimiter
        start = -1
        open_ch = ""
        for i, ch in enumerate(s):
            if ch in ('{', '['):
                start = i
                open_ch = ch
                break
        if start < 0:
            return None

        close_ch = '}' if open_ch == '{' else ']'
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if esc:
                esc = False
                continue
            if ch == '\\' and in_str:
                esc = True
                continue
            if ch == '"' and not esc:
                in_str = not in_str
                continue
            if not in_str:
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        return s[start: i + 1]
        return None

    @staticmethod
    def _repair_json_escapes(s: str) -> str:
        """Fix invalid backslash escapes that are illegal in JSON."""
        s = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', s)
        s = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', s)
        return s

    @staticmethod
    def _parse_json(json_string: str, strict_mode: bool = True) -> Any:
        """Parse JSON string with automatic fallback repairs.

        Repair pipeline (strict_mode=True):
        1. ``json.loads`` on the raw string.
        2. Extract the first balanced JSON object/array, discard trailing
           garbage, then ``json.loads`` again.
        3. Additionally fix invalid backslash escapes, then retry.
        Non-strict mode delegates to ``json_repair``.
        """
        if not strict_mode:
            return json_repair.loads(json_string)

        # 1. Fast path – raw string parses cleanly
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            pass

        # 2. Extract first JSON object, ignore trailing garbage
        first_obj = TinkerAgentInferenceModel._extract_first_json_object(json_string)
        if first_obj is not None:
            try:
                return json.loads(first_obj)
            except json.JSONDecodeError:
                pass

            # 3. Also fix bad escapes on the extracted object
            repaired = TinkerAgentInferenceModel._repair_json_escapes(first_obj)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

        # Preserve malformed policy output before attempting permissive repair.
        # Log repaired calls too, so evaluation can quantify format failures.
        try:
            repaired = json_repair.loads(json_string)
        except Exception:
            logger.exception(
                "tool_json_repair_failed",
                output_chars=len(json_string),
                raw_output=json_string[:8000],
            )
            # Re-raise the stdlib error with useful line/column details.
            return json.loads(json_string)

        logger.warning(
            "tool_json_repaired",
            output_chars=len(json_string),
            raw_output=json_string[:8000],
        )
        return repaired

    @staticmethod
    def handle_tool_message(
        message: Message,
        toolset: ToolSet,
        action_builder: ActionBuilder,
        strict_mode: bool = True,
    ) -> None:
        if message.recipient == "functions.multi_tool_use":
            args = TinkerAgentInferenceModel._parse_json(
                message.content[0].text, strict_mode
            )
            tool_calls: List[Dict[str, Any]] = []
            if isinstance(args, list):
                tool_calls = args
            elif isinstance(args, dict):
                tool_calls = args["tool_calls"]
            else:
                raise ValueError(f"Invalid tool calls: {args}")
            for tool_call in tool_calls:
                # Harmony formats tool names with a functions. prefix, remove it
                raw_name = tool_call.get("tool_name")
                if raw_name is None:
                    raise ValueError("Tool call missing 'tool_name'")
                parsed_tool_name = (raw_name or "").replace("functions.", "").replace("<|constrain|>", "").strip()
                if not parsed_tool_name:
                    raise ValueError("Tool name empty after parsing")
                tool = toolset.get_tool(parsed_tool_name)
                if tool is None:
                    raise ValueError(f"Tool not found: {parsed_tool_name}")
                tool_args = tool_call["parameters"]
                source = tool_call["tool_name"] + "_" + uuid.uuid4().hex
                action_builder.add_tool_call(tool=tool, params=tool_args, source=source)
        else:
            # Harmony formats tool names with a functions. prefix, remove it
            recipient = message.recipient
            if recipient is None:
                raise ValueError("Tool message has no recipient (malformed output)")
            parsed_tool_name = (recipient or "").replace("functions.", "").replace("<|constrain|>", "").strip()
            if not parsed_tool_name:
                raise ValueError("Tool name empty after parsing recipient")
            tool = toolset.get_tool(parsed_tool_name)
            if tool is None:
                raise ValueError(f"Tool not found: {parsed_tool_name}")
            tool_args = TinkerAgentInferenceModel._parse_json(
                message.content[0].text, strict_mode
            )
            source = (recipient or "") + "_" + uuid.uuid4().hex
            action_builder.add_tool_call(tool=tool, params=tool_args, source=source)

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=15),
        before_sleep=lambda _: logger.warning("Retrying tinker sample..."),
    )
    def _sample(
        self,
        tinker_tokens: tinker.types.ModelInput,
        sampling_params: tinker.types.SamplingParams,
    ) -> tinker.types.SampleResponse:
        fut = self.client.sample(
            prompt=tinker_tokens, sampling_params=sampling_params, num_samples=1
        )
        return fut.result(self.sampling_timeout_secs)

    def __call__(self, context: InferenceContext) -> Optional[Action]:
        trajectory = context.trajectory
        toolset = context.toolset
        max_tokens = context.max_tokens

        if self.model in ("openai/gpt-oss-20b", "openai/gpt-oss-120b"):
            request_messages = trajectory.to_provider_format(
                ProviderFormat.OPENAI_HARMONY
            )
        else:
            raise ValueError(f"Unsupported model: {self.model}")

        # TODO: append new role?
        input_tokens = self.enc.render_conversation(
            request_messages, config=RenderConversationConfig(auto_drop_analysis=False)
        )
        prompt_length = len(input_tokens)

        # Cap max_tokens to fit within context window (aligned with Kelly's eval)
        available_tokens = self.context_window - prompt_length - 100
        requested_max_tokens = max_tokens or self.sampling_params.max_tokens

        if available_tokens < requested_max_tokens:
            logger.warning(
                "Capping max_tokens to fit context window",
                prompt_length=prompt_length,
                requested_max_tokens=requested_max_tokens,
                available_tokens=available_tokens,
                context_window=self.context_window,
            )
            effective_max_tokens = max(256, available_tokens)
        else:
            effective_max_tokens = requested_max_tokens

        tinker_tokens = tinker.types.ModelInput.from_ints(tokens=input_tokens)
        sampling_params = self.sampling_params.model_copy(
            update={"max_tokens": effective_max_tokens}
        )

        result = self._sample(tinker_tokens, sampling_params)
        resp_tokens = result.sequences[0].tokens
        return TinkerAgentInferenceModel.harmony_tinker_tokens_to_action(
            self.enc, resp_tokens, toolset, self.strict_mode
        )

    @staticmethod
    def tinker_tokens_to_harmony_format(
        encoding: HarmonyEncoding, tokens: List[int]
    ) -> List[Message]:
        return encoding.parse_messages_from_completion_tokens(tokens)

    @staticmethod
    def harmony_tinker_tokens_to_action(
        encoding: HarmonyEncoding,
        tokens: List[int],
        toolset: ToolSet,
        strict_mode: bool = True,
    ) -> Action:
        action_builder = ActionBuilder()
        parsed = TinkerAgentInferenceModel.tinker_tokens_to_harmony_format(
            encoding, tokens
        )
        for i, message in enumerate[Message](parsed):
            if message.channel == "analysis":
                # NOTE: GPT oss 20b occasionally outputs a tool call on analysis, since built in tools are allowed to do so
                # we respect the call and redirect to commentary channel for now
                if message.recipient:
                    logger.warning(
                        "Output tool call on analysis channel, redirecting to commentary channel"
                    )
                    TinkerAgentInferenceModel.handle_tool_message(
                        message, toolset, action_builder, strict_mode
                    )
                else:
                    action_builder.add_reasoning(message.content[0].text)

            elif message.channel == "commentary":
                TinkerAgentInferenceModel.handle_tool_message(
                    message, toolset, action_builder, strict_mode
                )
            elif message.channel == "final":
                action_builder.add_tool_call(
                    tool=UserTextTool(),
                    params={"text": str(message.content[0].text)},
                    source="agent",
                )
            elif message.channel is None:
                # Handle messages with no channel - likely incomplete/malformed tokens
                # Try to extract any text content as reasoning if available
                if (
                    message.content
                    and hasattr(message.content[0], "text")
                    and message.content[0].text
                ):
                    logger.debug(
                        f"Message with None channel, treating as reasoning: {message.content[0].text[:100]}..."
                    )
                    action_builder.add_reasoning(message.content[0].text)
                else:
                    logger.debug(
                        f"Skipping message with None channel and no usable content"
                    )
            else:
                raise ValueError(f"Unknown channel: {message.channel}")
        return action_builder.build()


class ModalHarmonyAgentInferenceModel(AgentInferenceModel):
    """Inference model that uses Modal vLLM endpoint with native Harmony tokens in/out.

    This is similar to TinkerAgentInferenceModel but uses HTTP requests to a Modal
    vLLM deployment instead of the Tinker sampling client.
    """

    enc: HarmonyEncoding
    base_url: str
    model: str
    max_completion_tokens: int
    temperature: float
    top_p: float
    timeout_secs: int
    strict_mode: bool
    stop_token_ids: List[int]
    _session: requests.Session

    def __init__(
        self,
        base_url: str = "https://chroma-core--example-gpt-oss-inference-serve.modal.run",
        model: str = "llm",
        max_completion_tokens: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.9,
        timeout_secs: int = 500,
        strict_mode: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout_secs = timeout_secs
        self.strict_mode = strict_mode
        self.enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = list(self.enc.stop_tokens_for_assistant_actions())
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=15),
        before_sleep=lambda _: logger.warning("Retrying Modal vLLM sample..."),
    )
    def _sample(self, input_tokens: List[int], max_tokens: int) -> List[int]:
        """Send tokens to Modal vLLM endpoint and get response tokens back."""
        payload = {
            "model": self.model,
            "prompt": input_tokens,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False,
            "stop_token_ids": self.stop_token_ids,
            "return_token_ids": True,
        }

        resp = self._session.post(
            f"{self.base_url}/v1/completions",
            json=payload,
            timeout=self.timeout_secs,
        )

        if resp.status_code >= 400:
            raise RuntimeError(f"Modal vLLM error {resp.status_code}: {resp.text}")

        data = resp.json()
        return data["choices"][0].get("token_ids", [])

    def __call__(self, context: InferenceContext) -> Optional[Action]:
        trajectory = context.trajectory
        toolset = context.toolset
        max_tokens = context.max_tokens or self.max_completion_tokens

        # Convert trajectory to Harmony format
        request_messages = trajectory.to_provider_format(ProviderFormat.OPENAI_HARMONY)

        # Render to tokens
        input_tokens = self.enc.render_conversation(
            request_messages, config=RenderConversationConfig(auto_drop_analysis=False)
        )

        # Sample from Modal vLLM endpoint
        resp_tokens = self._sample(list(input_tokens), max_tokens)

        # Reuse TinkerAgentInferenceModel's parsing logic
        return TinkerAgentInferenceModel.harmony_tinker_tokens_to_action(
            self.enc, resp_tokens, toolset, self.strict_mode
        )


class Agent:
    """An agent that can take actions based on an observation and a toolset.

    An agent uses a inference model to determine the next action based on the current trajectory and toolset.

    It is designed to be a language model provider agnostic abstraction for experimenting with different agent behaviors.

    The agent is implemented as a state machine that can be driven manually or automatically:

    Manual driving:
        agent.reset()
        agent.observe(initial_observation)
        while not agent.is_done:
            action = agent.infer()
            if action is None:
                break
            observation = agent.act(action)
            if observation is not None:
                agent.observe(observation)
        trajectory = agent.trajectory

    Automatic driving:
        trajectory = agent(initial_observation)
    """

    toolset: ToolSet
    inference_model: AgentInferenceModel
    max_trajectory_length: int
    _trajectory_builder: Optional[TrajectoryBuilder]
    previous_response_id: Optional[
        str
    ]  # Only used for OpenAI Responses API, this is a bit messy but its one field so allowing it for now

    def __init__(
        self,
        toolset: ToolSet,
        inference_model: AgentInferenceModel,
        max_trajectory_length: int = 32,
    ):
        self.toolset = toolset
        self.inference_model = inference_model
        self.max_trajectory_length = max_trajectory_length
        self._trajectory_builder = TrajectoryBuilder()
        self.previous_response_id = None

    def reset(self) -> None:
        """Reset the agent state for a new run.

        Subclasses should override this to reset additional state.
        """
        self._trajectory_builder = TrajectoryBuilder()
        self.previous_response_id = None

    @property
    def trajectory(self) -> Trajectory:
        """Get the current trajectory.

        Raises RuntimeError if agent has not been initialized.
        """
        if self._trajectory_builder is None:
            raise RuntimeError(
                "Agent not initialized. Call reset() or observe() first."
            )
        return self._trajectory_builder.build()

    def prepare_for_inference(self) -> InferenceContext:
        """Prepare for inference, returning trajectory and constraints.

        Override in subclasses to preprocess trajectory or add constraints
        like limited toolset or max_tokens.

        Returns:
            InferenceContext with trajectory and any inference constraints.
        """
        return InferenceContext(
            trajectory=self.trajectory,
            toolset=self.toolset,
            previous_response_id=self.previous_response_id,
        )

    @property
    def is_done(self) -> bool:
        """Check if the agent has completed (no more actions to take).

        Returns True if the last entry in the trajectory is a terminal action
        (i.e., an action with only text tool calls).
        """
        if self._trajectory_builder is None:
            return False
        traj = self._trajectory_builder.build()
        if not traj.actions_and_observations:
            return False
        last = traj.actions_and_observations[-1]
        if isinstance(last, Action):
            non_text = [t for t in last.tools if not isinstance(t, UserTextTool)]
            return len(non_text) == 0
        return False

    def observe(self, observation: Observation) -> None:
        """Add an observation to the trajectory.

        If the agent has not been initialized, this will also initialize it.
        """
        if self._trajectory_builder is None:
            raise RuntimeError("Agent not initialized")
        self._trajectory_builder.add_observation(observation)

    def infer(self) -> Optional[Action]:
        """Run inference on current trajectory and return the next action.

        Returns:
            The action to take, or None if inference produced no action.

        Raises:
            RuntimeError: If agent not initialized or max trajectory length exceeded.
        """
        if self._trajectory_builder is None:
            raise RuntimeError("Agent not initialized. Call observe() first.")
        if len(self._trajectory_builder) >= self.max_trajectory_length:
            raise RuntimeError("Agent exceeded maximum trajectory length")
        context = self.prepare_for_inference()
        action = self.inference_model(context)
        # Propagate response ID back from context
        self.previous_response_id = context.previous_response_id
        return action

    def act(self, action: Action) -> Optional[Observation]:
        """Record the action and execute any tool calls.

        The action is added to the trajectory, then any non-text tool calls
        are executed.

        Returns:
            The observation from tool execution, or None if no tools to execute
            (i.e., the action only contains text tool calls).

        Note: The observation is NOT automatically added to the trajectory.
        Call observe() to add it.
        """
        if self._trajectory_builder is None:
            raise RuntimeError("Agent not initialized. Call observe() first.")
        self._trajectory_builder.add_action(action)

        non_text_tool_calls = [
            tool for tool in action.tools if not isinstance(tool, UserTextTool)
        ]
        if len(non_text_tool_calls) == 0:
            return None

        observation_builder = ObservationBuilder()
        for tool, params, source in zip(action.tools, action.params, action.sources):
            if isinstance(tool, UserTextTool):
                continue
            tool_output, tool_metadata = self._call_tool(tool, params)
            observation_builder.add_observation(
                observation=tool_output,
                source=source,
                tool_metadata=tool_metadata,
            )
        return observation_builder.build()

    def __call__(self, initial_observation: Observation) -> Trajectory:
        """Auto-drive the agent from initial observation to completion."""
        self.reset()
        self.observe(initial_observation)
        while not self.is_done:
            action = self.infer()
            if action is None:
                break
            observation = self.act(action)
            if observation is not None:
                self.observe(observation)
        return self.trajectory

    def _call_tool(
        self,
        tool: Tool,
        params: Dict[Any, Any],
        overrides: Optional[Dict[Any, Any]] = None,
    ) -> Tuple[str, Optional[ToolCallMetadata]]:
        return tool(params, overrides)


# ============================================================================
# Agent Implementations
# ============================================================================


class DeduplicatingPruningSearchAgent(Agent):
    """An agent that deduplicates search results across the trajectory and prunes the chunks that are not relevant to the main question."""

    _ids_seen: Set[str]
    _doc_id_to_query: Dict[str, str]  # Maps doc_id to the query that first found it

    def __init__(
        self,
        toolset: ToolSet,
        inference_model: AgentInferenceModel,
        max_trajectory_length: int = 64,
    ) -> None:
        super().__init__(toolset, inference_model, max_trajectory_length)
        self._ids_seen = set()
        self._doc_id_to_query = {}

    def reset(self) -> None:
        """Reset the agent state including deduplication tracking."""
        super().reset()
        self._ids_seen = set()
        self._doc_id_to_query = {}

    def _call_tool(
        self,
        tool: Tool,
        params: Dict[Any, Any],
        overrides: Optional[Dict[Any, Any]] = None,
    ) -> Tuple[str, Optional[ToolCallMetadata]]:
        # TODO: support grep deduplication
        if isinstance(tool, SearchTool):
            query = params.get("query", "")
            overrides = overrides or {}
            overrides["ignore_ids"] = list(self._ids_seen)
            tool_output, tool_metadata = super()._call_tool(tool, params, overrides)
            if tool_metadata is not None and isinstance(
                tool_metadata, SearchToolCallMetadata
            ):
                self._ids_seen.update(tool_metadata.returned_chunk_ids)
                # Track which query found each doc_id (first query wins)
                for chunk_id in tool_metadata.returned_chunk_ids:
                    doc_id = chunk_id.split("_")[0] if "_" in chunk_id else chunk_id
                    if doc_id not in self._doc_id_to_query:
                        self._doc_id_to_query[doc_id] = query
            return tool_output, tool_metadata
        if isinstance(tool, ReadTool):
            doc_id = params.get("doc_id") or params.get("id", "")
            # Extract doc_id if it includes chunk suffix
            if "_" in doc_id:
                doc_id = doc_id.split("_")[0]
            # Pass the query that found this document for reranking
            if doc_id in self._doc_id_to_query:
                overrides = overrides or {}
                overrides["query"] = self._doc_id_to_query[doc_id]
            return super()._call_tool(tool, params, overrides)
        if isinstance(tool, PruneChunksTool):
            tool_output, tool_metadata = super()._call_tool(tool, params, overrides)
            return tool_output, tool_metadata
        else:
            return super()._call_tool(tool, params, overrides)

    def prepare_for_inference(self) -> InferenceContext:
        """Prune chunks from trajectory before inference."""
        return InferenceContext(
            trajectory=prune_chunks_from_trajectory(self.trajectory),
            toolset=self.toolset,
            previous_response_id=self.previous_response_id,
        )


class TokenBudgetRetrievalSubagent(DeduplicatingPruningSearchAgent):
    """A retrieval subagent that has a token budget and when within threshold it is given the option to prune
    chunks or conclude its search.
    """

    token_budget: int
    tool_output_budget: int  # Maximum tokens for tool output when near budget limit
    _step_tokens_used: (
        int  # Tracks tokens used by tool calls within the current act() step
    )

    def __init__(
        self,
        toolset: ToolSet,
        inference_model: AgentInferenceModel,
        token_counter: Callable[[Trajectory], int],
        text_token_counter: Optional[Callable[[str], int]] = None,
        max_trajectory_length: int = 128,
        threshold_budget: int = 16384,  # Where user message prompts prune/conclude
        # HACK: There is some minor discrepancy between our token count and what tinker reports, so we are leaving a buffer from 32768
        token_budget: int = 32268,  # 32k tokens is the max input length for gpt oss 20b on tinker (no support for yarn scaling)
        tool_output_budget: int = 4096,  # Default max tokens for tool output when budget is tight
        spillage_fraction: float = 0.5,  # Fraction of (token_budget - threshold_budget) allowed as spillage before hard rejection
    ) -> None:
        self.token_counter = token_counter
        self.text_token_counter = text_token_counter
        self.token_budget = token_budget
        self.threshold_budget = threshold_budget
        self.tool_output_budget = tool_output_budget
        # Calculate rejection budget: threshold + spillage allowance
        spillage_allowance = int((token_budget - threshold_budget) * spillage_fraction)
        self.rejection_budget = threshold_budget + spillage_allowance
        self._step_tokens_used = 0
        logger.info(
            "TokenBudgetAgent initialized",
            threshold_budget=threshold_budget,
            rejection_budget=self.rejection_budget,
            token_budget=token_budget,
            spillage_fraction=spillage_fraction,
        )
        super().__init__(toolset, inference_model, max_trajectory_length)

    def reset(self) -> None:
        """Reset the agent state including step token tracking."""
        super().reset()
        self._step_tokens_used = 0

    def act(self, action: Action) -> Optional[Observation]:
        """Execute tool calls with budget tracking across parallel calls in the same step."""
        # Reset step token counter at the start of each act() call
        self._step_tokens_used = 0
        return super().act(action)

    def observe(self, observation: Observation) -> None:
        """Add an observation to the trajectory with token usage status appended.

        This persists the token usage in the trajectory for training data.
        """
        if self._trajectory_builder is None:
            raise RuntimeError("Agent not initialized")

        # Calculate token count on the pruned trajectory after adding this observation
        # We need to temporarily add it to calculate, then modify
        self._trajectory_builder.add_observation(observation)
        pruned_trajectory = prune_chunks_from_trajectory(self.trajectory)
        current_token_usage = self.token_counter(pruned_trajectory)

        # Modify the last observation in the trajectory to include token status
        # Just show usage here - the user message in prepare_for_inference handles over-budget prompting
        token_status = (
            f"\n\n[Token usage: {current_token_usage}/{self.threshold_budget}]"
        )
        last_item = self._trajectory_builder.trajectory.actions_and_observations[-1]
        if isinstance(last_item, Observation) and last_item.observations:
            # Modify in place - append to the last observation string
            last_item.observations[-1] = last_item.observations[-1] + token_status

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a string."""
        if self.text_token_counter is not None:
            return self.text_token_counter(text)
        # Fallback: rough estimate of ~4 characters per token
        return len(text) // 4

    def _call_tool(
        self,
        tool: Tool,
        params: Dict[Any, Any],
        overrides: Optional[Dict[Any, Any]] = None,
    ) -> Tuple[str, Optional[ToolCallMetadata]]:
        """Call tool with budget-aware max_tokens override for SearchTool and ReadTool.

        If over budget, errors on any tool that isn't PruneChunksTool.
        """
        # Check if we're over budget - only allow PruneChunksTool
        pruned_trajectory = prune_chunks_from_trajectory(self.trajectory)
        current_token_usage = self.token_counter(pruned_trajectory)

        # Include step tokens in threshold check to catch parallel tool calls
        # Use rejection_budget (threshold + spillage) for hard rejection, not threshold_budget
        # This allows some spillage past threshold before hard-stopping tool calls
        effective_token_usage = current_token_usage + self._step_tokens_used
        if effective_token_usage > self.rejection_budget and not isinstance(
            tool, PruneChunksTool
        ):
            logger.warning(
                "Tool call rejected - over rejection budget",
                tool=tool.tool_schema.name,
                current_token_usage=current_token_usage,
                step_tokens_used=self._step_tokens_used,
                effective_token_usage=effective_token_usage,
                rejection_budget=self.rejection_budget,
            )
            return (
                f"Error: Token budget exceeded ({effective_token_usage}/{self.threshold_budget} tokens). "
                "You must use prune_chunks to reduce context size or provide your final answer.",
                None,
            )

        # Calculate remaining budget for tool output
        if isinstance(tool, (SearchTool, ReadTool)):
            remaining_budget = (
                self.token_budget - current_token_usage - self._step_tokens_used
            )

            if remaining_budget < self.tool_output_budget:
                max_tokens_for_tool = max(512, remaining_budget // 2)
                logger.warning(
                    "Constraining tool output due to low budget",
                    tool=tool.tool_schema.name,
                    remaining_budget=remaining_budget,
                    step_tokens_used=self._step_tokens_used,
                    max_tokens_for_tool=max_tokens_for_tool,
                )
                overrides = overrides or {}
                overrides["max_tokens"] = max_tokens_for_tool

        tool_output, tool_metadata = super()._call_tool(tool, params, overrides)

        # Track tokens used by this tool call for subsequent calls in the same step
        self._step_tokens_used += self._estimate_tokens(tool_output)

        return tool_output, tool_metadata

    def prepare_for_inference(self) -> InferenceContext:
        """Prepare for inference with token budget constraints.

        When over threshold_budget, adds a user message prompting the model to prune
        or conclude, and limits the toolset to only PruneChunksTool.
        """
        # Get the pruned trajectory from parent
        base_context = super().prepare_for_inference()
        pruned_trajectory = base_context.trajectory

        current_token_usage = self.token_counter(pruned_trajectory)
        logger.info("Current token usage", current_token_usage=current_token_usage)

        if current_token_usage > self.token_budget:
            raise ValueError(
                f"Current token usage {current_token_usage} exceeds token budget {self.token_budget}, this is not currently supported."
            )

        # If over threshold budget, add user message prompting prune/conclude decision
        # and limit toolset to only PruneChunksTool
        if current_token_usage > self.threshold_budget:
            logger.warning(
                "Over threshold budget - adding user prompt for prune/conclude",
                current_token_usage=current_token_usage,
                threshold_budget=self.threshold_budget,
            )
            # Build a new trajectory with the budget exhausted message as a user observation
            trajectory_builder = TrajectoryBuilder()
            for item in pruned_trajectory.actions_and_observations:
                if isinstance(item, Action):
                    trajectory_builder.add_action(item)
                elif isinstance(item, Observation):
                    trajectory_builder.add_observation(item)

            trajectory_builder.add_observation(
                Observation(
                    observations=[
                        get_retrieval_subagent_budget_exhausted_message(
                            current_token_usage, self.threshold_budget
                        )
                    ],
                    sources=["user"],
                    tool_metadata=[None],
                )
            )
            limited_trajectory = trajectory_builder.build()
            return InferenceContext(
                trajectory=limited_trajectory,
                toolset=self.toolset,
                max_tokens=self.token_budget - current_token_usage,
                previous_response_id=base_context.previous_response_id,
                skip_response_id_update=True,
            )

        return InferenceContext(
            trajectory=pruned_trajectory,
            toolset=self.toolset,
            max_tokens=self.token_budget - current_token_usage,
            previous_response_id=base_context.previous_response_id,
        )


# ============================================================================
# Parsing utilities
# ============================================================================


def prune_chunks_from_trajectory(trajectory: Trajectory) -> Trajectory:
    """Prune the chunks from the trajectory.

    Returns a new trajectory with the specified chunks removed, leaving the input unchanged.
    """
    trajectory = trajectory.clone()

    chunk_ids = set()
    for action in trajectory.actions_and_observations:
        if isinstance(action, Action):
            for tool, params, source in zip(
                action.tools, action.params, action.sources
            ):
                if isinstance(tool, PruneChunksTool):
                    if "chunk_ids" not in params:
                        logger.warning(
                            "PruneChunksTool called without chunk_ids", params=params
                        )
                    chunk_ids.update(params["chunk_ids"])

    def _remove_chunks_from_text(text: str) -> str:
        if not text:
            return text
        matches = list(DOC_ID_PATTERN.finditer(text))
        if not matches:
            return text

        # Find token status marker so we don't prune past it
        token_status_match = re.search(r"\n\n\[Token usage:", text)
        text_end = token_status_match.start() if token_status_match else len(text)

        remove_ranges: List[Tuple[int, int]] = []
        for idx, match in enumerate(matches):
            doc_id = match.group("chunk_id")
            if doc_id in chunk_ids:
                start = match.start()
                end = matches[idx + 1].start() if idx + 1 < len(matches) else text_end
                remove_ranges.append((start, end))

        if not remove_ranges:
            return text

        pruned_parts: List[str] = []
        last_idx = 0
        for start, end in remove_ranges:
            pruned_parts.append(text[last_idx:start])
            last_idx = end
        pruned_parts.append(text[last_idx:])

        pruned_text = "".join(pruned_parts)
        pruned_text = re.sub(r"\n{3,}", "\n\n", pruned_text)
        return pruned_text.strip()

    for action in trajectory.actions_and_observations:
        if isinstance(action, Observation):
            for tool_metadata in action.tool_metadata:
                if (
                    tool_metadata is not None
                    and isinstance(tool_metadata, SearchToolCallMetadata)
                    or isinstance(tool_metadata, GrepCorpusToolCallMetadata)
                ):
                    # Regex for all document windows matching our chunk ids and prune them
                    action.observations = [
                        _remove_chunks_from_text(observation)
                        for observation in action.observations
                    ]
    return trajectory


if __name__ == "__main__":
    # Example Run

    from harness_search.tools import ToolSet, SearchTool, GrepCorpusTool
    from harness_search.config import get_config
    from harness_search.prompts import get_retrieval_subagent_prompt
    import tiktoken

    COLLECTION_NAME = "browsecompplus_openai_11"

    config = get_config()
    chroma_client = config.get_chroma_client()
    openai_client = config.get_openai_client()
    anthropic_client = config.get_anthropic_client()
    tinker_client = config.get_tinker_service_client()
    # tinker_sampling_client = tinker_client.create_sampling_client(
    #     base_model="Qwen/Qwen3.5-27B"
    # )
    tinker_sampling_client = tinker_client.create_sampling_client(
        model_path=os.environ["INTENT_CURATOR_TINKER_CHECKPOINT"]
    )

    dataset = get_dataset("browsecompplus")

    toolset = ToolSet()

    # Simple token counter just to demonstrate the concept, not accurate token for all models of course
    # TODO: use a better token counter
    tiktoken_encoding = tiktoken.get_encoding("o200k_harmony")
    rerank_token_counter = lambda text: len(tiktoken_encoding.encode(text))

    reranker = VLLMQwen3Reranker(
        token_counter=rerank_token_counter,
    )
    toolset.add_tool(
        SearchTool(
            chroma_client=chroma_client,
            openai_client=openai_client,
            chroma_collection_name=COLLECTION_NAME,
            reranker=reranker,
        )
    )
    toolset.add_tool(
        GrepCorpusTool(
            chroma_client=chroma_client,
            chroma_collection_name=COLLECTION_NAME,
            token_counter=rerank_token_counter,
        )
    )
    toolset.add_tool(
        ReadTool(
            chroma_client=chroma_client,
            chroma_collection_name=COLLECTION_NAME,
            reranker=reranker,
            token_counter=rerank_token_counter,
        )
    )
    toolset.add_tool(PruneChunksTool())

    inference_model = AnthropicAgentInferenceModel(anthropic_client)
    # inference_model = OpenAIAgentInferenceModel(openai_client)
    # inference_model = TinkerAgentInferenceModel(
    #     tinker_sampling_client, strict_mode=False
    # )
    # inference_model = ModalHarmonyAgentInferenceModel()
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    token_counter: Callable[[Trajectory], int] = lambda trajectory: len(
        enc.render_conversation(
            trajectory.to_openai_harmony_format(),
            config=RenderConversationConfig(auto_drop_analysis=False),
        )
    )
    agent = TokenBudgetRetrievalSubagent(
        toolset, inference_model, token_counter, text_token_counter=rerank_token_counter
    )
    query_id, query_text = dataset.get_query_by_id("422")
    # query_id, query_text = dataset.get_random_query()
    logger.info("query", query_id=query_id, query_text=query_text)
    initial_observation = Observation(
        observations=[get_retrieval_subagent_prompt(query_text)],
        sources=["user"],
        tool_metadata=[None],
    )

    start_time = time.perf_counter()
    trajectory = agent(initial_observation=initial_observation)
    elapsed_time = time.perf_counter() - start_time
    logger.info(
        "query_completed", query_id=query_id, elapsed_seconds=round(elapsed_time, 2)
    )

    # Save pydantic model to json
    with open("example_trajectory.json", "w") as f:
        # indent 2
        f.write(json.dumps(trajectory.model_dump(mode="json"), indent=2))

    # Save openai harmony conversation to text
    openai_harmony_conversation = trajectory.to_openai_harmony_format()
    tokens = enc.render_conversation(
        openai_harmony_conversation,
        config=RenderConversationConfig(auto_drop_analysis=False),
    )
    logger.info("trajectory_tokens", tokens=len(tokens))
    as_text = enc.decode_utf8(tokens)
    with open("example_openai_harmony_conversation.txt", "w") as f:
        f.write(as_text)

    search_task_output = SearchTaskOutput(
        trajectory=trajectory, query_id=query_id, dataset_name=dataset.name
    )
    search_task_output.log_trajectory_stats()

    # Evaluate using the standard evaluation output
    eval_output = SearchTaskEvaluationOutput.from_search_task_output(
        search_task_output, dataset
    )
    logger.info(
        "evaluation_output",
        query_id=query_id,
        recall=eval_output.recall,
        precision=eval_output.precision,
        f1=eval_output.f1,
        trajectory_recall=eval_output.trajectory_recall,
        final_answer_recall=eval_output.final_answer_recall,
        num_turns=eval_output.num_turns,
        prune_accuracy=eval_output.prune_accuracy,
        rerank_recall=eval_output.rerank_recall,
        rerank_dropped_relevant_count=eval_output.rerank_dropped_relevant_count,
    )

    # Log answer and expected document ids
    logger.info(
        "retrieval_subagent_answer",
        query_id=query_id,
        answer=dataset.get_expected_answer(query_id),
        expected_document_ids=dataset.get_expected_document_ids(query_id),
        retrieved_document_ids=eval_output.retrieved_document_ids,
    )
