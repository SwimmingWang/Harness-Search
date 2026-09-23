import copy
from datetime import datetime
import json
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)
import uuid

from pydantic import BaseModel, SerializeAsAny, model_validator
import structlog
from harness_search.utils import ProviderFormat
from harness_search.tools import (
    GREP_CORPUS_SCHEMA,
    MULTI_TOOL_USE_SCHEMA,
    PRUNE_CHUNKS_SCHEMA,
    READ_SCHEMA,
    SEARCH_SCHEMA,
    MultiToolUseTool,
    SerializedTool,
    Tool,
    ToolCallMetadata,
    ToolSet,
    UserTextTool,
)
from openai_harmony import (
    Author,
    SystemContent,
    ToolDescription,
    Role,
    Message,
    Conversation,
    DeveloperContent,
    ReasoningEffort,
)

if TYPE_CHECKING:
    from harness_search.config import Config

logger = structlog.get_logger("search_agent.trajectory")

Source = Union[
    str, Literal["user"], Literal["agent"]
]  # Tool call id source, or user source, or agent source (for sending text to the user)


class Action(BaseModel):
    """An action that the agent can take. Along with optionally any reasoning that was done to determine the action.

    An agent can take multiple actions in a single step, each with a different tool and parameters.

    """

    tools: List[Tool]
    params: List[dict]
    sources: List[Source]
    reasoning: Optional[str] = None
    reasoning_signature: Optional[str] = (
        None  # Only used for Anthropic, needed to carry through reasoning to the next step
    )
    memory: Optional[List[Dict[str, Any]]] = None
    """Snapshot of append-only memory H_t at this action: list of {"step": int, "queries": List[str], "reflection": str}."""

    def as_iter(self) -> Iterator[Tuple[Tool, dict, Source]]:
        return iter(zip(self.tools, self.params, self.sources))

    @model_validator(mode="before")
    @classmethod
    def _deserialize_tools(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        tools = data.get("tools")
        if not tools:
            return data
        deserialized_tools: List[Tool] = []
        for tool_entry in tools:
            # Already a concrete tool instance (runtime usage)
            if isinstance(tool_entry, Tool):
                deserialized_tools.append(tool_entry)
                continue
            # pydantic may have already validated into Tool subclasses
            if isinstance(tool_entry, dict):
                tool_schema_data = tool_entry.get("tool_schema")
                if tool_schema_data is None:
                    raise ValueError(
                        "Serialized tool entry missing 'tool_schema' field"
                    )
                tool_name = tool_schema_data.get("name")
                if tool_name == "user_text":
                    deserialized_tools.append(UserTextTool())
                else:
                    deserialized_tools.append(
                        SerializedTool(tool_schema=tool_schema_data)
                    )
                continue
            deserialized_tools.append(tool_entry)
        data = data.copy()
        data["tools"] = deserialized_tools
        return data


class ActionBuilder:
    """A builder for actions."""

    def __init__(self):
        self.action = Action(
            tools=[],
            params=[],
            sources=[],
            reasoning=None,
            memory=None,
        )

    def add_tool_call(
        self, tool: Tool, params: dict, source: Source
    ) -> "ActionBuilder":
        # TODO: Automatically decompose multi-tool use tool calls into multiple tool calls
        if isinstance(tool, MultiToolUseTool):
            raise ValueError(
                "MultiToolUseTool should not be added to an action builder"
            )
        self.action.tools.append(tool)
        self.action.params.append(params)
        self.action.sources.append(source)
        return self

    def add_reasoning(
        self, reasoning: str, signature: Optional[str] = None
    ) -> "ActionBuilder":
        if self.action.reasoning is not None:
            raise ValueError("Reasoning already added for this action")
        self.action.reasoning = reasoning
        self.action.reasoning_signature = signature
        return self

    def add_memory(
        self, memory: Optional[List[Dict[str, Any]]]
    ) -> "ActionBuilder":
        """Add memory snapshot to this action.

        Args:
            memory: List of memory entries, each with {"step": int, "queries": List[str], "reflection": str}
        """
        self.action.memory = memory
        return self

    def is_complete(self) -> bool:
        # It is possible to have just a reasoning step with no tool calls in some cases
        return (
            len(self.action.tools) > 0
            and len(self.action.params) > 0
            and len(self.action.sources) > 0
            and len(self.action.tools)
            == len(self.action.params)
            == len(self.action.sources)
        ) or (self.action.reasoning is not None)

    def build(self) -> Action:
        if not self.is_complete():
            raise ValueError(
                f"Action builder is not complete, missing tool calls or parameters or sources: {self.action}"
            )
        return self.action


class Observation(BaseModel):
    """An observation that the agent can make.

    Each observation is a string, but the agent can make multiple observations in a single step since it can use multiple tools in a single step.

    """

    observations: List[str]
    sources: List[Source]
    tool_metadata: List[Optional[SerializeAsAny[ToolCallMetadata]]]


class ObservationBuilder:
    """A builder for observations."""

    observations: List[str]
    sources: List[Source]
    tool_metadata: List[Optional[ToolCallMetadata]]

    def __init__(self):
        self.observations = []
        self.sources = []
        self.tool_metadata = []

    def add_observation(
        self,
        observation: str,
        source: Source,
        tool_metadata: Optional[ToolCallMetadata] = None,
    ) -> "ObservationBuilder":
        self.observations.append(observation)
        self.sources.append(source)
        self.tool_metadata.append(tool_metadata)
        return self

    def is_complete(self) -> bool:
        return (
            len(self.observations) > 0
            and len(self.tool_metadata) == len(self.observations)
            and len(self.sources) == len(self.observations)
        )

    def build(self) -> Observation:
        if not self.is_complete():
            raise ValueError(
                "Observation builder is not complete, missing observations, tool metadata, or sources"
            )
        return Observation(
            observations=self.observations,
            sources=self.sources,
            tool_metadata=self.tool_metadata,
        )


class Trajectory(BaseModel):
    """A sequence of actions and observations that the agent takes to solve a task.

    For example, for a GatheringSearchAgent the actions are tool calls and the observations are the results of the tool calls.
    """

    actions_and_observations: List[Action | Observation]
    id: uuid.UUID

    @property
    def num_turns(self) -> int:
        """Return the number of turns (actions) in the trajectory."""
        return sum(
            1 for entry in self.actions_and_observations if isinstance(entry, Action)
        )

    def clone(self) -> "Trajectory":
        """Create a deep copy of the trajectory.

        This is useful when you want to mutate a trajectory without affecting the original,
        e.g., for pruning chunks while preserving the unpruned version.

        Note: We manually copy rather than using model_copy(deep=True) because Tool objects
        contain clients (like PerformanceClient) that cannot be pickled/deep copied.
        We use model_construct to bypass validation since we're copying already-validated data.
        """
        cloned_entries: List[Action | Observation] = []

        for entry in self.actions_and_observations:
            if isinstance(entry, Action):
                # Deep copy params (dicts that could be mutated), but keep tool references
                # Use model_construct to bypass validation - data is already validated
                cloned_entries.append(
                    Action.model_construct(
                        tools=list(entry.tools),  # Shallow copy - tools are stateless
                        params=copy.deepcopy(entry.params),
                        sources=list(entry.sources),
                        reasoning=entry.reasoning,
                        reasoning_signature=entry.reasoning_signature,
                    )
                )
            elif isinstance(entry, Observation):
                cloned_entries.append(
                    Observation.model_construct(
                        observations=list(entry.observations),
                        sources=list(entry.sources),
                        tool_metadata=list(
                            entry.tool_metadata
                        ),  # Shallow copy - metadata is read-only
                    )
                )

        return Trajectory.model_construct(
            actions_and_observations=cloned_entries,
            id=self.id,
        )

    def __repr__(self) -> str:
        out = "Trajectory:\n"
        for i, action_or_observation in enumerate(self.actions_and_observations):
            if isinstance(action_or_observation, Action):
                out += f"[Step {i}] [Action] {repr(action_or_observation.tools)} with params {action_or_observation.params}\n"
            else:
                out += f"[Step {i}] [Observation] {[obs[:100] for obs in action_or_observation.observations]}...\n"
            out += "\n"
        return out

    def to_provider_format(self, provider: ProviderFormat) -> Any:
        if provider == ProviderFormat.ANTHROPIC:
            return self.to_anthropic_format()
        elif provider == ProviderFormat.OPENAI_HARMONY:
            return self.to_openai_harmony_format()
        elif provider == ProviderFormat.OPENAI:
            return self.to_openai_format()
        elif provider == ProviderFormat.OPENAI_RESPONSES:
            return self.to_openai_responses_input()
        else:
            raise ValueError(f"Unsupported provider format: {provider}")

    @classmethod
    def deserialize(
        cls,
        data: Union[str, Dict[str, Any]],
        *,
        config: "Config",
        # TODO: move chroma_collection_name to config
        chroma_collection_name: str,
        toolset: Optional[ToolSet] = None,
    ) -> "Trajectory":
        """
        Deserialize a trajectory from serialized data and hydrate tool placeholders.

        Args:
            data: Serialized trajectory (JSON string or already-parsed dict).
            config: Runtime configuration used to construct concrete tool instances.
            chroma_collection_name: Chroma collection backing retrieval tools.
            toolset: Optional pre-built toolset. If omitted, one is derived from config.
        """

        if isinstance(data, str):
            trajectory = cls.model_validate_json(data)
        else:
            trajectory = cls.model_validate(data)

        resolved_toolset = toolset or ToolSet.from_config(
            config,
            chroma_collection_name=chroma_collection_name,
        )

        for entry in trajectory.actions_and_observations:
            if not isinstance(entry, Action):
                continue
            hydrated_tools: List[Tool] = []
            for tool in entry.tools:
                if isinstance(tool, SerializedTool):
                    concrete_tool = resolved_toolset.get_tool(tool.tool_schema.name)
                    if concrete_tool is None:
                        raise ValueError(
                            f"Tool '{tool.tool_schema.name}' is not available in the constructed toolset."
                        )
                    hydrated_tools.append(concrete_tool)
                else:
                    hydrated_tools.append(tool)
            entry.tools = hydrated_tools

        return trajectory

    def to_openai_harmony_format(self) -> Conversation:
        system_message = (
            SystemContent.new()
            .with_reasoning_effort(ReasoningEffort.HIGH)
            .with_conversation_start_date(datetime.now().strftime("%Y-%m-%d"))
        )
        messages = [Message.from_role_and_content(Role.SYSTEM, system_message)]

        def format_parameters(
            parameters: Dict[str, Any], required: List[str]
        ) -> Dict[str, Any]:
            return {
                "type": "object",
                "properties": parameters,
                "required": required,
            }

        # Assume ReadDocument, Grep, SearchCorpus PruneChunks, and MultiToolUseTool are all available
        developer_message = DeveloperContent.new().with_function_tools(
            [
                ToolDescription.new(
                    SEARCH_SCHEMA.name,
                    SEARCH_SCHEMA.description,
                    format_parameters(
                        SEARCH_SCHEMA.parameters, SEARCH_SCHEMA.required
                    ),
                ),
                ToolDescription.new(
                    GREP_CORPUS_SCHEMA.name,
                    GREP_CORPUS_SCHEMA.description,
                    format_parameters(
                        GREP_CORPUS_SCHEMA.parameters, GREP_CORPUS_SCHEMA.required
                    ),
                ),
                ToolDescription.new(
                    READ_SCHEMA.name,
                    READ_SCHEMA.description,
                    format_parameters(
                        READ_SCHEMA.parameters, READ_SCHEMA.required
                    ),
                ),
                ToolDescription.new(
                    MULTI_TOOL_USE_SCHEMA.name,
                    MULTI_TOOL_USE_SCHEMA.description,
                    format_parameters(
                        MULTI_TOOL_USE_SCHEMA.parameters, MULTI_TOOL_USE_SCHEMA.required
                    ),
                ),
                ToolDescription.new(
                    PRUNE_CHUNKS_SCHEMA.name,
                    PRUNE_CHUNKS_SCHEMA.description,
                    format_parameters(
                        PRUNE_CHUNKS_SCHEMA.parameters, PRUNE_CHUNKS_SCHEMA.required
                    ),
                ),
            ]
        )
        messages.append(
            Message.from_role_and_content(Role.DEVELOPER, developer_message)
        )
        tool_use_source_to_tool_name: Dict[str, str] = {}
        for action_or_observation in self.actions_and_observations:
            # ================================================================
            # Action Handling
            # ================================================================
            if isinstance(action_or_observation, Action):
                action = action_or_observation
                if action.reasoning:
                    messages.append(
                        Message.from_role_and_content(
                            Role.ASSISTANT, action.reasoning
                        ).with_channel("analysis")
                    )
                if len(action.tools) > 1:
                    # GPT-OSS 20B was not natively trained with multiple tool calls, so we turn it into a single multi-tool use tool call
                    tool_calls = []
                    for tool, params, source in action.as_iter():
                        if isinstance(tool, UserTextTool):
                            messages.append(
                                Message.from_role_and_content(
                                    Role.ASSISTANT, params["text"]
                                ).with_channel("final")
                            )
                        else:
                            # TODO: confirm this is all you need by schema
                            tool_calls.append(
                                {
                                    "tool_name": tool.tool_schema.name,
                                    "parameters": params,
                                }
                            )
                            tool_use_source_to_tool_name[source] = tool.tool_schema.name
                    # Append the multi-tool use tool call to the messages
                    messages.append(
                        Message.from_role_and_content(
                            Role.ASSISTANT,
                            json.dumps(tool_calls),
                        )
                        .with_channel("commentary")
                        .with_recipient("functions.multi_tool_use")
                        .with_content_type("<|constrain|>json")
                    )
                elif len(action.tools) == 1:
                    tool = action.tools[0]
                    params = action.params[0]
                    source = action.sources[0]
                    if isinstance(tool, UserTextTool):
                        messages.append(
                            Message.from_role_and_content(
                                Role.ASSISTANT, params["text"]
                            ).with_channel("final")
                        )
                    else:
                        messages.append(
                            Message.from_role_and_content(
                                Role.ASSISTANT, json.dumps(params)
                            )
                            .with_channel("commentary")
                            .with_recipient("functions." + tool.tool_schema.name)
                            .with_content_type("<|constrain|>json")
                        )
                        tool_use_source_to_tool_name[source] = (
                            "functions." + tool.tool_schema.name
                        )
            # ================================================================
            # Observation Handling
            # ================================================================
            # TODO: it seems the observation name in harmony is sometimes the wrong tool name for multi tool use
            elif isinstance(action_or_observation, Observation):
                observation = action_or_observation
                if len(observation.observations) > 1:
                    # An observation can be from tools or user text, so we need to handle both cases
                    tool_results = []
                    for observation_text, source in zip(
                        observation.observations, observation.sources
                    ):
                        tool_use_tool_name = tool_use_source_to_tool_name[source]
                        if source == "user":
                            raise ValueError(
                                "User text observation found in a tool result node, this should not happen"
                            )
                        else:
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "name": tool_use_tool_name,
                                    "content": [observation_text],
                                }
                            )
                    messages.append(
                        Message.from_author_and_content(
                            Author(role=Role.TOOL, name="functions.multi_tool_use"),
                            json.dumps(tool_results),
                        )
                        .with_channel("commentary")
                        .with_recipient("assistant")
                    )
                else:
                    observation_source = observation.sources[0]
                    observation_text = observation.observations[0]
                    if observation_source == "user":
                        messages.append(
                            Message.from_role_and_content(Role.USER, observation_text)
                        )
                    else:
                        # Handle special cases for memory tools and other tools not in tool_use_source_to_tool_name
                        if observation_source in tool_use_source_to_tool_name:
                            observation_tool_name = tool_use_source_to_tool_name[
                                observation_source
                            ]
                        elif observation_source.startswith("toolu_read_memory"):
                            observation_tool_name = "read_memory"
                        elif observation_source.startswith("toolu_update_memory"):
                            observation_tool_name = "update_memory"
                        elif observation_source.startswith("toolu_backtrack"):
                            observation_tool_name = "backtrack"
                        else:
                            # Try to extract tool name from source pattern "toolu_<toolname>_..."
                            parts = observation_source.split("_")
                            if len(parts) >= 2 and parts[0] == "toolu":
                                observation_tool_name = parts[1]
                            else:
                                raise ValueError(
                                    f"Unknown observation source: {observation_source}"
                                )
                        messages.append(
                            Message.from_author_and_content(
                                Author(
                                    role=Role.TOOL,
                                    name="functions." + observation_tool_name,
                                ),
                                json.dumps(observation_text),
                            )
                            .with_channel("commentary")
                            .with_recipient("assistant")
                        )
            else:
                raise ValueError(
                    f"Unknown action or observation type: {type(action_or_observation)}"
                )
        return Conversation(messages=messages)

    def to_anthropic_format(self) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        for action_or_observation in self.actions_and_observations:
            content: List[Dict[str, Any]] = []
            if isinstance(action_or_observation, Action):
                out = {"role": "assistant", "content": content}
                if action_or_observation.reasoning:
                    content.append(
                        {
                            "type": "thinking",
                            "thinking": action_or_observation.reasoning,
                            "signature": action_or_observation.reasoning_signature,
                        }
                    )
                for tool, params, source in action_or_observation.as_iter():
                    if tool.tool_schema.name == "user_text":
                        content.append({"type": "text", "text": params["text"]})
                    else:
                        content.append(
                            {
                                "type": "tool_use",
                                "id": source,
                                "name": tool.tool_schema.name,
                                "input": params,
                            }
                        )
                messages.append(out)
            elif isinstance(action_or_observation, Observation):
                out = {"role": "user", "content": content}
                for observation_text, source in zip(
                    action_or_observation.observations, action_or_observation.sources
                ):
                    if source == "user":
                        content.append({"type": "text", "text": observation_text})
                    else:
                        content.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": source,
                                "content": [{"type": "text", "text": observation_text}],
                            }
                        )
                messages.append(out)
            else:
                # Defensive
                raise ValueError(
                    f"Unknown action or observation type: {type(action_or_observation)}"
                )
        return messages

    def _to_openai_like_format(self) -> List[Dict[str, Any]]:
        """Convert the trajectory into OpenAI-compatible message format."""

        def _make_text_content(text: str) -> Dict[str, str]:
            if text.strip() == "":
                logger.warning("Empty text content, maybe pruned?")
                return {"type": "text", "text": "Maybe pruned?"}
            return {"type": "text", "text": text}

        messages: List[Dict[str, Any]] = []
        for action_or_observation in self.actions_and_observations:
            if isinstance(action_or_observation, Action):
                action = action_or_observation
                assistant_message: Dict[str, Any] = {"role": "assistant"}
                content_items: List[Dict[str, Any]] = []
                tool_calls: List[Dict[str, Any]] = []
                for tool, params, source in action.as_iter():
                    if isinstance(tool, UserTextTool):
                        content_items.append(_make_text_content(params.get("text", "")))
                    else:
                        tool_calls.append(
                            {
                                "id": str(source),
                                "type": "function",
                                "function": {
                                    "name": tool.tool_schema.name,
                                    "arguments": json.dumps(params),
                                },
                            }
                        )
                assistant_message["content"] = content_items if content_items else ""
                if tool_calls:
                    assistant_message["tool_calls"] = tool_calls
                messages.append(assistant_message)
            elif isinstance(action_or_observation, Observation):
                observation = action_or_observation
                for observation_text, source in zip(
                    observation.observations, observation.sources
                ):
                    if source == "user":
                        messages.append(
                            {
                                "role": "user",
                                "content": [_make_text_content(observation_text)],
                            }
                        )
                    else:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": str(source),
                                "content": [_make_text_content(observation_text)],
                            }
                        )
            else:
                raise ValueError(
                    f"Unknown action or observation type: {type(action_or_observation)}"
                )
        return messages

    def to_openai_format(self) -> List[Dict[str, Any]]:
        """Convert the trajectory into OpenAI Chat Completions message format."""

        return self._to_openai_like_format()

    def to_openai_responses_input(self) -> List[Dict[str, Any]]:
        """Convert the entire trajectory into OpenAI Responses input format."""

        return self._to_openai_responses_items(self.actions_and_observations)

    def to_openai_responses_subset(
        self, entries: Iterable[Action | Observation]
    ) -> List[Dict[str, Any]]:
        """Convert a subset of trajectory entries into OpenAI Responses input format."""

        return self._to_openai_responses_items(entries)

    def _to_openai_responses_items(
        self, entries: Iterable[Action | Observation]
    ) -> List[Dict[str, Any]]:
        def _make_text_content(
            text: str, *, content_type: str = "input_text"
        ) -> Dict[str, Any]:
            if text.strip() == "":
                logger.warning("Empty text content, maybe pruned?")
                return {"type": "input_text", "text": "Maybe pruned?"}
            if content_type not in ("input_text", "output_text"):
                raise ValueError(f"Unsupported content type: {content_type}")
            return {"type": content_type, "text": text}

        input_items: List[Dict[str, Any]] = []
        for action_or_observation in entries:
            if isinstance(action_or_observation, Observation):
                for observation_text, source in zip(
                    action_or_observation.observations,
                    action_or_observation.sources,
                ):
                    if source == "user":
                        input_items.append(
                            {
                                "type": "message",
                                "role": "user",
                                "content": [_make_text_content(observation_text)],
                            }
                        )
                    else:
                        input_items.append(
                            {
                                "type": "function_call_output",
                                "call_id": str(source),
                                "output": observation_text,
                            }
                        )
            elif isinstance(action_or_observation, Action):
                for tool, params, source in action_or_observation.as_iter():
                    if isinstance(tool, UserTextTool):
                        input_items.append(
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    _make_text_content(
                                        params.get("text", ""),
                                        content_type="output_text",
                                    )
                                ],
                            }
                        )
                    else:
                        input_items.append(
                            {
                                "type": "function_call",
                                "call_id": str(source),
                                "name": tool.tool_schema.name,
                                "arguments": json.dumps(params),
                            }
                        )
            else:
                raise ValueError(
                    f"Unknown action or observation type: {type(action_or_observation)}"
                )
        return input_items


class TrajectoryBuilder:
    """A builder for trajectories."""

    def __init__(self):
        self.trajectory = Trajectory(
            actions_and_observations=[],
            id=uuid.uuid4(),
        )

    def add_action(self, action: Action) -> "TrajectoryBuilder":
        self.trajectory.actions_and_observations.append(action)
        return self

    def add_observation(self, observation: Observation) -> "TrajectoryBuilder":
        self.trajectory.actions_and_observations.append(observation)
        return self

    def __len__(self) -> int:
        return len(self.trajectory.actions_and_observations)

    def build(self) -> Trajectory:
        return self.trajectory
