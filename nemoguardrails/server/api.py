# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import contextvars
import importlib.util
import json
import logging
import os.path
import re
import time
import warnings
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Literal, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, root_validator, validator
from starlette.responses import StreamingResponse
from starlette.staticfiles import StaticFiles

from nemoguardrails import LLMRails, RailsConfig, utils
from nemoguardrails.rails.llm.options import (
    GenerationLog,
    GenerationLogOptions,
    GenerationOptions,
    GenerationRailsOptions,
    GenerationResponse,
)
from nemoguardrails.server.datastore.datastore import DataStore
from nemoguardrails.streaming import StreamingHandler

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class GuardrailsApp(FastAPI):
    """Custom FastAPI subclass with additional attributes for Guardrails server."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize custom attributes
        self.default_config_id: Optional[str] = None
        self.rails_config_path: str = ""
        self.disable_chat_ui: bool = False
        self.auto_reload: bool = False
        self.stop_signal: bool = False
        self.single_config_mode: bool = False
        self.single_config_id: Optional[str] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.task: Optional[asyncio.Future] = None


# The list of registered loggers. Can be used to send logs to various
# backends and storage engines.
registered_loggers: List[Callable] = []

api_description = """Guardrails Sever API."""

# The headers for each request
api_request_headers: contextvars.ContextVar = contextvars.ContextVar("headers")

# The datastore that the Server should use.
# This is currently used only for storing threads.
# TODO: refactor to wrap the FastAPI instance inside a RailsServer class
#  and get rid of all the global attributes.
datastore: Optional[DataStore] = None


@asynccontextmanager
async def lifespan(app: GuardrailsApp):
    # Startup logic here
    """Register any additional challenges, if available at startup."""
    challenges_files = os.path.join(app.rails_config_path, "challenges.json")

    if os.path.exists(challenges_files):
        with open(challenges_files) as f:
            register_challenges(json.load(f))

    # If there is a `config.yml` in the root `app.rails_config_path`, then
    # that means we are in single config mode.
    if os.path.exists(os.path.join(app.rails_config_path, "config.yml")) or os.path.exists(
        os.path.join(app.rails_config_path, "config.yaml")
    ):
        app.single_config_mode = True
        app.single_config_id = os.path.basename(app.rails_config_path)
    else:
        # If we're not in single-config mode, we check if we have a config.py for the
        # server configuration.
        filepath = os.path.join(app.rails_config_path, "config.py")
        if os.path.exists(filepath):
            filename = os.path.basename(filepath)
            spec = importlib.util.spec_from_file_location(filename, filepath)
            if spec is not None and spec.loader is not None:
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
            else:
                config_module = None

            # If there is an `init` function, we call it with the reference to the app.
            if config_module is not None and hasattr(config_module, "init"):
                config_module.init(app)

    # Finally, we register the static frontend UI serving

    if not app.disable_chat_ui:
        FRONTEND_DIR = utils.get_chat_ui_data_path("frontend")

        app.mount(
            "/",
            StaticFiles(
                directory=FRONTEND_DIR,
                html=True,
            ),
            name="chat",
        )
    else:

        @app.get("/")
        async def root_handler():
            return {"status": "ok"}

    if app.auto_reload:
        app.loop = asyncio.get_running_loop()
        # Store the future directly as task
        app.task = app.loop.run_in_executor(None, start_auto_reload_monitoring)

    yield

    # Shutdown logic here
    if app.auto_reload:
        app.stop_signal = True
        if hasattr(app, "task") and app.task is not None:
            app.task.cancel()
        log.info("Shutting down file observer")
    else:
        pass


app = GuardrailsApp(
    title="Guardrails Server API",
    description=api_description,
    version="0.1.0",
    license_info={"name": "Apache License, Version 2.0"},
    lifespan=lifespan,
)

ENABLE_CORS = os.getenv("NEMO_GUARDRAILS_SERVER_ENABLE_CORS", "false").lower() == "true"
ALLOWED_ORIGINS = os.getenv("NEMO_GUARDRAILS_SERVER_ALLOWED_ORIGINS", "*")

if ENABLE_CORS:
    # Split origins by comma
    origins = ALLOWED_ORIGINS.split(",")

    log.info(f"CORS enabled with the following origins: {origins}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.default_config_id = None

# By default, we use the rails in the examples folder
app.rails_config_path = utils.get_examples_data_path("bots")

# Weather the chat UI is enabled or not.
app.disable_chat_ui = False

# auto reload flag
app.auto_reload = False

# stop signal for observer
app.stop_signal = False

# Whether the server is pointed to a directory containing a single config.
app.single_config_mode = False
app.single_config_id = None


class RequestBody(BaseModel):
    config_id: Optional[str] = Field(
        default=os.getenv("DEFAULT_CONFIG_ID", None),
        description="The id of the configuration to be used. If not set, the default configuration will be used.",
    )
    config_ids: Optional[List[str]] = Field(
        default=None,
        description="The list of configuration ids to be used. If set, the configurations will be combined.",
        # alias="guardrails",
        validate_default=True,
    )
    thread_id: Optional[str] = Field(
        default=None,
        min_length=16,
        max_length=255,
        description="The id of an existing thread to which the messages should be added.",
    )
    messages: Optional[List[dict]] = Field(
        default=None, description="The list of messages in the current conversation."
    )
    context: Optional[dict] = Field(
        default=None,
        description="Additional context data to be added to the conversation.",
    )
    stream: Optional[bool] = Field(
        default=False,
        description="If set, partial message deltas will be sent, like in ChatGPT. "
        "Tokens will be sent as data-only server-sent events as they become "
        "available, with the stream terminated by a data: [DONE] message.",
    )
    options: GenerationOptions = Field(
        default_factory=GenerationOptions,
        description="Additional options for controlling the generation.",
    )
    state: Optional[dict] = Field(
        default=None,
        description="A state object that should be used to continue the interaction.",
    )

    @root_validator(pre=True)
    def ensure_config_id(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("config_id") is not None and data.get("config_ids") is not None:
                raise ValueError("Only one of config_id or config_ids should be specified")
            if data.get("config_id") is None and data.get("config_ids") is not None:
                data["config_id"] = None
            if data.get("config_id") is None and data.get("config_ids") is None:
                warnings.warn("No config_id or config_ids provided, using default config_id")
        return data

    @validator("config_ids", pre=True, always=True)
    def ensure_config_ids(cls, v, values):
        if v is None and values.get("config_id") and values.get("config_ids") is None:
            # populate config_ids with config_id if only config_id is provided
            return [values["config_id"]]
        return v


class ResponseBody(BaseModel):
    messages: Optional[List[dict]] = Field(default=None, description="The new messages in the conversation")
    llm_output: Optional[dict] = Field(
        default=None,
        description="Contains any additional output coming from the LLM.",
    )
    output_data: Optional[dict] = Field(
        default=None,
        description="The output data, i.e. a dict with the values corresponding to the `output_vars`.",
    )
    log: Optional[GenerationLog] = Field(default=None, description="Additional logging information.")
    state: Optional[dict] = Field(
        default=None,
        description="A state object that should be used to continue the interaction in the future.",
    )


class GuardrailCheckRequestBody(BaseModel):
    """Request body for the /v1/guardrail/checks endpoint.

    This endpoint validates messages against configured guardrails without generating
    new content. All guardrail parameters are optionally defined in the configuration - either
    referenced by config_id or provided inline.
    """

    model: str = Field(
        description="The model identifier (informational). "
        "The actual models used are defined in the guardrail configuration."
    )
    messages: List[dict] = Field(
        description="The list of messages to check against guardrails. "
        "Each message should have 'role' (user/assistant) and 'content' fields."
    )
    guardrails: Optional[dict] = Field(
        default=None,
        description="Guardrail configuration. Can contain either 'config_id' (string) to reference "
        "an existing server configuration, or 'config' (dict) with a complete inline guardrail configuration. "
        "If not provided, uses the server's default configuration.",
    )
    stream: Optional[bool] = Field(default=False, description="Whether to stream results as each message is checked.")
    use_conversation_context: Optional[bool] = Field(
        default=False,
        description="If true, checks all messages together with full conversation context. "
        "If false (default), checks each message independently.",
    )


class GuardrailCheckResponseBody(BaseModel):
    """Response body for the /v1/guardrail/checks endpoint."""

    status: Literal["success", "blocked", "error"] = Field(
        description="Overall status: 'success' if all rails passed, 'blocked' if any rail blocked, 'error' for system errors."
    )
    rails_status: Dict[str, dict] = Field(
        default_factory=dict, description="Status of each individual rail that was executed."
    )
    guardrails_data: Optional[dict] = Field(default=None, description="Additional data from guardrail execution.")


@app.get(
    "/v1/rails/configs",
    summary="Get List of available rails configurations.",
)
async def get_rails_configs():
    """Returns the list of available rails configurations."""

    # In single-config mode, we return a single config.
    if app.single_config_mode:
        # And we use the name of the root folder as the id of the config.
        return [{"id": app.single_config_id}]

    # We extract all folder names as config names
    config_ids = [
        f
        for f in os.listdir(app.rails_config_path)
        if os.path.isdir(os.path.join(app.rails_config_path, f))
        and f[0] != "."
        and f[0] != "_"
        # We filter out all the configs for which there is no `config.yml` file.
        and (
            os.path.exists(os.path.join(app.rails_config_path, f, "config.yml"))
            or os.path.exists(os.path.join(app.rails_config_path, f, "config.yaml"))
        )
    ]

    return [{"id": config_id} for config_id in config_ids]


# One instance of LLMRails per config id
llm_rails_instances: dict[str, LLMRails] = {}
llm_rails_events_history_cache: dict[str, dict] = {}


def _generate_cache_key(config_ids: List[str]) -> str:
    """Generates a cache key for the given config ids."""

    return "-".join((config_ids))  # remove sorted


def _get_rails(config_ids: List[str]) -> LLMRails:
    """Returns the rails instance for the given config id."""

    # If we have a single config id, we just use it as the key
    configs_cache_key = _generate_cache_key(config_ids)

    if configs_cache_key in llm_rails_instances:
        return llm_rails_instances[configs_cache_key]

    # In single-config mode, we only load the main config directory
    if app.single_config_mode:
        if config_ids != [app.single_config_id]:
            raise ValueError(f"Invalid configuration ids: {config_ids}")

        # We set this to an empty string so tha when joined with the root path, we
        # get the same thing.
        config_ids = [""]

    full_llm_rails_config: Optional[RailsConfig] = None

    for config_id in config_ids:
        base_path = os.path.abspath(app.rails_config_path)
        full_path = os.path.normpath(os.path.join(base_path, config_id))

        # @NOTE: (Rdinu) Reject config_ids that contain dangerous characters or sequences
        if re.search(r"[\\/]|(\.\.)", config_id):
            raise ValueError("Invalid config_id.")

        if os.path.commonprefix([full_path, base_path]) != base_path:
            raise ValueError("Access to the specified path is not allowed.")

        rails_config = RailsConfig.from_path(full_path)

        if not full_llm_rails_config:
            full_llm_rails_config = rails_config
        else:
            full_llm_rails_config += rails_config

    if full_llm_rails_config is None:
        raise ValueError("No valid rails configuration found.")

    llm_rails = LLMRails(config=full_llm_rails_config, verbose=True)
    llm_rails_instances[configs_cache_key] = llm_rails

    # If we have a cache for the events, we restore it
    llm_rails.events_history_cache = llm_rails_events_history_cache.get(configs_cache_key, {})

    return llm_rails


@app.post(
    "/v1/chat/completions",
    response_model=ResponseBody,
    response_model_exclude_none=True,
)
async def chat_completion(body: RequestBody, request: Request):
    """Chat completion for the provided conversation.

    TODO: add support for explicit state object.
    """
    log.info("Got request for config %s", body.config_id)
    for logger in registered_loggers:
        asyncio.get_event_loop().create_task(logger({"endpoint": "/v1/chat/completions", "body": body.json()}))

    # Save the request headers in a context variable.
    api_request_headers.set(request.headers)

    # Use Request config_ids if set, otherwise use the FastAPI default config.
    # If neither is available we can't generate any completions as we have no config_id
    config_ids = body.config_ids
    if not config_ids:
        if app.default_config_id:
            config_ids = [app.default_config_id]
        else:
            raise GuardrailsConfigurationError("No request config_ids provided and server has no default configuration")

    try:
        llm_rails = _get_rails(config_ids)
    except ValueError as ex:
        log.exception(ex)
        return ResponseBody(
            messages=[
                {
                    "role": "assistant",
                    "content": f"Could not load the {config_ids} guardrails configuration. "
                    f"An internal error has occurred.",
                }
            ]
        )

    try:
        messages = body.messages or []
        if body.context:
            messages.insert(0, {"role": "context", "content": body.context})

        # If we have a `thread_id` specified, we need to look up the thread
        datastore_key = None

        if body.thread_id:
            if datastore is None:
                raise RuntimeError("No DataStore has been configured.")

            # We make sure the `thread_id` meets the minimum complexity requirement.
            if len(body.thread_id) < 16:
                return ResponseBody(
                    messages=[
                        {
                            "role": "assistant",
                            "content": "The `thread_id` must have a minimum length of 16 characters.",
                        }
                    ]
                )

            # Fetch the existing thread messages. For easier management, we prepend
            # the string `thread-` to all thread keys.
            datastore_key = "thread-" + body.thread_id
            thread_messages = json.loads(await datastore.get(datastore_key) or "[]")

            # And prepend them.
            messages = thread_messages + messages

        if body.stream and llm_rails.config.streaming_supported and llm_rails.main_llm_supports_streaming:
            # Create the streaming handler instance
            streaming_handler = StreamingHandler()

            # Start the generation
            asyncio.create_task(
                llm_rails.generate_async(
                    messages=messages,
                    streaming_handler=streaming_handler,
                    options=body.options,
                    state=body.state,
                )
            )

            # TODO: Add support for thread_ids in streaming mode

            return StreamingResponse(streaming_handler)
        else:
            res = await llm_rails.generate_async(messages=messages, options=body.options, state=body.state)

            if isinstance(res, GenerationResponse):
                bot_message_content = res.response[0]
                # Ensure bot_message is always a dict
                if isinstance(bot_message_content, str):
                    bot_message = {"role": "assistant", "content": bot_message_content}
                else:
                    bot_message = bot_message_content
            else:
                assert isinstance(res, dict)
                bot_message = res

            # If we're using threads, we also need to update the data before returning
            # the message.
            if body.thread_id and datastore is not None and datastore_key is not None:
                await datastore.set(datastore_key, json.dumps(messages + [bot_message]))

            result = ResponseBody(messages=[bot_message])

            # If we have additional GenerationResponse fields, we return as well
            if isinstance(res, GenerationResponse):
                result.llm_output = res.llm_output
                result.output_data = res.output_data
                result.log = res.log
                result.state = res.state

            return result

    except Exception as ex:
        log.exception(ex)
        return ResponseBody(messages=[{"role": "assistant", "content": "Internal server error."}])


def _create_error_response(error: str, details: Optional[str] = None) -> GuardrailCheckResponseBody:
    """Create a standardized error response."""
    guardrails_data = {"error": error}
    if details:
        guardrails_data["details"] = details
    return GuardrailCheckResponseBody(status="error", rails_status={}, guardrails_data=guardrails_data)


def _load_rails(config_id: Optional[str] = None, inline_config: Optional[dict] = None) -> LLMRails:
    """Load rails from either config_id or inline config."""
    if inline_config:
        rails_config = (
            RailsConfig.from_content(yaml_content=inline_config)
            if isinstance(inline_config, str)
            else RailsConfig.from_content(config=inline_config)
        )
        return LLMRails(config=rails_config, verbose=True)

    # config_id must be provided if inline_config is not
    if not config_id:
        raise ValueError("Either config_id or inline_config must be provided")

    return _get_rails([config_id])


def _create_check_options(run_input: bool, run_output: bool) -> GenerationOptions:
    """Create GenerationOptions for guardrail checks.

    All LLM and rail parameters come from the guardrail configuration.
    """
    return GenerationOptions(
        rails=GenerationRailsOptions(
            input=run_input,
            output=run_output,
            retrieval=False,
            dialog=False,
            tool_input=False,
            tool_output=False,
        ),
        log=GenerationLogOptions(activated_rails=True, internal_events=True, llm_calls=True),
    )


def _calculate_status(rails_status: dict) -> str:
    """Calculate overall status from rails status dictionary."""
    return "blocked" if any(s.get("status") == "blocked" for s in rails_status.values()) else "success"


async def _check_conversation_context(body: GuardrailCheckRequestBody, llm_rails: LLMRails) -> str:
    """Check all messages together with full conversation context.

    Returns a JSON string of the response.
    """
    # Determine which rails to run based on last message role
    if not body.messages:
        return json.dumps(_create_error_response("Messages list cannot be empty.").model_dump()) + "\n"

    last_role = body.messages[-1].get("role")
    if last_role == "user":
        options = _create_check_options(run_input=True, run_output=False)
    elif last_role == "assistant":
        options = _create_check_options(run_input=False, run_output=True)
    else:
        return (
            json.dumps(_create_error_response("Last message must have 'user' or 'assistant' role.").model_dump()) + "\n"
        )

    # Check entire conversation with context
    res = await llm_rails.generate_async(messages=body.messages, options=options)

    # Build response from result
    rails_status = {}
    activated_rails_set = set()
    stats = None

    if isinstance(res, GenerationResponse) and res.log:
        if res.log.activated_rails:
            for rail in res.log.activated_rails:
                rails_status[rail.name] = {"status": "blocked" if rail.stop else "success"}
                activated_rails_set.add(rail.name)

        if res.log.stats:
            stats = {
                "input_rails_duration": res.log.stats.input_rails_duration or 0,
                "output_rails_duration": res.log.stats.output_rails_duration or 0,
                "total_duration": res.log.stats.total_duration or 0,
                "llm_calls_duration": res.log.stats.llm_calls_duration or 0,
                "llm_calls_count": res.log.stats.llm_calls_count or 0,
                "llm_calls_total_prompt_tokens": res.log.stats.llm_calls_total_prompt_tokens or 0,
                "llm_calls_total_completion_tokens": res.log.stats.llm_calls_total_completion_tokens or 0,
                "llm_calls_total_tokens": res.log.stats.llm_calls_total_tokens or 0,
            }

    guardrails_data = None
    if activated_rails_set or stats:
        guardrails_data = {"log": {"activated_rails": list(activated_rails_set), "stats": stats or {}}}

    final_result = GuardrailCheckResponseBody(
        status=_calculate_status(rails_status),
        rails_status=rails_status,
        guardrails_data=guardrails_data,
    )
    return json.dumps(final_result.model_dump()) + "\n"


@app.post(
    "/v1/guardrail/checks",
    response_model=GuardrailCheckResponseBody,
)
async def guardrail_checks(body: GuardrailCheckRequestBody, request: Request):
    """Check messages against guardrails without generating LLM responses.

    This endpoint runs input/output rails on the provided messages and returns
    the status of each rail. It does not generate new content, only validates
    existing messages against configured guardrails.
    """
    log.info("Got guardrail check request")
    for logger in registered_loggers:
        asyncio.get_event_loop().create_task(
            logger({"endpoint": "/v1/guardrail/checks", "body": body.model_dump_json()})
        )

    api_request_headers.set(request.headers)

    async def process_checks():
        """Process guardrail checks and yield results."""
        try:
            # Validate request
            if not body.messages:
                yield json.dumps(_create_error_response("Messages list cannot be empty.").model_dump()) + "\n"
                return

            # Determine which config to use
            if body.guardrails is not None:
                config_id = body.guardrails.get("config_id")
                inline_config = body.guardrails.get("config")

                if config_id and inline_config:
                    yield (
                        json.dumps(
                            _create_error_response(
                                "Only one of 'config_id' or 'config' should be provided in guardrails field."
                            ).model_dump()
                        )
                        + "\n"
                    )
                    return
                if not (config_id or inline_config):
                    yield (
                        json.dumps(
                            _create_error_response(
                                "Either 'config_id' or 'config' must be provided in guardrails field."
                            ).model_dump()
                        )
                        + "\n"
                    )
                    return
            else:
                # Use default config if no guardrails specified
                config_id = app.default_config_id
                inline_config = None

                if not config_id:
                    yield (
                        json.dumps(
                            _create_error_response(
                                "No guardrails configuration provided and no default configuration set on server."
                            ).model_dump()
                        )
                        + "\n"
                    )
                    return

            # Load rails configuration
            try:
                llm_rails = _load_rails(config_id, inline_config)
            except Exception as ex:
                log.exception(ex)
                error_msg = (
                    "Could not load guardrails configuration."
                    if isinstance(ex, ValueError)
                    else "Failed to load guardrails configuration."
                )
                yield json.dumps(_create_error_response(error_msg, str(ex)).model_dump()) + "\n"
                return

            # Check if conversation context mode is enabled
            if body.use_conversation_context:
                # Conversation mode: check all messages together with context
                yield await _check_conversation_context(body, llm_rails)
                return

            # Process messages: check each user/assistant message with appropriate rails
            all_rails_status = {}
            activated_rails_set = set()
            all_stats = []

            for msg in body.messages:
                if not isinstance(msg, dict) or "role" not in msg:
                    continue

                role = msg.get("role")
                content = msg.get("content", "")

                # Determine which rails to run
                if role == "user":
                    options = _create_check_options(run_input=True, run_output=False)
                    check_messages = [{"role": "user", "content": content}]
                elif role == "assistant":
                    options = _create_check_options(run_input=False, run_output=True)
                    check_messages = [
                        {"role": "user", "content": ""},
                        {"role": "assistant", "content": content},
                    ]
                else:
                    continue

                # Execute guardrail check
                res = await llm_rails.generate_async(messages=check_messages, options=options)

                # Collect results
                if isinstance(res, GenerationResponse) and res.log:
                    if res.log.activated_rails:
                        for rail in res.log.activated_rails:
                            # Keep blocked status if already blocked, don't downgrade to success
                            current_status = "blocked" if rail.stop else "success"
                            if rail.name not in all_rails_status or current_status == "blocked":
                                all_rails_status[rail.name] = {"status": current_status}
                            activated_rails_set.add(rail.name)

                    if res.log.stats:
                        all_stats.append(res.log.stats)

                # Stream intermediate result if streaming enabled
                if body.stream:
                    intermediate_result = {
                        "status": _calculate_status(all_rails_status),
                        "rails_status": all_rails_status.copy(),
                        "guardrails_data": None,
                    }
                    yield json.dumps(intermediate_result) + "\n"

            # Build final response
            guardrails_data = None
            if all_stats or activated_rails_set:
                aggregated_stats = {}
                if all_stats:
                    fields = [
                        "input_rails_duration",
                        "output_rails_duration",
                        "total_duration",
                        "llm_calls_duration",
                        "llm_calls_count",
                        "llm_calls_total_prompt_tokens",
                        "llm_calls_total_completion_tokens",
                        "llm_calls_total_tokens",
                    ]
                    aggregated_stats = {
                        field: sum(getattr(stat, field, 0) or 0 for stat in all_stats) for field in fields
                    }

                guardrails_data = {"log": {"activated_rails": list(activated_rails_set), "stats": aggregated_stats}}

            final_result = GuardrailCheckResponseBody(
                status=_calculate_status(all_rails_status),
                rails_status=all_rails_status,
                guardrails_data=guardrails_data,
            )
            yield json.dumps(final_result.model_dump()) + "\n"

        except Exception as ex:
            log.exception(ex)
            yield json.dumps(_create_error_response("Internal server error.", str(ex)).model_dump()) + "\n"

    if body.stream:
        return StreamingResponse(process_checks(), media_type="application/x-ndjson")
    else:
        # Non-streaming: collect all results and return final response
        results = []
        async for result in process_checks():
            results.append(result)

        # Return the last result (final response)
        if results:
            return GuardrailCheckResponseBody.model_validate_json(results[-1])
        else:
            return _create_error_response("No results generated.")


# By default, there are no challenges
challenges = []


def register_challenges(additional_challenges: List[dict]):
    """Register additional challenges

    Args:
        additional_challenges: The new challenges to be registered.
    """
    challenges.extend(additional_challenges)


@app.get(
    "/v1/challenges",
    summary="Get list of available challenges.",
)
async def get_challenges():
    """Returns the list of available challenges for red teaming."""

    return challenges


def register_datastore(datastore_instance: DataStore):
    """Registers a DataStore to be used by the server."""
    global datastore

    datastore = datastore_instance


def register_logger(logger: Callable):
    """Register an additional logger"""
    registered_loggers.append(logger)


def start_auto_reload_monitoring():
    """Start a thread that monitors the config folder for changes."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        class Handler(FileSystemEventHandler):
            def on_any_event(self, event):
                if event.is_directory:
                    return None

                elif event.event_type == "created" or event.event_type == "modified":
                    log.info(f"Watchdog received {event.event_type} event for file {event.src_path}")

                    # Compute the relative path
                    src_path_str = str(event.src_path)
                    rel_path = os.path.relpath(src_path_str, app.rails_config_path)

                    # The config_id is the first component
                    parts = rel_path.split(os.path.sep)
                    config_id = parts[0]

                    if (
                        not parts[-1].startswith(".")
                        and ".ipynb_checkpoints" not in parts
                        and os.path.isfile(src_path_str)
                    ):
                        # We just remove the config from the cache so that a new one is used next time
                        if config_id in llm_rails_instances:
                            instance = llm_rails_instances[config_id]
                            del llm_rails_instances[config_id]
                            if instance:
                                val = instance.events_history_cache
                                # We save the events history cache, to restore it on the new instance
                                llm_rails_events_history_cache[config_id] = val

                            log.info(f"Configuration {config_id} has changed. Clearing cache.")

        observer = Observer()
        event_handler = Handler()
        observer.schedule(event_handler, app.rails_config_path, recursive=True)
        observer.start()
        try:
            while not app.stop_signal:
                time.sleep(5)
        finally:
            observer.stop()
            observer.join()

    except ImportError:
        # Since this is running in a separate thread, we just print the error.
        print("The auto-reload feature requires `watchdog`. Please install using `pip install watchdog`.")
        # Force close everything.
        os._exit(-1)


def set_default_config_id(config_id: str):
    app.default_config_id = config_id


class GuardrailsConfigurationError(Exception):
    """Exception raised for errors in the configuration."""

    pass


# # Register a nicer error message for 422 error
# def register_exception(app: FastAPI):
#     @app.exception_handler(RequestValidationError)
#     async def validation_exception_handler(
#         request: Request, exc: RequestValidationError
#     ):
#         exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
#         # or logger.error(f'{exc}')
#         log.error(request, exc_str)
#         content = {"status_code": 10422, "message": exc_str, "data": None}
#         return JSONResponse(
#             content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
#         )
#
#
# register_exception(app)
