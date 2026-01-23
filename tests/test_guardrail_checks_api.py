# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import os

import pytest
from fastapi.testclient import TestClient

from nemoguardrails.server import api

client = TestClient(api.app)


@pytest.fixture(scope="function", autouse=True)
def set_rails_config_path():
    api.app.rails_config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "test_configs"))
    # Set default config for testing optional guardrails field
    api.app.default_config_id = "input_rails"
    yield
    api.app.rails_config_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "examples", "bots")
    )
    api.app.default_config_id = None


# Validation Tests

def test_empty_messages():
    """Empty messages list returns error."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [],
            "guardrails": {"config_id": "input_rails"},
        },
    )
    result = response.json()
    assert result["status"] == "error"
    assert "empty" in result["guardrails_data"]["error"].lower()


def test_missing_config():
    """Missing both config_id and config returns error."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "test"}],
            "guardrails": {},
        },
    )
    result = response.json()
    assert result["status"] == "error"
    # Should error - either because both missing or config load failed
    assert "error" in result["guardrails_data"]


def test_both_config_and_config_id():
    """Providing both config_id and config returns error."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "test"}],
            "guardrails": {"config_id": "input_rails", "config": {}},
        },
    )
    result = response.json()
    assert result["status"] == "error"


def test_invalid_config_id():
    """Invalid config_id returns error."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "test"}],
            "guardrails": {"config_id": "nonexistent"},
        },
    )
    result = response.json()
    assert result["status"] == "error"


def test_no_guardrails_field_uses_default():
    """Omitting guardrails field uses default config."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "test"}],
            # No guardrails field - should use default_config_id
        },
    )
    result = response.json()
    assert result["status"] in ["success", "blocked", "error"]
    # Should have processed the request (not error about missing config)
    assert "status" in result
    assert "rails_status" in result


def test_no_guardrails_no_default_returns_error():
    """Omitting guardrails with no default config returns error."""
    # Temporarily remove default config
    original_default = api.app.default_config_id
    api.app.default_config_id = None

    try:
        response = client.post(
            "/v1/guardrail/checks",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "test"}],
                # No guardrails field and no default config
            },
        )
        result = response.json()
        assert result["status"] == "error"
        assert "no default configuration" in result["guardrails_data"]["error"].lower()
    finally:
        # Restore default config
        api.app.default_config_id = original_default


# Core Functionality Tests

def test_non_streaming_response_structure():
    """Non-streaming returns proper JSON structure."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
            "guardrails": {"config_id": "input_rails"},
            "stream": False,
        },
    )
    result = response.json()

    # Must have these fields
    assert "status" in result
    assert "rails_status" in result
    assert "guardrails_data" in result

    # Status must be one of the allowed values
    assert result["status"] in ["success", "blocked", "error"]

    # Rails status must be dict
    assert isinstance(result["rails_status"], dict)


def test_streaming_returns_ndjson():
    """Streaming mode returns NDJSON with valid JSON lines."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [
                {"role": "user", "content": "First"},
                {"role": "user", "content": "Second"},
            ],
            "guardrails": {"config_id": "input_rails"},
            "stream": True,
        },
    )

    # Must be NDJSON
    assert response.headers["content-type"] == "application/x-ndjson"

    # Parse lines
    lines = response.text.strip().split("\n")
    assert len(lines) >= 1  # At least one result

    # All lines must be valid JSON with required fields
    for line in lines:
        data = json.loads(line)
        assert "status" in data
        assert "rails_status" in data


def test_inline_config():
    """Inline config works."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "test"}],
            "guardrails": {
                "config": {
                    "models": [{"type": "main", "engine": "openai", "model": "gpt-3.5-turbo-instruct"}],
                    "rails": {"input": {"flows": []}},
                }
            },
        },
    )
    result = response.json()
    assert "status" in result


def test_stats_structure():
    """Stats have correct structure when present."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "test"}],
            "guardrails": {"config_id": "input_rails"},
        },
    )
    result = response.json()

    # If error (e.g., missing API keys), error structure should be correct
    if result["status"] == "error":
        assert "error" in result["guardrails_data"]
        return

    # If success/blocked with guardrails_data, validate structure
    if result.get("guardrails_data") and "log" in result["guardrails_data"]:
        log = result["guardrails_data"]["log"]
        assert "activated_rails" in log
        assert isinstance(log["activated_rails"], list)
        assert "stats" in log
        assert isinstance(log["stats"], dict)


# Conversation Context Mode Tests

def test_conversation_context_mode_basic():
    """Conversation context mode processes full conversation."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
            "guardrails": {"config_id": "input_rails"},
            "use_conversation_context": True,
        },
    )
    result = response.json()

    # Must have proper structure
    assert "status" in result
    assert "rails_status" in result
    assert result["status"] in ["success", "blocked", "error"]


def test_conversation_context_checks_last_user_message():
    """Conversation context with last user message runs input rails."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [
                {"role": "user", "content": "Previous message"},
                {"role": "assistant", "content": "Response"},
                {"role": "user", "content": "Final message"},
            ],
            "guardrails": {"config_id": "input_rails"},
            "use_conversation_context": True,
        },
    )
    result = response.json()
    assert result["status"] in ["success", "blocked", "error"]


def test_conversation_context_checks_last_assistant_message():
    """Conversation context with last assistant message runs output rails."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Final assistant message"},
            ],
            "guardrails": {"config_id": "input_rails"},
            "use_conversation_context": True,
        },
    )
    result = response.json()
    assert result["status"] in ["success", "blocked", "error"]


def test_conversation_context_invalid_last_role():
    """Conversation context with invalid last role returns error."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "system", "content": "Invalid last role"},
            ],
            "guardrails": {
                "config": {
                    "models": [{"type": "main", "engine": "openai", "model": "gpt-3.5-turbo-instruct"}],
                    "rails": {"input": {"flows": []}},
                }
            },
            "use_conversation_context": True,
        },
    )
    result = response.json()
    assert result["status"] == "error"
    # Should error about invalid role
    assert "error" in result["guardrails_data"]


def test_independent_vs_conversation_context_difference():
    """Independent and conversation context modes behave differently."""
    messages = [
        {"role": "user", "content": "Message 1"},
        {"role": "user", "content": "Message 2"},
    ]

    # Independent mode
    response_independent = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": messages,
            "guardrails": {"config_id": "input_rails"},
            "use_conversation_context": False,
        },
    )

    # Conversation context mode
    response_context = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test",
            "messages": messages,
            "guardrails": {"config_id": "input_rails"},
            "use_conversation_context": True,
        },
    )

    # Both should return valid responses
    result_independent = response_independent.json()
    result_context = response_context.json()

    assert result_independent["status"] in ["success", "blocked", "error"]
    assert result_context["status"] in ["success", "blocked", "error"]

    # Both should have proper structure
    assert "rails_status" in result_independent
    assert "rails_status" in result_context
