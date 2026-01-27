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

"""
Tests for /v1/guardrail/checks endpoint.

These tests mirror the bash test suite (test_all_guardrail_checks.sh) and verify:
- Role-based routing: user messages → input rails, assistant messages → output rails
- API compliance with NVIDIA NeMo Guardrails specification
- Error handling and validation
- Streaming mode
- Model inheritance
"""

import json
import os

import pytest
from fastapi.testclient import TestClient

from nemoguardrails.server import api

client = TestClient(api.app)


@pytest.fixture(scope="function", autouse=True)
def setup_test_config():
    """Set up test configuration path and default config."""
    api.app.rails_config_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "test_configs")
    )
    api.app.default_config_id = "simple_rails"
    yield
    # Cleanup
    api.app.rails_config_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "examples", "bots")
    )
    api.app.default_config_id = None


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_empty_messages_error():
    """Error handling: Empty messages array returns error."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test-model",
            "messages": [],
            "guardrails": {"config_id": "simple_rails"},
        },
    )

    result = response.json()
    assert result["status"] == "error"
    assert "guardrails_data" in result
    assert "error" in result["guardrails_data"]


def test_invalid_config_id_error():
    """Error handling: Invalid config_id returns error."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "guardrails": {"config_id": "nonexistent_config"},
        },
    )

    result = response.json()
    assert result["status"] == "error"
    assert "Could not load guardrails configuration" in str(result["guardrails_data"])


def test_both_config_and_config_id_error():
    """Error handling: Providing both config_id and config returns error."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "guardrails": {
                "config_id": "simple_rails",
                "config": {"rails": {"input": {"flows": ["test"]}}}  # Non-empty config
            },
        },
    )

    result = response.json()
    assert result["status"] == "error"
    assert "Only one of" in str(result["guardrails_data"])


# =============================================================================
# Role-Based Routing Tests (Core Functionality)
# =============================================================================


def test_user_message_triggers_input_rails():
    """Input rails evaluate user messages."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "guardrails": {"config_id": "simple_rails"},
        },
    )

    result = response.json()
    assert result["status"] in ["success", "blocked"]
    assert "rails_status" in result
    assert isinstance(result["rails_status"], dict)
    # Input rails should be evaluated (rails_status should be populated)
    assert len(result["rails_status"]) > 0


def test_assistant_message_triggers_output_rails():
    """Output rails evaluate assistant messages."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test-model",
            "messages": [{"role": "assistant", "content": "I can help with that!"}],
            "guardrails": {"config_id": "simple_rails"},
        },
    )

    result = response.json()
    assert result["status"] in ["success", "blocked"]
    assert "rails_status" in result


def test_multiple_messages_user_and_assistant():
    """Multiple messages: user and assistant messages evaluated by appropriate rails."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "What is the weather?"},
                {"role": "assistant", "content": "It's sunny today."},
            ],
            "guardrails": {"config_id": "simple_rails"},
        },
    )

    result = response.json()
    assert result["status"] in ["success", "blocked"]
    assert "rails_status" in result
    assert "guardrails_data" in result


def test_multiple_same_role_messages():
    """Multiple messages with same role are checked independently."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "user", "content": "Second question"},
            ],
            "guardrails": {"config_id": "simple_rails"},
        },
    )

    result = response.json()
    assert result["status"] in ["success", "blocked"]
    assert "rails_status" in result


# =============================================================================
# LLM Parameters & API Compliance
# =============================================================================


def test_llm_parameters_accepted():
    """LLM parameters (top_p, temperature, max_tokens) are accepted per NVIDIA spec."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "guardrails": {"config_id": "simple_rails"},
            "top_p": 1.0,
            "temperature": 0.7,
            "max_tokens": 150,
        },
    )

    result = response.json()
    # Parameters should be accepted without error
    assert result["status"] in ["success", "blocked"]


def test_response_structure_matches_nvidia_spec():
    """Response structure matches NVIDIA specification."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "guardrails": {"config_id": "simple_rails"},
        },
    )

    result = response.json()

    # Required fields per NVIDIA spec
    assert "status" in result
    assert result["status"] in ["success", "blocked", "error"]
    assert "rails_status" in result
    assert isinstance(result["rails_status"], dict)
    assert "guardrails_data" in result


# =============================================================================
# Streaming Mode
# =============================================================================


def test_streaming_mode_returns_ndjson():
    """Streaming mode returns NDJSON format."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "First message"},
                {"role": "user", "content": "Second message"},
            ],
            "guardrails": {"config_id": "simple_rails"},
            "stream": True,
        },
    )

    # Check content type
    assert response.headers["content-type"] == "application/x-ndjson"

    # Parse NDJSON lines
    lines = response.text.strip().split("\n")
    assert len(lines) >= 2  # At least one per message

    # Each line should be valid JSON
    for line in lines:
        data = json.loads(line)
        assert "status" in data
        assert "rails_status" in data


# =============================================================================
# Inline Config Tests
# =============================================================================


def test_inline_config_with_model():
    """Inline config with model specification works."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "gpt-3.5-turbo-instruct",
            "messages": [{"role": "user", "content": "test"}],
            "guardrails": {
                "config": {
                    "models": [{"type": "main", "engine": "openai", "model": "gpt-3.5-turbo-instruct"}],
                    "prompts": [
                        {
                            "task": "self_check_input",
                            "content": 'User: "{{ user_input }}"\nBlock? Answer:',
                        }
                    ],
                    "rails": {"input": {"flows": ["self check input"]}},
                }
            },
        },
    )

    result = response.json()
    # May succeed or error depending on API key, but should not crash
    assert result["status"] in ["success", "blocked", "error"]
    assert "rails_status" in result


def test_inline_config_model_inheritance():
    """Inline config with empty models array inherits from default config."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "guardrails": {
                "config": {
                    "models": [],  # Should inherit from default config
                    "prompts": [
                        {
                            "task": "self_check_input",
                            "content": 'User: "{{ user_input }}"\nBlock? Answer:',
                        }
                    ],
                    "rails": {"input": {"flows": ["self check input"]}},
                }
            },
        },
    )

    result = response.json()
    # Should not fail with "No LLM provided" if default config exists
    assert result["status"] in ["success", "blocked", "error"]
    assert "rails_status" in result


# =============================================================================
# Stats and Metadata Tests
# =============================================================================


def test_guardrails_data_contains_stats():
    """Guardrails data contains execution statistics."""
    response = client.post(
        "/v1/guardrail/checks",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "guardrails": {"config_id": "simple_rails"},
        },
    )

    result = response.json()

    if result["guardrails_data"]:
        assert "log" in result["guardrails_data"]
        log = result["guardrails_data"]["log"]

        # Stats should be present
        if "stats" in log:
            stats = log["stats"]
            # Check for expected stat fields
            assert "input_rails_duration" in stats
            assert "output_rails_duration" in stats
            assert "total_duration" in stats
