# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for header forwarding and log redaction."""

import pytest

from nemoguardrails.header_forwarding import api_request_headers_var, get_extra_headers_from_request


def _set_headers(headers):
    api_request_headers_var.set(headers)


def test_no_headers_returns_none():
    api_request_headers_var.set(None)
    assert get_extra_headers_from_request() is None


def test_mixed_headers_full_scenario():
    """Core test: infra filtered, x-auth wins, non-x ignored, custom forwarded."""
    _set_headers(
        {
            "authorization": "Bearer oauth-token",
            "x-authorization": "Bearer llm-key",
            "x-forwarded-for": "1.2.3.4",
            "x-remote-user": "admin",
            "x-real-ip": "5.6.7.8",
            "x-request-id": "abc",
            "x-maas-subscription": "sub-key",
            "content-type": "application/json",
        }
    )
    result = get_extra_headers_from_request(forward_auth=True)
    assert result == {
        "Authorization": "Bearer llm-key",
        "x-maas-subscription": "sub-key",
    }


def test_forward_auth_false_skips_auth():
    _set_headers({"authorization": "Bearer key", "x-authorization": "Bearer key2"})
    assert get_extra_headers_from_request(forward_auth=False) is None


def test_authorization_never_forwarded_to_llm():
    """Authorization header (K8s/proxy auth) must never reach the LLM."""
    _set_headers({"authorization": "Bearer k8s-token"})
    result = get_extra_headers_from_request(forward_auth=True)
    assert result is None


def test_x_authorization_forwarded_without_authorization():
    _set_headers({"x-authorization": "Bearer llm-key"})
    result = get_extra_headers_from_request(forward_auth=True)
    assert result == {"Authorization": "Bearer llm-key"}


# ---------------------------------------------------------------------------
# Regression: extra_headers must arrive as HTTP headers, not JSON body fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extra_headers_sent_as_http_headers_not_json_body(httpx_mock):
    """Regression test for rhoai-3.5: extra_headers was serialised into the
    JSON request body instead of being sent as HTTP headers."""
    from nemoguardrails.llm.clients.openai_compatible import OpenAICompatibleClient

    httpx_mock.add_response(
        json={
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "model": "test",
        }
    )

    client = OpenAICompatibleClient(base_url="https://llm.example.com/v1", api_key="static-key")
    await client.chat_completion(
        "test-model",
        [{"role": "user", "content": "hi"}],
        extra_headers={"Authorization": "Bearer forwarded-key", "x-custom": "val"},
    )

    request = httpx_mock.get_request()
    assert request.headers["Authorization"] == "Bearer forwarded-key"
    assert request.headers["x-custom"] == "val"

    import json

    body = json.loads(request.content)
    assert "extra_headers" not in body


@pytest.mark.asyncio
async def test_extra_headers_not_provided_uses_static_key(httpx_mock):
    """When no extra_headers are passed, static api_key auth is used."""
    from nemoguardrails.llm.clients.openai_compatible import OpenAICompatibleClient

    httpx_mock.add_response(
        json={
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "model": "test",
        }
    )

    client = OpenAICompatibleClient(base_url="https://llm.example.com/v1", api_key="static-key")
    await client.chat_completion("test-model", [{"role": "user", "content": "hi"}])

    request = httpx_mock.get_request()
    assert request.headers["Authorization"] == "Bearer static-key"
