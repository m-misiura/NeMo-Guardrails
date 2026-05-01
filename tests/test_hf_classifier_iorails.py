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

"""Tests for HF classifier IORails integration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from nemoguardrails.guardrails.actions.hf_classifier_action import (
    HFClassifierInputAction,
    HFClassifierOutputAction,
    _extract_classifier,
)
from nemoguardrails.library.hf_classifier import backends as backends_mod
from nemoguardrails.library.hf_classifier.backends import ClassificationResult
from nemoguardrails.rails.llm.config import (
    RemoteHFClassifierConfig,
)


def _remote(**overrides) -> RemoteHFClassifierConfig:
    defaults = dict(
        backend="vllm",
        model_name="test-model",
        endpoint="http://localhost:8000",
        threshold=0.5,
        blocked_labels=["toxic"],
    )
    return RemoteHFClassifierConfig(**{**defaults, **overrides})


def _make_task_manager(hf_classifier_config: dict):
    config = SimpleNamespace(rails=SimpleNamespace(config=SimpleNamespace(hf_classifier=hf_classifier_config)))
    return SimpleNamespace(config=config)


def _make_action(action_cls, hf_classifier_config: dict):
    tm = _make_task_manager(hf_classifier_config)
    engine_registry = SimpleNamespace()
    return action_cls(engine_registry, tm)


@pytest.fixture(autouse=True)
def _clear_caches():
    backends_mod._backend_instances.clear()
    yield
    backends_mod._backend_instances.clear()


class TestExtractClassifier:
    def test_extracts_name(self):
        assert _extract_classifier("hf classifier check input $classifier=hap") == "hap"

    def test_returns_none_without_prefix(self):
        assert _extract_classifier("hf classifier check input") is None

    def test_strips_whitespace(self):
        assert _extract_classifier("hf classifier check input $classifier=hap  ") == "hap"


class TestHFClassifierInputAction:
    @pytest.mark.asyncio
    async def test_blocks_above_threshold(self):
        cfg = _remote(threshold=0.5, blocked_labels=["toxic"])
        action = _make_action(HFClassifierInputAction, {"t": cfg})

        with mock.patch(
            "nemoguardrails.guardrails.actions.hf_classifier_action.get_backend",
            return_value=mock.AsyncMock(
                classify=mock.AsyncMock(return_value=[ClassificationResult(label="toxic", score=0.9)])
            ),
        ):
            result = await action.run(
                "hf classifier check input $classifier=t",
                [{"role": "user", "content": "bad text"}],
            )

        assert not result.is_safe
        assert "toxic" in result.reason

    @pytest.mark.asyncio
    async def test_allows_below_threshold(self):
        cfg = _remote(threshold=0.5, blocked_labels=["toxic"])
        action = _make_action(HFClassifierInputAction, {"t": cfg})

        with mock.patch(
            "nemoguardrails.guardrails.actions.hf_classifier_action.get_backend",
            return_value=mock.AsyncMock(
                classify=mock.AsyncMock(return_value=[ClassificationResult(label="toxic", score=0.2)])
            ),
        ):
            result = await action.run(
                "hf classifier check input $classifier=t",
                [{"role": "user", "content": "ok text"}],
            )

        assert result.is_safe

    @pytest.mark.asyncio
    async def test_missing_classifier_raises(self):
        action = _make_action(HFClassifierInputAction, {"other": _remote()})

        result = await action.run(
            "hf classifier check input $classifier=nonexistent",
            [{"role": "user", "content": "text"}],
        )

        assert not result.is_safe
        assert "Unknown hf_classifier" in result.reason

    @pytest.mark.asyncio
    async def test_missing_classifier_param_raises(self):
        action = _make_action(HFClassifierInputAction, {"t": _remote()})

        with pytest.raises(RuntimeError, match="No \\$model="):
            await action.run(
                "hf classifier check input",
                [{"role": "user", "content": "text"}],
            )


class TestHFClassifierOutputAction:
    @pytest.mark.asyncio
    async def test_checks_bot_response(self):
        cfg = _remote(threshold=0.5, blocked_labels=["toxic"])
        action = _make_action(HFClassifierOutputAction, {"t": cfg})

        with mock.patch(
            "nemoguardrails.guardrails.actions.hf_classifier_action.get_backend",
            return_value=mock.AsyncMock(
                classify=mock.AsyncMock(return_value=[ClassificationResult(label="toxic", score=0.8)])
            ),
        ) as mock_backend:
            result = await action.run(
                "hf classifier check output $classifier=t",
                [{"role": "user", "content": "hi"}],
                bot_response="toxic output",
            )
            mock_backend.return_value.classify.assert_called_once_with("toxic output")

        assert not result.is_safe

    @pytest.mark.asyncio
    async def test_empty_bot_response(self):
        cfg = _remote(threshold=0.5, blocked_labels=["toxic"])
        action = _make_action(HFClassifierOutputAction, {"t": cfg})

        with mock.patch(
            "nemoguardrails.guardrails.actions.hf_classifier_action.get_backend",
            return_value=mock.AsyncMock(classify=mock.AsyncMock(return_value=[])),
        ):
            result = await action.run(
                "hf classifier check output $classifier=t",
                [{"role": "user", "content": "hi"}],
                bot_response=None,
            )

        assert result.is_safe


class TestIORailsFlowSelection:
    def test_hf_classifier_input_in_iorails_flows(self):
        from nemoguardrails.guardrails.guardrails import IORAILS_INPUT_FLOWS

        assert "hf classifier check input" in IORAILS_INPUT_FLOWS

    def test_hf_classifier_output_in_iorails_flows(self):
        from nemoguardrails.guardrails.guardrails import IORAILS_OUTPUT_FLOWS

        assert "hf classifier check output" in IORAILS_OUTPUT_FLOWS

    def test_action_registered_in_rails_manager(self):
        from nemoguardrails.guardrails.rails_manager import _ACTION_CLASSES

        assert "hf classifier check input" in _ACTION_CLASSES
        assert "hf classifier check output" in _ACTION_CLASSES
        assert _ACTION_CLASSES["hf classifier check input"] is HFClassifierInputAction
        assert _ACTION_CLASSES["hf classifier check output"] is HFClassifierOutputAction
