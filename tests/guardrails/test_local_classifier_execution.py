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

"""The real classifier action receives managed execution only under IORails."""

import asyncio
import threading

import pytest

from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.guardrails.local_execution import LocalExecution
from nemoguardrails.library.hf_classifier import backends
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.options import RailStatus, RailType
from tests.guardrails.async_helpers import wait_for_queue_state, wait_until
from tests.utils import TestChat


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    monkeypatch.setattr(backends, "_pipelines", {})
    monkeypatch.setattr(backends, "_backend_instances", {})
    monkeypatch.setattr("nemoguardrails.guardrails.compiled_rail._is_installed", lambda _: True)


@pytest.fixture
def config(request):
    return RailsConfig.from_content(
        yaml_content=f"""
models: []
rails:
  input:
    flows: [hf classifier check input $classifier=toxicity]
  config:
    hf_classifier:
      toxicity:
        engine: local
        model: /nonexistent/hf-classifier-execution-test
        num_instances: {getattr(request, "param", 1)}
        blocked_labels: [toxic]
"""
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("text,expected", [("safe", RailStatus.PASSED), ("toxic", RailStatus.BLOCKED)])
@pytest.mark.parametrize("config", [1, 2], indirect=True)
async def test_both_engines_and_direct_backend_preserve_decisions(config, monkeypatch, text, expected):
    threads = []

    def classify(text):
        threads.append(threading.current_thread().name)
        return [{"label": text, "score": 0.99}]

    monkeypatch.setattr(backends, "_get_or_create_pipeline", lambda *args: classify)
    messages = [{"role": "user", "content": text}]
    engine = IORails(config)
    assert engine._local_executor._max_concurrency == config.rails.config.hf_classifier["toxicity"].num_instances
    try:
        result = await engine.check_async(messages, rail_types=[RailType.INPUT])
        assert result.status == expected
        assert threads[-1].startswith("guardrails-local")
    finally:
        await engine.stop()

    chat = TestChat(config)
    result = await chat.app.check_async(messages, rail_types=[RailType.INPUT])
    assert result.status == expected
    assert not threads[-1].startswith("guardrails-local")
    local_config = config.rails.config.hf_classifier["toxicity"]
    assert await backends.LocalBackend(local_config).classify(text) == [{"label": text, "score": 0.99}]


@pytest.mark.asyncio
async def test_stopping_one_owner_preserves_other_owner_and_shared_pipeline(config, monkeypatch):
    local_config = config.rails.config.hf_classifier["toxicity"]
    key = backends._pipeline_cache_key(local_config.model, local_config.task, local_config.parameters)

    def pipeline(text):
        return [{"label": "safe", "score": 0.99}]

    monkeypatch.setitem(backends._pipelines, key, pipeline)
    first, second = IORails(config), IORails(config)
    messages = [{"role": "user", "content": "safe"}]
    try:
        results = await asyncio.gather(first.check_async(messages), second.check_async(messages))
        assert all(result.status == RailStatus.PASSED for result in results)
        await first.stop()
        assert backends._pipelines[key] is pipeline
        assert (await second.check_async(messages)).status == RailStatus.PASSED
        managed = backends.get_backend(local_config, local_executor=first._local_executor)
        assert managed is not backends.get_backend(local_config, local_executor=second._local_executor)
        assert backends.get_backend(local_config)._local_executor is None
    finally:
        await first.stop()
        await second.stop()


@pytest.mark.asyncio
async def test_local_execution_failure_is_not_an_allow(config, monkeypatch):
    def fail(*args):
        raise RuntimeError("local model failed")

    monkeypatch.setattr(backends, "_get_or_create_pipeline", fail)
    engine = IORails(config)
    try:
        result = await engine.check_async([{"role": "user", "content": "safe"}])
        assert result.status != RailStatus.PASSED
    finally:
        await engine.stop()

    legacy = TestChat(config)
    result = await legacy.app.check_async([{"role": "user", "content": "safe"}])
    assert result.status != RailStatus.PASSED


@pytest.mark.asyncio
async def test_local_queue_overload_fails_closed(config, monkeypatch):
    monkeypatch.setattr(
        "nemoguardrails.guardrails.iorails.LocalExecution",
        lambda **kwargs: LocalExecution(max_queue_size=1, max_concurrency=1),
    )
    entered = threading.Event()
    release = threading.Event()

    def blocking():
        entered.set()
        assert release.wait(5)

    engine = IORails(config)
    await engine.start()
    running = asyncio.create_task(engine._local_executor.run(blocking))
    queued = None
    try:
        await wait_until(entered.is_set)
        queued = asyncio.create_task(engine._local_executor.run(lambda: None))
        await wait_for_queue_state(engine._local_executor._queue, busy=1, pending=1)
        result = await engine.check_async([{"role": "user", "content": "safe"}])
        assert result.status != RailStatus.PASSED
    finally:
        release.set()
        await asyncio.gather(running, *([queued] if queued else []))
        await engine.stop()
