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

"""Pipeline leases follow physical inference across runtimes and event loops."""

import asyncio
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemoguardrails.guardrails.local_execution import LocalExecution
from nemoguardrails.library.hf_classifier import backends
from nemoguardrails.rails.llm.config import RailsConfig
from tests.guardrails.async_helpers import wait_until


def config(**options):
    return RailsConfig.from_content(
        config={
            "models": [],
            "rails": {
                "config": {
                    "hf_classifier": {
                        "test": {
                            "engine": "local",
                            "model": "/nonexistent/hf-classifier-pool-test",
                            "blocked_labels": ["toxic"],
                            **options,
                        }
                    }
                }
            },
        }
    ).rails.config.hf_classifier["test"]


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    monkeypatch.setattr(backends, "_pipelines", {})
    monkeypatch.setattr(backends, "_backend_instances", {})


@pytest.fixture
def transformers(monkeypatch):
    module = MagicMock()
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setitem(sys.modules, "nemoguardrails.library.hf_classifier.chunking", MagicMock())
    return module


@pytest.mark.parametrize("count", [0, -1, True, 2.5, "2"])
def test_invalid_capacity_is_rejected(count):
    with pytest.raises(ValidationError):
        config(num_instances=count)


def test_default_and_explicit_capacity_and_shared_tokenizer_validation():
    assert config().num_instances == 1
    assert config(num_instances=2).num_instances == 2
    with pytest.raises(ValidationError, match="shared tokenizer object"):
        config(num_instances=2, parameters={"tokenizer": object()})
    assert config(num_instances=2, parameters={"tokenizer": "tokenizer-path"}).num_instances == 2


def test_pool_identity_includes_capacity_but_not_policy_or_http_options():
    first = backends._get_pipeline_pool(config(num_instances=2))
    assert first is backends._get_pipeline_pool(config(num_instances=2, threshold=0.9, parameters={"timeout": 1}))
    assert first is not backends._get_pipeline_pool(config(num_instances=1))
    assert first is not backends._get_pipeline_pool(config(num_instances=2, parameters={"revision": "other"}))


@pytest.mark.asyncio
@pytest.mark.parametrize("task", ["text-classification", "token-classification"])
async def test_independent_instances_execute_concurrently_and_are_reused(task, monkeypatch, transformers):
    entered = threading.Barrier(2)
    created = []
    calls = []
    building = threading.Lock()

    def factory(**kwargs):
        with building:
            index = len(created)
            exclusive = threading.Lock()
            created.append(kwargs)

        def classify(text):
            assert exclusive.acquire(blocking=False), "Pipeline was borrowed twice"
            try:
                calls.append(index)
                if text == "hold":
                    entered.wait(timeout=10)
                label = "label" if task == "text-classification" else "entity"
                return [{label: "safe", "score": 0.9}]
            finally:
                exclusive.release()

        return classify

    monkeypatch.setattr(transformers, "pipeline", factory)
    cfg = config(num_instances=2, task=task)
    first = backends.get_backend(cfg, name="first")
    second = backends.get_backend(cfg, name="second")
    result = await asyncio.wait_for(asyncio.gather(first.classify("hold"), second.classify("hold")), 15)
    assert result == [[{"label": "safe", "score": 0.9}]] * 2
    assert sorted(calls) == [0, 1]
    assert len(created) == 2
    assert all("num_instances" not in kwargs for kwargs in created)
    await first.classify("again")
    assert len(created) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("managed", [False, True])
async def test_cancellation_does_not_release_pipeline_early(managed, monkeypatch):
    cfg = config()
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def pipeline(text):
        calls.append(text)
        if text == "first":
            entered.set()
            assert release.wait(10)
        return [{"label": "safe", "score": 0.9}]

    monkeypatch.setattr(backends, "_get_or_create_pipeline", lambda *args: pipeline)
    execution = LocalExecution(max_queue_size=4, max_concurrency=2)
    await execution.start()
    backend = backends.LocalBackend(cfg, local_executor=execution if managed else None)
    first = asyncio.create_task(backend.classify("first"))
    second = None
    try:
        await wait_until(entered.is_set)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        pool = backends._get_pipeline_pool(cfg)
        assert pool._available.qsize() == 0
        # A legacy caller shares the lease even when the cancelled caller is managed.
        second = asyncio.create_task(backends.LocalBackend(cfg).classify("second"))
        await asyncio.sleep(0)
        assert calls == ["first"]
        release.set()
        assert await asyncio.wait_for(second, 10) == [{"label": "safe", "score": 0.9}]
        assert calls == ["first", "second"]
        assert pool._available.qsize() == 1
    finally:
        release.set()
        await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
        await execution.stop()


@pytest.mark.asyncio
async def test_load_failure_is_cached_and_releases_lease(monkeypatch, transformers):
    loads = []

    def fail(**kwargs):
        loads.append(kwargs)
        raise ValueError("model unavailable")

    monkeypatch.setattr(transformers, "pipeline", fail)
    cfg = config(num_instances=2)
    backend = backends.LocalBackend(cfg)
    with pytest.raises(ValueError, match="model unavailable"):
        await backend.classify("text")
    assert backends._get_pipeline_pool(cfg)._available.qsize() == 2
    with pytest.raises(RuntimeError, match="failed to load"):
        await backend.classify("text")
    assert len(loads) == 1


@pytest.mark.asyncio
async def test_cold_construction_is_serialized_but_extra_instances_are_lazy(monkeypatch, transformers):
    entered = threading.Event()
    release = threading.Event()
    loads = []

    def factory(**kwargs):
        loads.append(kwargs)
        if len(loads) == 1:
            entered.set()
            assert release.wait(10)
        return lambda text: [{"label": "safe", "score": 0.9}]

    monkeypatch.setattr(transformers, "pipeline", factory)
    backend = backends.LocalBackend(config(num_instances=2))
    first = asyncio.create_task(backend.classify("first"))
    second = None
    try:
        await wait_until(entered.is_set)
        second = asyncio.create_task(backend.classify("second"))
        await wait_until(
            lambda: sum(isinstance(value, backends._PipelineLoading) for value in backends._pipelines.values()) == 2
        )
        assert len(loads) == 1
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), 10)
        assert len(loads) == 2
        lazy = backends.LocalBackend(config(num_instances=4))
        for _ in range(4):
            await lazy.classify("sequential")
        assert len(loads) == 3
    finally:
        release.set()
        await asyncio.gather(first, *([second] if second else []), return_exceptions=True)


@pytest.mark.asyncio
async def test_inference_failure_releases_lease(monkeypatch):
    def pipeline(text):
        if text == "fail":
            raise ValueError("inference failed")
        return [{"label": "safe", "score": 0.9}]

    monkeypatch.setattr(backends, "_get_or_create_pipeline", lambda *args: pipeline)
    backend = backends.LocalBackend(config(num_instances=2))
    with pytest.raises(ValueError, match="inference failed"):
        await backend.classify("fail")
    assert await asyncio.wait_for(backend.classify("safe"), 5) == [{"label": "safe", "score": 0.9}]


def test_different_event_loops_share_exclusive_lease(monkeypatch):
    exclusive = threading.Lock()
    entered = threading.Event()
    release = threading.Event()

    def pipeline(text):
        assert exclusive.acquire(blocking=False)
        try:
            if text == "first":
                entered.set()
                assert release.wait(10)
            return [{"label": "safe", "score": 0.9}]
        finally:
            exclusive.release()

    monkeypatch.setattr(backends, "_get_or_create_pipeline", lambda *args: pipeline)
    cfg = config()

    def run(text):
        return asyncio.run(backends.LocalBackend(cfg).classify(text))

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(run, "first")
        try:
            assert entered.wait(5)
            second = executor.submit(run, "second")
        finally:
            release.set()
        assert first.result(timeout=10) == second.result(timeout=10)
