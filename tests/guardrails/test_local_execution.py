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

"""Local execution capacity follows physical work, not cancelled awaiters."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar

import pytest

from nemoguardrails.guardrails.local_execution import LocalExecution
from tests.guardrails.async_helpers import wait_for_queue_state, wait_until


@pytest.mark.asyncio
async def test_running_cancellation_retains_capacity_and_queued_cancellation_skips_work():
    execution = LocalExecution(max_queue_size=1, max_concurrency=1)
    await execution.start()
    entered = threading.Event()
    release = threading.Event()
    called = []

    def blocking():
        entered.set()
        assert release.wait(5)

    first = asyncio.create_task(execution.run(blocking))
    pending = None
    try:
        await wait_until(entered.is_set)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        pending = asyncio.create_task(execution.run(lambda: called.append(True)))
        await wait_for_queue_state(execution._queue, busy=1, pending=1)
        with pytest.raises(asyncio.QueueFull):
            await execution.run(lambda: None)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        release.set()
        await execution.stop()
        assert called == []
    finally:
        release.set()
        await execution.stop()
        await asyncio.gather(first, *([pending] if pending else []), return_exceptions=True)


@pytest.mark.asyncio
async def test_request_context_and_failures_are_isolated():
    execution = LocalExecution(max_queue_size=8)
    request = ContextVar("local_execution_test_request", default="outside")
    await execution.start()

    async def call(value):
        token = request.set(value)
        try:
            return await execution.run(request.get)
        finally:
            request.reset(token)

    def fail():
        raise ValueError("inference failed")

    try:
        assert await asyncio.gather(call("one"), call("two")) == ["one", "two"]
        with pytest.raises(ValueError, match="inference failed"):
            await execution.run(fail)
        assert await execution.run(request.get) == "outside"
    finally:
        await execution.stop()


@pytest.mark.asyncio
async def test_cancelled_shutdown_closes_admission_and_drains_before_restart():
    execution = LocalExecution(max_queue_size=1, max_concurrency=1)
    await execution.start()
    entered = threading.Event()
    release = threading.Event()

    def blocking():
        entered.set()
        assert release.wait(5)
        return "finished"

    running = asyncio.create_task(execution.run(blocking))
    try:
        await wait_until(entered.is_set)
        stopping = asyncio.create_task(execution.stop())
        await wait_until(lambda: not execution._accepting)
        stopping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stopping
        with pytest.raises(RuntimeError, match="not accepting"):
            await execution.run(lambda: None)
        assert execution._executor is not None
        release.set()
        assert await running == "finished"
        await execution.stop()
        assert execution._executor is None
        await execution.start()
        assert await execution.run(lambda: 42) == 42
    finally:
        release.set()
        await execution.stop()
        await asyncio.gather(running, return_exceptions=True)


@pytest.mark.asyncio
async def test_failure_during_shutdown_is_retryable(monkeypatch):
    execution = LocalExecution(max_queue_size=1)
    await execution.start()
    await execution.run(lambda: None)
    stop = execution._queue.stop

    async def fail():
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(execution._queue, "stop", fail)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        await execution.stop()
    monkeypatch.setattr(execution._queue, "stop", stop)
    await execution.stop()
    assert execution._executor is None


@pytest.mark.asyncio
async def test_lazy_resources_and_loop_ownership():
    execution = LocalExecution(max_queue_size=1)
    await execution.start()
    assert execution._executor is None
    assert execution._queue is None

    def another_loop():
        async def run():
            with pytest.raises(RuntimeError, match="another event loop"):
                await execution.run(lambda: None)
            with pytest.raises(RuntimeError, match="another event loop"):
                await execution.stop()

        asyncio.run(run())

    await asyncio.to_thread(another_loop)
    assert await execution.run(lambda: "owner still usable") == "owner still usable"
    await execution.stop()


@pytest.mark.asyncio
async def test_blocked_local_work_does_not_occupy_default_executor(monkeypatch):
    execution = LocalExecution(max_queue_size=4, max_concurrency=2)
    await execution.start()
    entered = threading.Barrier(3)
    release = threading.Event()

    def blocking():
        entered.wait(timeout=5)
        assert release.wait(5)

    with ThreadPoolExecutor(max_workers=1) as default_executor:
        monkeypatch.setattr(asyncio.get_running_loop(), "_default_executor", default_executor)
        jobs = [asyncio.create_task(execution.run(blocking)) for _ in range(2)]
        try:
            await asyncio.to_thread(entered.wait, 5)
            assert await asyncio.wait_for(asyncio.to_thread(lambda: "available"), 2) == "available"
        finally:
            release.set()
            await asyncio.gather(*jobs)
            await execution.stop()
