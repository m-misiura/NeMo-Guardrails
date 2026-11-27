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

"""Bounded local inference owned by an IORails runtime."""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context, copy_context
from typing import Any, TypeVar

from nemoguardrails.guardrails.async_work_queue import AsyncWorkQueue

T = TypeVar("T")


class LocalExecution:
    """Keep blocking inference and lock waiters out of the default executor.

    Resources are allocated on first use. Closing admission precedes draining;
    cancellation of a caller never releases a worker running physical inference.
    """

    def __init__(self, *, max_queue_size: int, max_concurrency: int = 1) -> None:
        if max_queue_size < 1 or max_concurrency < 1:
            raise ValueError("Local execution requires positive queue and concurrency limits.")
        self._max_queue_size = max_queue_size
        self._max_concurrency = max_concurrency
        self._queue: AsyncWorkQueue[Any] | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._accepting = False
        self._shutdown_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._shutdown_task is not None:
            await self.stop()
        if self._loop is not None and self._loop is not asyncio.get_running_loop():
            raise RuntimeError("Local execution belongs to another event loop.")
        self._loop = asyncio.get_running_loop()
        self._accepting = True

    async def run(self, function: Callable[[], T]) -> T:
        if not self._accepting:
            raise RuntimeError("Local execution is not accepting work.")
        if self._loop is not asyncio.get_running_loop():
            raise RuntimeError("Local execution belongs to another event loop.")
        if self._queue is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_concurrency, thread_name_prefix="guardrails-local"
            )
            self._queue = AsyncWorkQueue(
                "iorails_local_queue", self._max_queue_size, self._max_concurrency, reject_on_full=True
            )
        return await self._queue.submit(self._execute, copy_context(), function)

    async def _execute(self, context: Context, function: Callable[[], T]) -> T:
        assert self._executor is not None
        # The queue owns this coroutine. Caller cancellation cancels only the
        # result future; the worker stays busy until the thread finishes.
        return await asyncio.get_running_loop().run_in_executor(self._executor, context.run, function)

    async def stop(self) -> None:
        if self._loop is not None and self._loop is not asyncio.get_running_loop():
            raise RuntimeError("Local execution belongs to another event loop.")
        self._accepting = False
        if self._loop is None:
            return
        if self._shutdown_task is None or self._shutdown_task.done():
            self._shutdown_task = asyncio.create_task(self._shutdown())
        # Keep cleanup alive and retryable when the owner's stop is cancelled.
        await asyncio.shield(self._shutdown_task)
        self._shutdown_task = None
        self._loop = None

    async def _shutdown(self) -> None:
        if self._queue is not None:
            await self._queue.stop()
            self._queue = None
        if self._executor is not None:
            await asyncio.to_thread(self._executor.shutdown, wait=True)
            self._executor = None
