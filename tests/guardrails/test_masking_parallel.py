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

"""Masking composes before independent checks run concurrently."""

import asyncio

import pytest

from nemoguardrails.actions.rail_outcome import RailOutcome, TransformTarget
from nemoguardrails.guardrails.guardrails_types import RailDirection
from nemoguardrails.rails.llm.config import RailsConfig
from tests.guardrails.rail_stubs import StubRail, rails_compiled_as
from tests.guardrails.test_rails_manager import _make_rails_manager


def manager_with_rails(direction, rails, parallel=True):
    config = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {direction.value.lower(): {"parallel": parallel, "flows": list(rails)}},
        }
    )
    with rails_compiled_as(rails):
        return _make_rails_manager(config)


def target_for(direction):
    return TransformTarget.USER_MESSAGE if direction is RailDirection.INPUT else TransformTarget.BOT_MESSAGE


async def run_direction(manager, direction, messages, enabled=True):
    if direction is RailDirection.INPUT:
        return await manager.is_input_safe(messages, enabled=enabled)
    return await manager.is_output_safe(messages, "original", enabled=enabled)


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", list(RailDirection))
async def test_checks_overlap_only_after_masking(direction):
    mask_started, release_mask, release_checks = asyncio.Event(), asyncio.Event(), asyncio.Event()
    checks_started = [asyncio.Event(), asyncio.Event()]
    target = target_for(direction)

    class Mask(StubRail):
        async def execute(self, messages, bot_response=None):
            mask_started.set()
            await release_mask.wait()
            return await super().execute(messages, bot_response)

    class Check(StubRail):
        def __init__(self, started):
            super().__init__()
            self.started = started

        async def execute(self, messages, bot_response=None):
            self.started.set()
            await release_checks.wait()
            return await super().execute(messages, bot_response)

    mask = Mask(RailOutcome.transform([(target, "masked")]), transform_target=target)
    checks = [Check(event) for event in checks_started]
    manager = manager_with_rails(direction, {"check one": checks[0], "mask": mask, "check two": checks[1]})
    messages = [{"role": "user", "content": "original"}]
    task = asyncio.create_task(run_direction(manager, direction, messages))
    try:
        await asyncio.wait_for(mask_started.wait(), 2)
        assert not any(event.is_set() for event in checks_started)
        release_mask.set()
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in checks_started)), 2)
        release_checks.set()
        result = await task
        assert result.outcome.transform_text == {target.value: "masked"}
        assert [record.flow for record in result.records] == ["mask", "check one", "check two"]
        for check in checks:
            seen = check.seen_messages[-1]["content"] if direction is RailDirection.INPUT else check.seen_bot_response
            assert seen == "masked"
        assert messages == [{"role": "user", "content": "original"}]
    finally:
        release_mask.set()
        release_checks.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", list(RailDirection))
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("final_text", ["masked twice", "", "original"])
async def test_masks_compose_and_preserve_the_net_rewrite(direction, parallel, final_text):
    target = target_for(direction)
    first = StubRail(RailOutcome.transform([(target, "masked")]), transform_target=target)
    second = StubRail(RailOutcome.transform([(target, final_text)]), transform_target=target)
    check = StubRail()
    manager = manager_with_rails(direction, {"check": check, "first mask": first, "second mask": second}, parallel)
    result = await run_direction(manager, direction, [{"role": "user", "content": "original"}])
    assert result.outcome.transform_text == ({target.value: final_text} if final_text != "original" else {})
    assert len(result.records) == 3
    seen = check.seen_messages[-1]["content"] if direction is RailDirection.INPUT else check.seen_bot_response
    assert seen == final_text
    seen_by_second = (
        second.seen_messages[-1]["content"] if direction is RailDirection.INPUT else second.seen_bot_response
    )
    assert seen_by_second == "masked"


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", list(RailDirection))
@pytest.mark.parametrize("blocking", ["mask", "check"])
async def test_blocks_keep_their_reason_and_prior_records(direction, blocking):
    target = target_for(direction)
    mask = StubRail(RailOutcome.transform([(target, "masked")]), transform_target=target)
    check = StubRail()
    rails = {"mask": mask, "check": check}
    rails[blocking].outcome = RailOutcome.block(reason="blocked")
    result = await run_direction(
        manager_with_rails(direction, rails), direction, [{"role": "user", "content": "original"}]
    )
    assert not result.is_safe
    assert result.reason == "blocked"
    assert result.triggered_rail == blocking
    assert len(result.records) == (1 if blocking == "mask" else 2)
    assert check.call_count == (0 if blocking == "mask" else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, ["check"], ["mask"]])
async def test_per_request_selection_does_not_run_disabled_rails(enabled):
    mask = StubRail(
        RailOutcome.transform([(TransformTarget.USER_MESSAGE, "masked")]), transform_target=TransformTarget.USER_MESSAGE
    )
    check = StubRail()
    manager = manager_with_rails(RailDirection.INPUT, {"mask": mask, "check": check})
    result = await run_direction(manager, RailDirection.INPUT, [{"role": "user", "content": "original"}], enabled)
    assert mask.call_count == (enabled == ["mask"])
    assert check.call_count == (enabled == ["check"])
    assert result.outcome.is_transform == (enabled == ["mask"])


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["block", "cancel", "rewrite"])
async def test_parallel_checks_are_drained_after_masking(termination):
    started, stopped = asyncio.Event(), asyncio.Event()
    mask = StubRail(
        RailOutcome.transform([(TransformTarget.USER_MESSAGE, "masked")]), transform_target=TransformTarget.USER_MESSAGE
    )

    class Pending(StubRail):
        async def execute(self, messages, bot_response=None):
            assert messages[-1]["content"] == "masked"
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

    class Terminator(StubRail):
        async def execute(self, messages, bot_response=None):
            await started.wait()
            if termination == "cancel":
                await asyncio.Event().wait()
            return await super().execute(messages, bot_response)

    outcome = (
        RailOutcome.transform([(TransformTarget.USER_MESSAGE, "unexpected")])
        if termination == "rewrite"
        else RailOutcome.block(reason="unsafe")
    )
    manager = manager_with_rails(RailDirection.INPUT, {"mask": mask, "pending": Pending(), "stop": Terminator(outcome)})
    task = asyncio.create_task(manager.is_input_safe([{"role": "user", "content": "original"}]))
    try:
        await asyncio.wait_for(started.wait(), 2)
        if termination == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif termination == "rewrite":
            with pytest.raises(NotImplementedError, match="in parallel"):
                await task
        else:
            result = await task
            assert not result.is_safe
            assert [record.flow for record in result.records] == ["mask", "stop"]
        assert stopped.is_set()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
