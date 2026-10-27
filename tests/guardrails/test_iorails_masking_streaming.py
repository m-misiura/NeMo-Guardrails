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

"""Output masks inspect complete responses without exposing fragments or failed generations."""

import asyncio
import json

import pytest

from nemoguardrails.actions.rail_outcome import RailOutcome, TransformTarget
from nemoguardrails.guardrails.compiled_rail import RailExecution
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.imports import check_optional_dependency
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.options import GenerationOptions
from nemoguardrails.testing.fake_model import FakeLLMModel
from nemoguardrails.types import LLMResponseChunk
from tests.guardrails.rail_stubs import StubRail, rails_compiled_as

MASK_FLOW = "mask sensitive data on output"
CHECK_FLOW = "detect sensitive data on output"
EMAIL = "alice@example.com"


class EmailMask(StubRail):
    def __init__(self):
        super().__init__(transform_target=TransformTarget.BOT_MESSAGE)

    async def execute(self, messages, bot_response=None):
        await super().execute(messages, bot_response)
        masked = bot_response.replace(EMAIL, "<EMAIL_ADDRESS>")
        outcome = (
            RailOutcome.transform([(TransformTarget.BOT_MESSAGE, masked)])
            if masked != bot_response
            else RailOutcome.allow()
        )
        return RailExecution(outcome=outcome)


class FragmentedModel(FakeLLMModel):
    def __init__(self, fragments, *, pause=False, fail=False):
        super().__init__(responses=["".join(fragments)])
        self.fragments = fragments
        self.pause = pause
        self.fail = fail
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()

    async def stream_async(self, prompt, **kwargs):
        self._next_response()
        try:
            for fragment in self.fragments:
                yield LLMResponseChunk(delta_content=fragment)
                await asyncio.sleep(0)
            if self.pause:
                self.waiting.set()
                await self.release.wait()
            if self.fail:
                raise RuntimeError("synthetic provider failure")
        finally:
            self.finished.set()


def masking_engine(monkeypatch, model, *, check=None, chunk_size=1):
    config = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {
                "output": {
                    "parallel": True,
                    "flows": [MASK_FLOW, CHECK_FLOW],
                    "streaming": {"enabled": True, "stream_first": False, "context_size": 0, "chunk_size": chunk_size},
                },
                "config": {"sensitive_data_detection": {"output": {"entities": ["EMAIL_ADDRESS"]}}},
            },
        }
    )
    mask = EmailMask()
    check = check or StubRail()
    with rails_compiled_as({MASK_FLOW: mask, CHECK_FLOW: check}):
        engine = IORails(config, _report_usage=False)

    async def stream(model_type, messages, **kwargs):
        async for chunk in model.stream_async(messages, **kwargs):
            yield chunk

    monkeypatch.setattr(engine.engine_registry, "stream_model_call", stream)
    return engine, mask, check


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fragments,expected",
    [
        (["alice@", "example.com", " tail"], "<EMAIL_ADDRESS> tail"),
        ([EMAIL, " tail"], "<EMAIL_ADDRESS> tail"),
        (["ordinary", " text"], "ordinary text"),
        (
            [
                '{"error":',
                '{"type":"generation_error","message":"alice@example.com"}}',
            ],
            '{"error":{"type":"generation_error","message":"<EMAIL_ADDRESS>"}}',
        ),
        ([], ""),
    ],
)
@pytest.mark.parametrize("chunk_size", [1, 2, 200])
async def test_masking_is_independent_of_stream_fragmentation(monkeypatch, fragments, expected, chunk_size):
    engine, mask, check = masking_engine(monkeypatch, FragmentedModel(fragments), chunk_size=chunk_size)
    async with engine:
        chunks = [chunk async for chunk in engine.stream_async(messages=[{"role": "user", "content": "hello"}])]
    assert "".join(chunks) == expected
    assert EMAIL not in "".join(chunks)
    assert mask.call_count == check.call_count == bool(fragments)
    if fragments:
        assert check.seen_bot_response == expected
        assert len(chunks) == 1


@pytest.mark.asyncio
async def test_no_text_or_checks_before_generation_finishes(monkeypatch):
    model = FragmentedModel(["alice@", "example.com"], pause=True)
    engine, mask, check = masking_engine(monkeypatch, model)
    async with engine:
        stream = engine.stream_async(messages=[{"role": "user", "content": "hello"}])
        first = asyncio.create_task(anext(stream))
        try:
            await asyncio.wait_for(model.waiting.wait(), 2)
            assert not first.done()
            assert mask.call_count == check.call_count == 0
            model.release.set()
            assert await first == "<EMAIL_ADDRESS>"
            assert [chunk async for chunk in stream] == []
        finally:
            model.release.set()
            await asyncio.gather(first, return_exceptions=True)
            await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, [CHECK_FLOW]])
async def test_disabled_mask_keeps_incremental_streaming(monkeypatch, enabled):
    model = FragmentedModel(["ordinary"], pause=True)
    engine, mask, check = masking_engine(monkeypatch, model)
    async with engine:
        stream = engine.stream_async(
            messages=[{"role": "user", "content": "hello"}], options=GenerationOptions(rails={"output": enabled})
        )
        try:
            assert await asyncio.wait_for(anext(stream), 2) == "ordinary"
            assert not model.finished.is_set()
            assert mask.call_count == 0
            assert check.call_count == (enabled == [CHECK_FLOW])
        finally:
            model.release.set()
            await stream.aclose()


@pytest.mark.asyncio
async def test_provider_failure_discards_buffered_text(monkeypatch):
    engine, mask, check = masking_engine(monkeypatch, FragmentedModel([EMAIL], fail=True))
    async with engine:
        chunks = [chunk async for chunk in engine.stream_async(messages=[{"role": "user", "content": "hello"}])]
    assert len(chunks) == 1
    assert json.loads(chunks[0])["error"]
    assert EMAIL not in chunks[0]
    assert mask.call_count == check.call_count == 0


@pytest.mark.asyncio
async def test_blocked_response_never_emits_text(monkeypatch):
    engine, mask, check = masking_engine(
        monkeypatch, FragmentedModel([EMAIL]), check=StubRail(RailOutcome.block(reason="unsafe"))
    )
    async with engine:
        chunks = [chunk async for chunk in engine.stream_async(messages=[{"role": "user", "content": "hello"}])]
    assert len(chunks) == 1
    assert json.loads(chunks[0])["error"]["code"] == "content_blocked"
    assert EMAIL not in chunks[0]
    assert check.seen_bot_response == "<EMAIL_ADDRESS>"


@pytest.mark.asyncio
async def test_cancellation_discards_text_and_stops_generation(monkeypatch):
    model = FragmentedModel([EMAIL], pause=True)
    engine, mask, check = masking_engine(monkeypatch, model)
    async with engine:
        stream = engine.stream_async(messages=[{"role": "user", "content": "hello"}])
        first = asyncio.create_task(anext(stream))
        try:
            await asyncio.wait_for(model.waiting.wait(), 2)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            await asyncio.wait_for(model.finished.wait(), 2)
            assert mask.call_count == check.call_count == 0
        finally:
            model.release.set()
            await asyncio.gather(first, return_exceptions=True)
            await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not all(
        check_optional_dependency(name)
        for name in ["presidio_analyzer", "presidio_anonymizer", "spacy", "en_core_web_lg"]
    ),
    reason="Requires Presidio and the installed en_core_web_lg model",
)
async def test_real_presidio_masks_input_and_fragmented_output_with_hap(monkeypatch):
    from nemoguardrails.library.hf_classifier import backends

    seen = []

    def classify(text, **kwargs):
        seen.append(text)
        return [{"label": "LABEL_0", "score": 0.99}]

    monkeypatch.setattr(backends, "_get_or_create_pipeline", lambda *args, **kwargs: classify)
    config = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {
                "input": {
                    "parallel": True,
                    "flows": [
                        "hf classifier check input $classifier=hap",
                        "detect sensitive data on input",
                        "mask sensitive data on input",
                    ],
                },
                "output": {
                    "parallel": True,
                    "flows": [CHECK_FLOW, MASK_FLOW],
                    "streaming": {"enabled": True, "stream_first": False, "context_size": 0, "chunk_size": 1},
                },
                "config": {
                    "hf_classifier": {
                        "hap": {
                            "engine": "local",
                            "model": "ibm-granite/granite-guardian-hap-125m",
                            "threshold": 0.5,
                            "blocked_labels": ["LABEL_1"],
                        }
                    },
                    "sensitive_data_detection": {
                        "input": {"entities": ["EMAIL_ADDRESS"]},
                        "output": {"entities": ["EMAIL_ADDRESS"]},
                    },
                },
            },
        }
    )
    model = FragmentedModel(["alice@", "example.com", " tail"])
    engine = IORails(config, _report_usage=False)
    sent = []

    async def stream(model_type, messages, **kwargs):
        sent.extend(messages)
        async for chunk in model.stream_async(messages, **kwargs):
            yield chunk

    monkeypatch.setattr(engine.engine_registry, "stream_model_call", stream)
    messages = [{"role": "user", "content": EMAIL}]
    async with engine:
        chunks = [chunk async for chunk in engine.stream_async(messages=messages)]
    assert chunks == ["<EMAIL_ADDRESS> tail"]
    assert seen == ["<EMAIL_ADDRESS>"]
    assert sent[-1]["content"] == "<EMAIL_ADDRESS>"
    assert messages[-1]["content"] == EMAIL
