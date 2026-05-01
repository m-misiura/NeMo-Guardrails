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

"""Unit tests for the hf_classifier rail."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest import mock

import pytest
from aioresponses import aioresponses
from pydantic import ValidationError

from nemoguardrails.library.hf_classifier import backends as backends_mod
from nemoguardrails.library.hf_classifier.actions import _classify_and_check
from nemoguardrails.library.hf_classifier.backends import (
    ClassificationResult,
    FMSBackend,
    KServeBackend,
    LocalBackend,
    VLLMBackend,
    _build_headers,
    _build_ssl_context,
    _get_timeout,
    get_backend,
)
from nemoguardrails.rails.llm.config import (
    LocalHFClassifierConfig,
    RemoteHFClassifierConfig,
)

_REMOTE_DEFAULTS = dict(
    backend="vllm",
    model_name="test-model",
    endpoint="http://localhost:8000",
    threshold=0.5,
    blocked_labels=["toxic"],
)


def _remote(**overrides) -> RemoteHFClassifierConfig:
    return RemoteHFClassifierConfig(**{**_REMOTE_DEFAULTS, **overrides})


def _local(**overrides) -> LocalHFClassifierConfig:
    defaults = dict(backend="local", model_name="test-model", blocked_labels=["toxic"])
    return LocalHFClassifierConfig(**{**defaults, **overrides})


def _rails_cfg(name, hf_config):
    return SimpleNamespace(rails=SimpleNamespace(config=SimpleNamespace(hf_classifier={name: hf_config})))


@pytest.fixture(autouse=True)
def _clear_caches():
    backends_mod._pipelines.clear()
    backends_mod._ssl_cache.clear()
    backends_mod._warned_env_vars.clear()
    yield
    backends_mod._pipelines.clear()
    backends_mod._ssl_cache.clear()
    backends_mod._warned_env_vars.clear()


class TestConfig:
    def test_local_has_no_endpoint(self):
        c = _local()
        assert not hasattr(c, "endpoint") or c.model_fields.get("endpoint") is None

    @pytest.mark.parametrize("backend", ["vllm", "kserve", "fms"])
    def test_remote_requires_endpoint(self, backend):
        with pytest.raises(ValidationError, match="endpoint"):
            RemoteHFClassifierConfig(backend=backend, model_name="m", blocked_labels=["x"])

    def test_invalid_endpoint_scheme(self):
        with pytest.raises(ValidationError, match="http://"):
            _remote(endpoint="ftp://host:8000")

    def test_aggregation_rejects_text_classification(self):
        with pytest.raises(ValidationError, match="aggregation_strategy"):
            _local(parameters={"aggregation_strategy": "simple"}, task="text-classification")

    def test_aggregation_accepts_token_classification(self):
        c = _local(task="token-classification", parameters={"aggregation_strategy": "simple"})
        assert c.parameters["aggregation_strategy"] == "simple"

    def test_remote_has_no_aggregation(self):
        assert "aggregation_strategy" not in RemoteHFClassifierConfig.model_fields

    @pytest.mark.parametrize("val", [-0.1, 1.1])
    def test_threshold_out_of_range(self, val):
        with pytest.raises(ValidationError):
            _remote(threshold=val)

    def test_empty_blocked_labels_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            _remote(blocked_labels=[])
        assert "blocked_labels is empty" in caplog.text

    def test_verify_ssl_false_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            _remote(parameters={"verify_ssl": False})
        assert "TLS verification is disabled" in caplog.text


class TestHeaders:
    def test_defaults(self):
        assert _build_headers(_remote()) == {"Content-Type": "application/json"}

    def test_api_key(self, monkeypatch):
        monkeypatch.setenv("K", "secret")
        h = _build_headers(_remote(api_key_env_var="K"))
        assert h["Authorization"] == "Bearer secret"

    def test_missing_key_warns_once(self, caplog):
        c = _remote(api_key_env_var="MISSING")
        with caplog.at_level(logging.WARNING):
            _build_headers(c)
        assert "MISSING" in caplog.text
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            _build_headers(c)
        assert "MISSING" not in caplog.text

    def test_case_insensitive_override(self):
        h = _build_headers(_remote(parameters={"default_headers": {"content-type": "text/plain"}}))
        assert "Content-Type" not in h
        assert h["content-type"] == "text/plain"


class TestSSLTimeout:
    def test_timeout_default(self):
        assert _get_timeout(_remote()).total == 30.0

    def test_timeout_custom(self):
        assert _get_timeout(_remote(parameters={"timeout": 10.0})).total == 10.0

    def test_ssl_default(self):
        assert _build_ssl_context(_remote()) is None

    def test_ssl_disabled(self):
        assert _build_ssl_context(_remote(parameters={"verify_ssl": False})) is False

    def test_ssl_cached(self):
        c = _remote(parameters={"verify_ssl": False})
        assert _build_ssl_context(c) is _build_ssl_context(c)


class TestLocalBackend:
    @pytest.mark.asyncio
    async def test_text_classification(self):
        c = _local()
        backends_mod._pipelines["text-classification:test-model:None"] = mock.MagicMock(
            return_value=[{"label": "toxic", "score": 0.9}]
        )
        r = await LocalBackend(c).classify("text")
        assert r == [ClassificationResult(label="toxic", score=0.9)]

    @pytest.mark.asyncio
    async def test_token_entity_group(self):
        c = _local(
            task="token-classification",
            blocked_labels=["PER"],
            parameters={"aggregation_strategy": "simple"},
        )
        backends_mod._pipelines["token-classification:test-model:simple"] = mock.MagicMock(
            return_value=[{"entity_group": "PER", "score": 0.85}]
        )
        r = await LocalBackend(c).classify("John")
        assert r[0]["label"] == "PER"

    @pytest.mark.asyncio
    async def test_token_entity_fallback(self):
        c = _local(
            task="token-classification",
            blocked_labels=["LOC"],
            parameters={"aggregation_strategy": "simple"},
        )
        backends_mod._pipelines["token-classification:test-model:simple"] = mock.MagicMock(
            return_value=[{"entity": "LOC", "score": 0.7}]
        )
        r = await LocalBackend(c).classify("Paris")
        assert r[0]["label"] == "LOC"


class TestVLLMBackend:
    _URL = "http://vllm:8000/classify"

    def _backend(self):
        return VLLMBackend(_remote(backend="vllm", endpoint="http://vllm:8000"))

    @pytest.mark.asyncio
    async def test_success(self):
        with aioresponses() as m:
            m.post(self._URL, payload={"data": [{"label": "toxic", "probs": [0.9, 0.1]}]})
            r = await self._backend().classify("text")
        assert r[0] == ClassificationResult(label="toxic", score=0.9)

    @pytest.mark.asyncio
    async def test_empty_probs_defaults_to_one(self):
        with aioresponses() as m:
            m.post(self._URL, payload={"data": [{"label": "safe", "probs": []}]})
            r = await self._backend().classify("text")
        assert r[0]["score"] == 1.0

    @pytest.mark.asyncio
    async def test_non_200(self):
        with aioresponses() as m:
            m.post(self._URL, status=500, body="error")
            with pytest.raises(ValueError, match="500"):
                await self._backend().classify("text")

    @pytest.mark.asyncio
    async def test_missing_data_key(self):
        with aioresponses() as m:
            m.post(self._URL, payload={"wrong": []})
            with pytest.raises(ValueError, match="Unexpected vLLM"):
                await self._backend().classify("text")


class TestKServeBackend:
    _URL = "http://ks:8080/v1/models/m:predict"

    def _backend(self):
        return KServeBackend(_remote(backend="kserve", endpoint="http://ks:8080", model_name="m"))

    @pytest.mark.asyncio
    async def test_dict_prediction(self):
        with aioresponses() as m:
            m.post(self._URL, payload={"predictions": [{"0": 0.1, "1": 0.9}]})
            r = await self._backend().classify("text")
        assert {x["label"] for x in r} == {"0", "1"}

    @pytest.mark.asyncio
    async def test_int_prediction(self):
        with aioresponses() as m:
            m.post(self._URL, payload={"predictions": [2]})
            r = await self._backend().classify("text")
        assert r == [ClassificationResult(label="2", score=1.0)]

    @pytest.mark.asyncio
    async def test_list_flattens_dedupes(self):
        with aioresponses() as m:
            m.post(self._URL, payload={"predictions": [[[0, 1, 2], [1, 0, 3]]]})
            r = await self._backend().classify("text")
        assert [x["label"] for x in r] == ["1", "2", "3"]

    @pytest.mark.asyncio
    async def test_empty_predictions(self):
        with aioresponses() as m:
            m.post(self._URL, payload={"predictions": []})
            assert await self._backend().classify("text") == []

    @pytest.mark.asyncio
    async def test_unknown_type_raises(self):
        with aioresponses() as m:
            m.post(self._URL, payload={"predictions": ["bad"]})
            with pytest.raises(ValueError, match="Unexpected KServe prediction type"):
                await self._backend().classify("text")


class TestFMSBackend:
    _URL = "http://fms:9000/api/v1/text/contents"

    def _backend(self):
        return FMSBackend(_remote(backend="fms", endpoint="http://fms:9000"))

    @pytest.mark.asyncio
    async def test_success(self):
        with aioresponses() as m:
            m.post(self._URL, payload=[[{"detection_type": "harm", "score": 0.95}]])
            r = await self._backend().classify("text")
        assert r == [ClassificationResult(label="harm", score=0.95)]

    @pytest.mark.asyncio
    async def test_empty_detections(self):
        with aioresponses() as m:
            m.post(self._URL, payload=[[]])
            assert await self._backend().classify("text") == []

    @pytest.mark.asyncio
    async def test_non_200(self):
        with aioresponses() as m:
            m.post(self._URL, status=503, body="down")
            with pytest.raises(ValueError, match="503"):
                await self._backend().classify("text")

    @pytest.mark.asyncio
    async def test_malformed_structure(self):
        with aioresponses() as m:
            m.post(self._URL, payload={"bad": []})
            with pytest.raises(ValueError, match="Unexpected FMS response"):
                await self._backend().classify("text")


class TestGetBackend:
    @pytest.mark.parametrize(
        "name,cls",
        [
            ("local", LocalBackend),
            ("vllm", VLLMBackend),
            ("kserve", KServeBackend),
            ("fms", FMSBackend),
        ],
    )
    def test_known(self, name, cls):
        if name == "local":
            cfg = _local()
        else:
            cfg = _remote(backend=name)
        assert isinstance(get_backend(cfg), cls)

    def test_unknown_raises(self):
        c = RemoteHFClassifierConfig.model_construct(
            backend="bogus",
            model_name="x",
            endpoint="http://x",
            threshold=0.5,
            blocked_labels=[],
            parameters={},
            api_key_env_var=None,
        )
        with pytest.raises(ValueError, match="Unknown hf_classifier backend"):
            get_backend(c)


class TestClassifyAndCheck:
    def _mock_backend(self, results):
        b = mock.AsyncMock()
        b.classify.return_value = results
        return b

    def _patch(self, results):
        return mock.patch(
            "nemoguardrails.library.hf_classifier.actions.get_backend",
            return_value=self._mock_backend(results),
        )

    @pytest.mark.asyncio
    async def test_blocks_above_threshold(self):
        c = _remote(threshold=0.5, blocked_labels=["toxic"])
        with self._patch([ClassificationResult(label="toxic", score=0.8)]):
            assert await _classify_and_check("t", "bad", _rails_cfg("t", c)) is False

    @pytest.mark.asyncio
    async def test_allows_below_threshold(self):
        c = _remote(threshold=0.5, blocked_labels=["toxic"])
        with self._patch([ClassificationResult(label="toxic", score=0.3)]):
            assert await _classify_and_check("t", "ok", _rails_cfg("t", c)) is True

    @pytest.mark.asyncio
    async def test_blocks_at_exact_threshold(self):
        c = _remote(threshold=0.5, blocked_labels=["toxic"])
        with self._patch([ClassificationResult(label="toxic", score=0.5)]):
            assert await _classify_and_check("t", "edge", _rails_cfg("t", c)) is False

    @pytest.mark.asyncio
    async def test_allows_non_blocked_label(self):
        c = _remote(threshold=0.5, blocked_labels=["toxic"])
        with self._patch([ClassificationResult(label="safe", score=0.99)]):
            assert await _classify_and_check("t", "ok", _rails_cfg("t", c)) is True

    @pytest.mark.asyncio
    async def test_no_config_raises(self):
        with pytest.raises(ValueError, match="no 'hf_classifier' section"):
            await _classify_and_check("t", "text", None)

    @pytest.mark.asyncio
    async def test_unknown_classifier_raises(self):
        c = _remote()
        with pytest.raises(ValueError, match="Unknown classifier 'bad'"):
            await _classify_and_check("bad", "text", _rails_cfg("good", c))

    @pytest.mark.asyncio
    async def test_empty_results_warns(self, caplog):
        c = _remote(threshold=0.5, blocked_labels=["toxic"])
        with self._patch([]):
            with caplog.at_level(logging.WARNING):
                result = await _classify_and_check("t", "text", _rails_cfg("t", c))
        assert result is True
        assert "returned no results" in caplog.text
