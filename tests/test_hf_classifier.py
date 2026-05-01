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
from nemoguardrails.library.hf_classifier.actions import (
    _classify_and_check,
    hf_classifier_check_input,
    hf_classifier_check_output,
    hf_classifier_check_retrieval,
    hf_classifier_check_tool_input,
    hf_classifier_check_tool_output,
)
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
        assert "endpoint" not in LocalHFClassifierConfig.model_fields

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
        with mock.patch("nemoguardrails.library.hf_classifier.backends.ssl") as mock_ssl:
            c = _remote(parameters={"ca_cert": "/ca.pem"})
            first = _build_ssl_context(c)
            second = _build_ssl_context(c)
            assert first is second
            mock_ssl.create_default_context.assert_called_once()


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

    @pytest.mark.asyncio
    async def test_empty_text_no_warning(self, caplog):
        c = _remote(threshold=0.5, blocked_labels=["toxic"])
        with self._patch([]):
            with caplog.at_level(logging.WARNING):
                result = await _classify_and_check("t", "", _rails_cfg("t", c))
        assert result is True
        assert "returned no results" not in caplog.text


class TestActionContextKeys:
    def _patch(self, results):
        return mock.patch(
            "nemoguardrails.library.hf_classifier.actions.get_backend",
            return_value=mock.AsyncMock(classify=mock.AsyncMock(return_value=results)),
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "action_fn,context_key",
        [
            (hf_classifier_check_input, "user_message"),
            (hf_classifier_check_output, "bot_message"),
            (hf_classifier_check_retrieval, "relevant_chunks"),
            (hf_classifier_check_tool_input, "tool_message"),
        ],
    )
    async def test_reads_correct_context_key(self, action_fn, context_key):
        c = _remote(threshold=0.5, blocked_labels=["toxic"])
        cfg = _rails_cfg("t", c)
        with self._patch([ClassificationResult(label="toxic", score=0.9)]) as p:
            await action_fn(classifier="t", config=cfg, context={context_key: "bad"})
            p.return_value.classify.assert_called_once_with("bad")

    @pytest.mark.asyncio
    async def test_tool_output_reads_tool_calls(self):
        c = _remote(threshold=0.5, blocked_labels=["toxic"])
        cfg = _rails_cfg("t", c)
        calls = [{"function": {"name": "rm", "arguments": {}}, "id": "1"}]
        with self._patch([ClassificationResult(label="safe", score=0.1)]) as p:
            await hf_classifier_check_tool_output(
                classifier="t",
                tool_calls=calls,
                config=cfg,
                context={},
            )
            text_arg = p.return_value.classify.call_args[0][0]
            assert "rm" in text_arg

    @pytest.mark.asyncio
    async def test_tool_output_falls_back_to_context(self):
        c = _remote(threshold=0.5, blocked_labels=["toxic"])
        cfg = _rails_cfg("t", c)
        calls = [{"function": {"name": "eval"}, "id": "2"}]
        with self._patch([ClassificationResult(label="safe", score=0.1)]) as p:
            await hf_classifier_check_tool_output(
                classifier="t",
                config=cfg,
                context={"tool_calls": calls},
            )
            text_arg = p.return_value.classify.call_args[0][0]
            assert "eval" in text_arg

    @pytest.mark.asyncio
    async def test_none_context_defaults_to_empty(self):
        c = _remote(threshold=0.5, blocked_labels=["toxic"])
        cfg = _rails_cfg("t", c)
        with self._patch([]):
            result = await hf_classifier_check_input(
                classifier="t",
                config=cfg,
                context=None,
            )
        assert result is True


class TestSSLCerts:
    def test_ca_cert(self):
        with mock.patch("nemoguardrails.library.hf_classifier.backends.ssl") as mock_ssl:
            ctx = mock_ssl.create_default_context.return_value
            c = _remote(parameters={"ca_cert": "/ca.pem"})
            result = _build_ssl_context(c)
            assert result is ctx
            ctx.load_verify_locations.assert_called_once_with(cafile="/ca.pem")

    def test_mtls(self):
        with mock.patch("nemoguardrails.library.hf_classifier.backends.ssl") as mock_ssl:
            ctx = mock_ssl.create_default_context.return_value
            c = _remote(parameters={"client_cert": "/cert.pem", "client_key": "/key.pem"})
            result = _build_ssl_context(c)
            assert result is ctx
            ctx.load_cert_chain.assert_called_once_with(certfile="/cert.pem", keyfile="/key.pem")

    def test_ca_cert_plus_mtls(self):
        with mock.patch("nemoguardrails.library.hf_classifier.backends.ssl") as mock_ssl:
            ctx = mock_ssl.create_default_context.return_value
            c = _remote(
                parameters={
                    "ca_cert": "/ca.pem",
                    "client_cert": "/cert.pem",
                    "client_key": "/key.pem",
                }
            )
            result = _build_ssl_context(c)
            ctx.load_verify_locations.assert_called_once_with(cafile="/ca.pem")
            ctx.load_cert_chain.assert_called_once_with(certfile="/cert.pem", keyfile="/key.pem")
            assert result is ctx


class TestKServeErrors:
    _URL = "http://ks:8080/v1/models/m:predict"

    def _backend(self):
        return KServeBackend(_remote(backend="kserve", endpoint="http://ks:8080", model_name="m"))

    @pytest.mark.asyncio
    async def test_non_200(self):
        with aioresponses() as m:
            m.post(self._URL, status=500, body="error")
            with pytest.raises(ValueError, match="KServe predict returned 500"):
                await self._backend().classify("text")

    @pytest.mark.asyncio
    async def test_missing_predictions_key(self):
        with aioresponses() as m:
            m.post(self._URL, payload={"wrong": []})
            with pytest.raises(ValueError, match="Unexpected KServe predict response"):
                await self._backend().classify("text")


class TestFMSEdgeCases:
    _URL = "http://fms:9000/api/v1/text/contents"

    def _backend(self):
        return FMSBackend(_remote(backend="fms", endpoint="http://fms:9000"))

    @pytest.mark.asyncio
    async def test_inner_not_a_list(self):
        with aioresponses() as m:
            m.post(self._URL, payload=[42])
            with pytest.raises(ValueError, match="Unexpected FMS response"):
                await self._backend().classify("text")

    @pytest.mark.asyncio
    async def test_malformed_detection_entry(self):
        with aioresponses() as m:
            m.post(self._URL, payload=[[{"wrong_key": "x"}]])
            with pytest.raises(ValueError, match="Unexpected FMS detection entry"):
                await self._backend().classify("text")


class TestLocalImportError:
    @pytest.mark.asyncio
    async def test_missing_transformers(self):
        c = _local()
        with mock.patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError, match="transformers"):
                await LocalBackend(c).classify("text")


class TestDiscriminatedUnion:
    def test_local_backend_resolves(self):
        from pydantic import TypeAdapter

        from nemoguardrails.rails.llm.config import HFClassifierConfig

        ta = TypeAdapter(HFClassifierConfig)
        c = ta.validate_python({"backend": "local", "model_name": "m", "blocked_labels": ["x"]})
        assert isinstance(c, LocalHFClassifierConfig)

    def test_remote_backend_resolves(self):
        from pydantic import TypeAdapter

        from nemoguardrails.rails.llm.config import HFClassifierConfig

        ta = TypeAdapter(HFClassifierConfig)
        c = ta.validate_python(
            {
                "backend": "vllm",
                "model_name": "m",
                "endpoint": "http://x:8000",
                "blocked_labels": ["x"],
            }
        )
        assert isinstance(c, RemoteHFClassifierConfig)

    def test_invalid_backend_rejected(self):
        from pydantic import TypeAdapter
        from pydantic import ValidationError as PydanticValidationError

        from nemoguardrails.rails.llm.config import HFClassifierConfig

        ta = TypeAdapter(HFClassifierConfig)
        with pytest.raises(PydanticValidationError, match="backend"):
            ta.validate_python({"backend": "bogus", "model_name": "m", "blocked_labels": ["x"]})

    def test_task_default(self):
        c = _local()
        assert c.task == "text-classification"

    def test_remote_has_no_task(self):
        assert "task" not in RemoteHFClassifierConfig.model_fields
