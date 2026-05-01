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

"""Pluggable inference backends for HuggingFace classifier rails."""

from __future__ import annotations

import abc
import asyncio
import logging
import os
import ssl
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict, Union

import aiohttp

if TYPE_CHECKING:
    from nemoguardrails.rails.llm.config import HFClassifierConfig

log = logging.getLogger(__name__)


class ClassificationResult(TypedDict):
    label: str
    score: float


class ClassifierBackend(abc.ABC):
    """Abstract interface for HuggingFace classifier inference."""

    @abc.abstractmethod
    async def classify(self, text: str) -> List[ClassificationResult]:
        """Classify a single text. Returns list of label/score detections."""
        ...


_DEFAULT_TIMEOUT = 30.0
_warned_env_vars: set = set()


def _build_headers(config: HFClassifierConfig) -> Dict[str, str]:
    """Build HTTP request headers from classifier config."""
    headers: Dict[str, str] = {"Content-Type": "application/json"}

    if config.api_key_env_var:
        api_key = os.environ.get(config.api_key_env_var)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        elif config.api_key_env_var not in _warned_env_vars:
            _warned_env_vars.add(config.api_key_env_var)
            log.warning(
                "api_key_env_var '%s' is configured but not set in the environment.",
                config.api_key_env_var,
            )

    for name, value in config.parameters.get("default_headers", {}).items():
        existing = next((k for k in headers if k.lower() == name.lower()), None)
        if existing is not None:
            del headers[existing]
        headers[name] = value

    return headers


def _get_timeout(config: HFClassifierConfig) -> aiohttp.ClientTimeout:
    total = config.parameters.get("timeout", _DEFAULT_TIMEOUT)
    return aiohttp.ClientTimeout(total=total)


_ssl_cache: Dict[tuple, Union[ssl.SSLContext, bool, None]] = {}


def _build_ssl_context(
    config: HFClassifierConfig,
) -> Union[ssl.SSLContext, bool, None]:
    """Build SSL context from config parameters (cached).

    Reads from ``config.parameters``:
      - ``verify_ssl`` (bool, default True): set to False to skip TLS verification.
      - ``ca_cert`` (str): path to a CA bundle file for custom/internal CAs.
      - ``client_cert`` (str) + ``client_key`` (str): paths for mTLS client auth.

    Returns:
      - ``None``  — use system defaults (also respects ``SSL_CERT_FILE`` env var).
      - ``False`` — disable TLS verification entirely.
      - ``ssl.SSLContext`` — custom CA and/or client certificate configuration.
    """
    params = config.parameters
    verify = params.get("verify_ssl", True)
    ca_cert: Optional[str] = params.get("ca_cert")
    client_cert: Optional[str] = params.get("client_cert")
    client_key: Optional[str] = params.get("client_key")

    cache_key = (verify, ca_cert, client_cert, client_key)
    if cache_key in _ssl_cache:
        return _ssl_cache[cache_key]

    if verify is False:
        _ssl_cache[cache_key] = False
        return False

    if not ca_cert and not client_cert:
        _ssl_cache[cache_key] = None
        return None

    ctx = ssl.create_default_context(cafile=ca_cert)
    if client_cert:
        ctx.load_cert_chain(certfile=client_cert, keyfile=client_key)
    _ssl_cache[cache_key] = ctx
    return ctx


_pipelines: Dict[str, Any] = {}
_HTTP_ONLY_PARAMS = frozenset({
    "default_headers", "timeout", "verify_ssl", "ca_cert", "client_cert", "client_key",
})


def _get_or_create_pipeline(
    model_name: str,
    task: str,
    aggregation_strategy: Optional[str],
    parameters: Dict[str, Any],
) -> Any:
    cache_key = f"{task}:{model_name}:{aggregation_strategy}"
    if cache_key not in _pipelines:
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "The 'transformers' package is required for the local HF classifier "
                "backend. Install it with: pip install nemoguardrails[hf-classifier]"
            )
        kwargs = {k: v for k, v in parameters.items() if k not in _HTTP_ONLY_PARAMS}
        if task == "token-classification" and aggregation_strategy:
            kwargs["aggregation_strategy"] = aggregation_strategy
        _pipelines[cache_key] = pipeline(task=task, model=model_name, **kwargs)
        log.info("Loaded HF pipeline: task=%s model=%s", task, model_name)
    return _pipelines[cache_key]


class LocalBackend(ClassifierBackend):
    """Local HF Transformers pipeline backend.

    Tested against: ``transformers >= 4.35`` pipeline API for
    ``text-classification`` and ``token-classification`` tasks.
    """

    def __init__(self, config: HFClassifierConfig) -> None:
        self._config = config

    async def classify(self, text: str) -> List[ClassificationResult]:
        pipe = _get_or_create_pipeline(
            self._config.model_name,
            self._config.task,
            self._config.aggregation_strategy,
            self._config.parameters,
        )
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(None, pipe, text)

        results: List[ClassificationResult] = []
        for item in raw:
            if self._config.task == "text-classification":
                results.append(ClassificationResult(label=item["label"], score=item["score"]))
            else:
                label = item.get("entity_group") or item.get("entity", "")
                results.append(ClassificationResult(label=label, score=item["score"]))
        return results


class VLLMBackend(ClassifierBackend):
    """vLLM ``/classify`` endpoint backend.

    Tested against: vLLM v0.6.x ``/classify`` API. Expects response shape::

        {"data": [{"label": "...", "probs": [float, ...]}]}

    Raises ``ValueError`` if the response is missing required keys (``data``,
    ``label``), indicating an API change.
    """

    def __init__(self, config: HFClassifierConfig) -> None:
        self._url = config.endpoint.rstrip("/") + "/classify"
        self._model_name = config.model_name
        self._headers = _build_headers(config)
        self._timeout = _get_timeout(config)
        self._ssl = _build_ssl_context(config)

    async def classify(self, text: str) -> List[ClassificationResult]:
        payload = {"model": self._model_name, "input": text}

        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.post(self._url, json=payload, headers=self._headers, ssl=self._ssl) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise ValueError(f"vLLM /classify returned {resp.status}: {body[:500]}")
                data = await resp.json()

        try:
            items = data["data"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Unexpected vLLM /classify response structure: {exc}. "
                f"Raw: {str(data)[:500]}"
            ) from exc

        results: List[ClassificationResult] = []
        for item in items:
            try:
                label = item["label"]
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f"vLLM /classify item missing 'label': {exc}. "
                    f"Raw item: {str(item)[:200]}"
                ) from exc
            probs = item.get("probs", [])
            results.append(ClassificationResult(
                label=label,
                score=max(probs) if probs else 1.0,
            ))
        return results


class KServeBackend(ClassifierBackend):
    """KServe v1 inference predict backend.

    Tested against: KServe v1 predict API (``/v1/models/{name}:predict``).
    Handles three ``predictions`` shapes:

    - ``dict``: class-index → probability (``--return_probabilities``)
    - ``int/float``: argmax class index, score assumed 1.0
    - nested ``list``: token-level class indices, flattened and deduplicated

    Raises ``ValueError`` if ``predictions`` key is missing or the prediction
    has an unrecognised type.
    """

    def __init__(self, config: HFClassifierConfig) -> None:
        base = config.endpoint.rstrip("/")
        self._url = f"{base}/v1/models/{config.model_name}:predict"
        self._headers = _build_headers(config)
        self._timeout = _get_timeout(config)
        self._ssl = _build_ssl_context(config)

    async def classify(self, text: str) -> List[ClassificationResult]:
        payload: Dict[str, Any] = {"instances": [text]}

        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.post(self._url, json=payload, headers=self._headers, ssl=self._ssl) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise ValueError(f"KServe predict returned {resp.status}: {body[:500]}")
                data = await resp.json()

        try:
            predictions = data["predictions"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Unexpected KServe predict response structure: {exc}. "
                f"Raw: {str(data)[:500]}"
            ) from exc

        if not predictions:
            return []

        pred = predictions[0]
        if isinstance(pred, dict):
            return [
                ClassificationResult(label=cls_idx, score=float(score))
                for cls_idx, score in pred.items()
            ]
        if isinstance(pred, (int, float)):
            return [ClassificationResult(label=str(int(pred)), score=1.0)]
        if isinstance(pred, list):
            flat = _flatten_ints(pred)
            return [
                ClassificationResult(label=str(cls), score=1.0)
                for cls in sorted(set(flat))
                if cls != 0
            ]
        raise ValueError(
            f"Unexpected KServe prediction type: {type(pred).__name__}. "
            f"Raw: {str(data)[:500]}"
        )


def _flatten_ints(nested: Any) -> List[int]:
    if isinstance(nested, (int, float)):
        return [int(nested)]
    out: List[int] = []
    for item in nested:
        out.extend(_flatten_ints(item))
    return out


class FMSBackend(ClassifierBackend):
    """FMS guardrails-detectors ``/api/v1/text/contents`` backend.

    Tested against: FMS guardrails-detectors API v1. Expects response shape::

        [[{"detection_type": "...", "score": float}, ...]]

    Raises ``ValueError`` if the response is not a list-of-lists or detection
    entries lack required keys.
    """

    def __init__(self, config: HFClassifierConfig) -> None:
        self._url = config.endpoint.rstrip("/") + "/api/v1/text/contents"
        self._threshold = config.threshold
        self._headers = _build_headers(config)
        self._timeout = _get_timeout(config)
        self._ssl = _build_ssl_context(config)

    async def classify(self, text: str) -> List[ClassificationResult]:
        payload: Dict[str, Any] = {
            "contents": [text],
            "detector_params": {"threshold": self._threshold},
        }

        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.post(self._url, json=payload, headers=self._headers, ssl=self._ssl) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise ValueError(f"FMS detectors returned {resp.status}: {body[:500]}")
                data = await resp.json()

        if not isinstance(data, list) or not data:
            raise ValueError(
                f"Unexpected FMS response structure (expected list of lists). "
                f"Raw: {str(data)[:500]}"
            )

        if not data[0]:
            return []

        try:
            return [
                ClassificationResult(label=d["detection_type"], score=d["score"])
                for d in data[0]
            ]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Unexpected FMS detection entry structure: {exc}. "
                f"Raw: {str(data)[:500]}"
            ) from exc


_BACKENDS = {
    "local": LocalBackend,
    "vllm": VLLMBackend,
    "kserve": KServeBackend,
    "fms": FMSBackend,
}


def get_backend(config: HFClassifierConfig) -> ClassifierBackend:
    """Create a backend instance from classifier config."""
    cls = _BACKENDS.get(config.backend)
    if cls is None:
        raise ValueError(
            f"Unknown hf_classifier backend: '{config.backend}'. "
            f"Supported: {', '.join(_BACKENDS)}"
        )
    return cls(config)
