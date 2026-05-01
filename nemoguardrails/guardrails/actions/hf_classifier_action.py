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

"""HuggingFace classifier rail actions for IORails."""

import logging
from typing import Any, Optional

from nemoguardrails.guardrails.guardrails_types import LLMMessages, RailResult
from nemoguardrails.guardrails.rail_action import RailAction
from nemoguardrails.library.hf_classifier.backends import get_backend

log = logging.getLogger(__name__)

_CLASSIFIER_PREFIX = "$classifier="


def _extract_classifier(flow: str) -> Optional[str]:
    if _CLASSIFIER_PREFIX in flow:
        return flow.split(_CLASSIFIER_PREFIX)[-1].strip()
    return None


def _parse_classify_response(response: dict[str, Any]) -> RailResult:
    blocked = set(response["config"].blocked_labels)
    threshold = response["config"].threshold
    triggered = [
        (r["label"], r["score"]) for r in response["results"] if r["label"] in blocked and r["score"] >= threshold
    ]
    if triggered:
        return RailResult(
            is_safe=False,
            reason=f"HF classifier '{response['classifier']}': {triggered}",
        )
    return RailResult(is_safe=True)


async def _call_classifier(
    task_manager: Any,
    classifier_name: str,
    text: str,
) -> dict[str, Any]:
    classifiers = getattr(task_manager.config.rails.config, "hf_classifier", None)
    if not classifiers or classifier_name not in classifiers:
        raise RuntimeError(
            f"Unknown hf_classifier '{classifier_name}'. Available: {list(classifiers) if classifiers else []}"
        )
    cfg = classifiers[classifier_name]
    backend = get_backend(cfg, name=classifier_name)
    results = await backend.classify(text)
    return {"classifier": classifier_name, "results": results, "config": cfg}


class HFClassifierInputAction(RailAction):
    """Check user input against a HuggingFace classifier (IORails)."""

    action_name = "hf classifier check input"
    requires_model = True

    def _get_model_type(self, flow: str) -> Optional[str]:
        return _extract_classifier(flow) or self.fallback_model

    def _extract_messages(self, messages: LLMMessages, bot_response: Optional[str]) -> dict[str, Any]:
        return {"text": self._last_user_content(messages)}

    def _create_prompt(self, model_type: Optional[str], extracted: dict[str, Any]) -> Any:
        return extracted["text"]

    async def _get_response(self, model_type: Optional[str], prompt: Any) -> Any:
        assert model_type is not None
        return await _call_classifier(self.task_manager, model_type, prompt)

    def _parse_response(self, response: Any) -> RailResult:
        return _parse_classify_response(response)


class HFClassifierOutputAction(RailAction):
    """Check bot output against a HuggingFace classifier (IORails)."""

    action_name = "hf classifier check output"
    requires_model = True

    def _get_model_type(self, flow: str) -> Optional[str]:
        return _extract_classifier(flow) or self.fallback_model

    def _extract_messages(self, messages: LLMMessages, bot_response: Optional[str]) -> dict[str, Any]:
        return {"text": bot_response or ""}

    def _create_prompt(self, model_type: Optional[str], extracted: dict[str, Any]) -> Any:
        return extracted["text"]

    async def _get_response(self, model_type: Optional[str], prompt: Any) -> Any:
        assert model_type is not None
        return await _call_classifier(self.task_manager, model_type, prompt)

    def _parse_response(self, response: Any) -> RailResult:
        return _parse_classify_response(response)
