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

"""HuggingFace classifier-based detection actions."""

import json
import logging
from typing import Any, List, Optional, Tuple

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.library.hf_classifier.backends import get_backend

log = logging.getLogger(__name__)


async def _classify_and_check(
    classifier_name: str,
    text: str,
    config: Optional[RailsConfig],
) -> bool:
    """Classify *text* and check against blocked labels.

    Returns ``True`` if allowed, ``False`` if blocked.
    """
    classifiers = getattr(config.rails.config, "hf_classifier", None) if config else None
    if not classifiers:
        raise ValueError(
            "hf_classifier action called but no 'hf_classifier' section found in "
            "rails.config. Check your config.yml for typos."
        )

    classifier_config = classifiers.get(classifier_name)
    if classifier_config is None:
        raise ValueError(f"Unknown classifier '{classifier_name}'. Available: {list(classifiers)}")

    backend = get_backend(classifier_config)
    results = await backend.classify(text)

    if text and not results:
        log.warning(
            "HF classifier '%s' returned no results for non-empty input — "
            "possible API compatibility issue with the '%s' backend.",
            classifier_name,
            classifier_config.backend,
        )

    blocked = set(classifier_config.blocked_labels)
    threshold = classifier_config.threshold

    triggered: List[Tuple[str, float]] = [
        (r["label"], r["score"]) for r in results if r["label"] in blocked and r["score"] >= threshold
    ]

    if triggered:
        log.info(
            "HF classifier '%s': blocked (detections: %s)",
            classifier_name,
            triggered,
        )
        return False

    log.info("HF classifier '%s': allowed", classifier_name)
    return True


@action(is_system_action=True)
async def hf_classifier_check_input(
    classifier: str,
    config: Optional[RailsConfig] = None,
    context: Optional[dict] = None,
    **kwargs,
) -> bool:
    """Check user input against a HuggingFace classifier."""
    text = context.get("user_message", "") if context else ""
    return await _classify_and_check(classifier, text, config)


@action(is_system_action=True)
async def hf_classifier_check_output(
    classifier: str,
    config: Optional[RailsConfig] = None,
    context: Optional[dict] = None,
    **kwargs,
) -> bool:
    """Check bot output against a HuggingFace classifier."""
    text = context.get("bot_message", "") if context else ""
    return await _classify_and_check(classifier, text, config)


@action(is_system_action=True)
async def hf_classifier_check_retrieval(
    classifier: str,
    config: Optional[RailsConfig] = None,
    context: Optional[dict] = None,
    **kwargs,
) -> bool:
    """Check retrieved chunks against a HuggingFace classifier."""
    text = context.get("relevant_chunks", "") if context else ""
    return await _classify_and_check(classifier, text, config)


@action(is_system_action=True)
async def hf_classifier_check_tool_input(
    classifier: str,
    config: Optional[RailsConfig] = None,
    context: Optional[dict] = None,
    **kwargs,
) -> bool:
    """Check tool input against a HuggingFace classifier."""
    text = context.get("tool_message", "") if context else ""
    return await _classify_and_check(classifier, text, config)


@action(is_system_action=True)
async def hf_classifier_check_tool_output(
    classifier: str,
    tool_calls: Optional[Any] = None,
    config: Optional[RailsConfig] = None,
    context: Optional[dict] = None,
    **kwargs,
) -> bool:
    """Check tool output against a HuggingFace classifier."""
    calls = tool_calls or (context.get("tool_calls", []) if context else [])
    text = json.dumps(calls) if calls else ""
    return await _classify_and_check(classifier, text, config)
