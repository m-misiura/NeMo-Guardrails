#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pre-commit hook: breaking-change detection + coverage regression check.
# Breaking-change check skipped when commit message contains "!:" or "BREAKING CHANGE:".

set -euo pipefail

SPEC="fern/openapi.yml"

# --- Breaking-change detection ---
if ! { [ -f ".git/COMMIT_EDITMSG" ] && grep -qE '!:|BREAKING CHANGE:' .git/COMMIT_EDITMSG 2>/dev/null; }; then
    if [ -f "$SPEC" ]; then
        BASE_SPEC=$(mktemp)
        trap 'rm -f "$BASE_SPEC"' EXIT
        if git show HEAD:"$SPEC" > "$BASE_SPEC" 2>/dev/null; then
            if ! output=$(oasdiff breaking --fail-on ERR --flatten-allof --match-path '^/v1/' "$BASE_SPEC" "$SPEC" 2>&1); then
                echo "$output"
                exit 1
            fi
        fi
    fi
fi

# --- Coverage regression check ---
uv run --locked python scripts/openai_coverage.py --check-regression --update
