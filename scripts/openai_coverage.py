#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SPEC_PATH = "fern/openapi.yml"


def _run_oasdiff(subcommand: str, base: str, revision: str, *, extra_flags: list[str] | None = None) -> Any:
    cmd = ["oasdiff", subcommand, base, revision, "--format", "json", "--auto-upgrade", "--flatten-allof"]
    if extra_flags:
        cmd.extend(extra_flags)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and not result.stdout:
        raise RuntimeError(f"oasdiff {subcommand} failed: {result.stderr}")
    return json.loads(result.stdout) if result.stdout else {}


def _check_breaking(spec: str = SPEC_PATH) -> bool:
    if not Path(spec).exists():
        return True
    result = subprocess.run(
        [
            "oasdiff",
            "breaking",
            f"HEAD:{spec}",
            spec,
            "--fail-on",
            "ERR",
            "--auto-upgrade",
            "--flatten-allof",
            "--match-path",
            "^/v1/",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout or result.stderr)
        return False
    return True


def analyze(openai_spec: Path, guardrails_spec: Path, match_path: str | None = None) -> dict[str, Any]:
    import yaml

    flags = ["--strip-prefix-revision", "/v1"]
    if match_path:
        flags.extend(["--match-path", match_path])
    changelog = _run_oasdiff("changelog", str(openai_spec), str(guardrails_spec), extra_flags=flags)

    changes = [
        {k: v for k, v in entry.items() if k not in ("baseSource", "revisionSource", "fingerprint")}
        for entry in (changelog if isinstance(changelog, list) else [])
        if entry.get("section") == "paths"
    ]
    changes.sort(key=lambda e: (e.get("path", ""), e.get("operation", ""), e.get("text", "")))

    spec_text = openai_spec.read_text()
    spec_data = yaml.safe_load(spec_text) if openai_spec.suffix in (".yml", ".yaml") else json.loads(spec_text)

    return {
        "openai_version": spec_data.get("info", {}).get("version", "unknown"),
        "changes": changes,
    }


def _print_report(report: dict[str, Any], prev_changes: int | None) -> None:
    ver = report["openai_version"]
    changes = report["changes"]

    header = f"OpenAI v{ver}: {len(changes)} changes"
    if prev_changes is not None:
        header += f" (baseline: {prev_changes})"
    print(header)

    current_endpoint = ""
    for c in changes:
        endpoint = f"{c.get('operation', '?')} {c.get('path', '?')}"
        if endpoint != current_endpoint:
            current_endpoint = endpoint
            print(f"  {endpoint}:")
        print(f"    {c['text']}")


def main():
    parser = argparse.ArgumentParser(description="OpenAI API conformance analyzer for NeMo Guardrails")
    parser.add_argument("--openai-spec", type=Path, default=Path("schemas/openai-spec.yml"))
    parser.add_argument("--guardrails-spec", type=Path, default=Path("fern/openapi.yml"))
    parser.add_argument("--output", type=Path, default=Path("schemas/openai-conformance-baseline.json"))
    parser.add_argument("--match-path", type=str, default="/chat/completions")
    parser.add_argument("--update", action="store_true", help="Update the coverage baseline file")
    parser.add_argument("--quiet", action="store_true", help="Only output errors")
    parser.add_argument("--check-regression", action="store_true", help="Fail if change count increases")
    parser.add_argument("--check-breaking", action="store_true", help="Fail on breaking API changes vs HEAD")
    args = parser.parse_args()

    if args.check_breaking:
        if not _check_breaking():
            sys.exit(1)

    previous: dict[str, Any] | None = None
    if args.output.exists():
        try:
            previous = json.loads(args.output.read_text())
        except (json.JSONDecodeError, OSError) as e:
            if args.check_regression:
                print(f"Error: could not load previous report: {e}")
                sys.exit(1)

    try:
        report = analyze(args.openai_spec, args.guardrails_spec, match_path=args.match_path)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    n_changes = len(report["changes"])
    prev_changes = len(previous["changes"]) if previous and "changes" in previous else None

    if args.check_regression and prev_changes is not None:
        if n_changes > prev_changes:
            print(f"Coverage regression: {prev_changes} -> {n_changes} changes (+{n_changes - prev_changes})")
            print("To update the baseline: python scripts/openai_coverage.py --update")
            sys.exit(1)
        elif n_changes < prev_changes and not args.quiet:
            print(f"Coverage improved: {prev_changes} -> {n_changes} changes (-{prev_changes - n_changes})")

    if not args.quiet:
        _print_report(report, prev_changes)

    if args.update:
        new_content = json.dumps(report, indent=2) + "\n"
        try:
            old_content = args.output.read_text()
        except FileNotFoundError:
            old_content = ""
        if new_content != old_content:
            args.output.write_text(new_content)


if __name__ == "__main__":
    main()
