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
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SPEC_PATH = "fern/openapi.yml"
OASDIFF_SHARED_FLAGS = ["--auto-upgrade", "--flatten-allof"]


def _require_oasdiff() -> None:
    if not shutil.which("oasdiff"):
        print("Error: oasdiff is not installed or not in PATH")
        print("Install with: go install github.com/oasdiff/oasdiff@v1.23.0")
        sys.exit(1)


def _run_oasdiff(
    subcommand: str,
    base: Path | str,
    revision: Path | str,
    *,
    match_path: str | None = None,
    extra_flags: list[str] | None = None,
) -> dict[str, Any]:
    _require_oasdiff()
    cmd = ["oasdiff", subcommand, str(base), str(revision), "--format", "json"]
    cmd.extend(OASDIFF_SHARED_FLAGS)
    if match_path:
        cmd.extend(["--match-path", match_path])
    if extra_flags:
        cmd.extend(extra_flags)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and not result.stdout:
        raise RuntimeError(f"oasdiff {subcommand} failed: {result.stderr}")
    return json.loads(result.stdout) if result.stdout else {}


def _check_breaking(spec: str = SPEC_PATH) -> bool:
    commit_msg = Path(".git/COMMIT_EDITMSG")
    if commit_msg.exists():
        try:
            if re.search(r"!:|BREAKING CHANGE:", commit_msg.read_text()):
                return True
        except OSError:
            pass

    if not Path(spec).exists():
        return True

    _require_oasdiff()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=True) as tmp:
        git_result = subprocess.run(["git", "show", f"HEAD:{spec}"], capture_output=True, text=True)
        if git_result.returncode != 0:
            return True
        tmp.write(git_result.stdout)
        tmp.flush()

        result = subprocess.run(
            ["oasdiff", "breaking", "--fail-on", "ERR", *OASDIFF_SHARED_FLAGS, "--match-path", "^/v1/", tmp.name, spec],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(result.stdout or result.stderr)
            return False
    return True


def _extract_endpoint_gaps(paths_diff: dict[str, Any]) -> dict[str, Any]:
    """Extract missing/modified properties per implemented endpoint from oasdiff diff output."""
    endpoints: dict[str, Any] = {}
    for path, path_data in paths_diff.get("modified", {}).items():
        ops = path_data.get("operations", {}).get("modified", {})
        for method, op_data in ops.items():
            key = f"{method} {path}"
            missing: list[str] = []
            modified: list[str] = []

            req = op_data.get("requestBody", {}).get("content", {}).get("modified", {})
            for content_type, ct_data in req.items():
                props = ct_data.get("schema", {}).get("properties", {})
                missing.extend(sorted(props.get("deleted", [])))
                modified.extend(sorted(props.get("modified", {}).keys()))

            resp = op_data.get("responses", {}).get("modified", {})
            for _status, status_data in resp.items():
                for content_type, ct_data in status_data.get("content", {}).get("modified", {}).items():
                    props = ct_data.get("schema", {}).get("properties", {})
                    resp_missing = sorted(props.get("deleted", []))
                    resp_modified = sorted(props.get("modified", {}).keys())
                    if content_type != "application/json":
                        missing.extend(f"{content_type}:{p}" for p in resp_missing)
                        modified.extend(f"{content_type}:{p}" for p in resp_modified)
                    else:
                        missing.extend(resp_missing)
                        modified.extend(resp_modified)

            endpoints[key] = {"missing": missing, "modified": modified}
    return endpoints


def analyze(openai_spec: Path, guardrails_spec: Path, match_path: str | None = None) -> dict[str, Any]:
    import yaml

    strip = ["--strip-prefix-revision", "/v1"]
    diff = _run_oasdiff("diff", openai_spec, guardrails_spec, match_path=match_path, extra_flags=strip)
    paths_diff = diff.get("paths", {})

    spec_text = openai_spec.read_text()
    spec_data = yaml.safe_load(spec_text) if openai_spec.suffix in (".yml", ".yaml") else json.loads(spec_text)

    return {
        "openai_version": spec_data.get("info", {}).get("version", "unknown"),
        "missing_paths": sorted(paths_diff.get("deleted", [])),
        "endpoints": _extract_endpoint_gaps(paths_diff),
        "diff": paths_diff,
    }


def main():
    parser = argparse.ArgumentParser(description="OpenAI API conformance analyzer for NeMo Guardrails")
    parser.add_argument("--openai-spec", type=Path, default=Path("schemas/openai-spec.yml"))
    parser.add_argument("--guardrails-spec", type=Path, default=Path("fern/openapi.yml"))
    parser.add_argument("--output", type=Path, default=Path("schemas/openai-coverage.json"))
    parser.add_argument("--match-path", type=str, default="/chat/completions")
    parser.add_argument("--update", action="store_true", help="Update the coverage baseline file")
    parser.add_argument("--quiet", action="store_true", help="Only output errors")
    parser.add_argument("--check-regression", action="store_true", help="Fail if missing property count increases")
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

    total_missing = sum(len(ep["missing"]) for ep in report["endpoints"].values())
    total_modified = sum(len(ep["modified"]) for ep in report["endpoints"].values())

    prev_missing: int | None = None
    if previous:
        prev_missing = sum(len(ep.get("missing", [])) for ep in previous.get("endpoints", {}).values())

    if args.check_regression and prev_missing is not None:
        if total_missing > prev_missing:
            print(f"Coverage regression: {prev_missing} -> {total_missing} missing properties (+{total_missing - prev_missing})")
            print("To update the baseline: python scripts/openai_coverage.py --update")
            sys.exit(1)
        elif total_missing < prev_missing and not args.quiet:
            print(f"Coverage improved: {prev_missing} -> {total_missing} missing properties (-{prev_missing - total_missing})")

    if not args.quiet:
        ver = report["openai_version"]
        for key, ep in report["endpoints"].items():
            n_missing = len(ep["missing"])
            n_modified = len(ep["modified"])
            line = f"{key} (OpenAI v{ver}): {n_missing} missing, {n_modified} modified"
            if prev_missing is not None:
                line += f" (baseline: {prev_missing} missing)"
            print(line)
            if ep["missing"]:
                print(f"  Missing: {', '.join(ep['missing'])}")
            if ep["modified"]:
                print(f"  Modified: {', '.join(ep['modified'])}")
        if report["missing_paths"]:
            print(f"  Unimplemented paths: {', '.join(report['missing_paths'])}")

    if args.update:
        stable = {k: v for k, v in report.items() if k != "diff"}
        try:
            old_stable = {k: v for k, v in json.loads(args.output.read_text()).items() if k != "diff"}
        except (FileNotFoundError, json.JSONDecodeError):
            old_stable = None
        if stable != old_stable:
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
