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
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_spec(spec_path: Path) -> dict[str, Any]:
    """Load an OpenAPI spec from YAML or JSON."""
    import yaml

    content = spec_path.read_text()
    if spec_path.suffix in (".yml", ".yaml"):
        return yaml.safe_load(content)
    return json.loads(content)


def _check_oasdiff_installed() -> bool:
    return shutil.which("oasdiff") is not None


def _run_oasdiff(openai_spec: Path, guardrails_spec: Path, match_path: str | None = None) -> dict[str, Any]:
    """Run oasdiff diff and return JSON output."""
    if not _check_oasdiff_installed():
        print("Error: oasdiff is not installed or not in PATH")
        print("Install with: go install github.com/oasdiff/oasdiff@v1.23.0")
        print("Or: brew install oasdiff")
        sys.exit(1)

    cmd = [
        "oasdiff",
        "diff",
        str(openai_spec),
        str(guardrails_spec),
        "--format",
        "json",
        "--strip-prefix-revision",
        "/v1",
        "--auto-upgrade",
        "--flatten-allof",
    ]
    if match_path:
        cmd.extend(["--match-path", match_path])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 and not result.stdout:
        raise RuntimeError(f"oasdiff failed: {result.stderr}")

    return json.loads(result.stdout) if result.stdout else {}


def _count_schema_properties(schema: dict[str, Any], spec: dict[str, Any], visited: set[str] | None = None) -> int:
    """Recursively count properties in a schema, resolving $ref."""
    if visited is None:
        visited = set()
    if not isinstance(schema, dict):
        return 0

    count = 0

    if "$ref" in schema:
        ref = schema["$ref"]
        if ref in visited:
            return 0
        visited.add(ref)
        if ref.startswith("#/"):
            parts = ref[2:].split("/")
            resolved = spec
            for part in parts:
                resolved = resolved.get(part, {})
            count += _count_schema_properties(resolved, spec, visited)
        return count

    if "properties" in schema:
        props = schema["properties"]
        count += len(props)
        for prop_schema in props.values():
            count += _count_schema_properties(prop_schema, spec, visited)

    for key in ("allOf", "oneOf", "anyOf"):
        if key in schema:
            for sub_schema in schema[key]:
                count += _count_schema_properties(sub_schema, spec, visited)

    if "items" in schema:
        count += _count_schema_properties(schema["items"], spec, visited)

    if "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
        count += _count_schema_properties(schema["additionalProperties"], spec, visited)

    return count


def _count_endpoint_properties(spec: dict[str, Any], paths: list[str]) -> int:
    """Count total properties for the given endpoint paths."""
    total = 0
    for path in paths:
        path_item = spec.get("paths", {}).get(path, {})
        if not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            if method in ("parameters", "servers", "summary", "description"):
                continue
            if not isinstance(operation, dict):
                continue
            request_body = operation.get("requestBody", {})
            if isinstance(request_body, dict):
                content = request_body.get("content", {})
                json_content = content.get("application/json", {})
                if isinstance(json_content, dict) and "schema" in json_content:
                    total += _count_schema_properties(json_content["schema"], spec)
            responses = operation.get("responses", {})
            for response in responses.values():
                if isinstance(response, dict):
                    content = response.get("content", {})
                    json_content = content.get("application/json", {})
                    if isinstance(json_content, dict) and "schema" in json_content:
                        total += _count_schema_properties(json_content["schema"], spec)
            params = operation.get("parameters", [])
            total += len(params)
    return total


def _extract_issues(obj: Any, path: str = "") -> dict[str, Any]:
    """Recursively extract issues from oasdiff output."""
    result: dict[str, Any] = {"missing": [], "issues": []}
    if not isinstance(obj, dict):
        return result

    if "deleted" in obj:
        deleted = obj["deleted"]
        if isinstance(deleted, list):
            for item in deleted:
                if isinstance(item, str):
                    prop_path = f"{path}.{item}" if path else item
                    prop_path = prop_path.replace(".modified.", ".")
                    result["missing"].append(prop_path)
        elif isinstance(deleted, dict):
            for location, items in deleted.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, str):
                            prop_path = f"{path}.{location}.{item}" if path else f"{location}.{item}"
                            prop_path = prop_path.replace(".modified.", ".")
                            result["missing"].append(prop_path)

    if "modified" in obj and isinstance(obj["modified"], dict):
        for prop_name, prop_diff in obj["modified"].items():
            if not isinstance(prop_diff, dict):
                continue
            schema_indicators = {"enum", "type", "listOfTypes", "anyOf", "oneOf", "default", "schema"}
            if not any(key in prop_diff for key in schema_indicators):
                nested_path = f"{path}.{prop_name}" if path else prop_name
                nested = _extract_issues({"modified": prop_diff}, nested_path)
                result["missing"].extend(nested["missing"])
                result["issues"].extend(nested["issues"])
                continue

            clean_name = prop_name.replace(".modified.", ".")
            prop_path = f"{path}.{clean_name}" if path else clean_name
            prop_path = prop_path.replace(".modified.", ".")
            issue_details = []

            if "enum" in prop_diff:
                enum_diff = prop_diff["enum"]
                if enum_diff.get("enumDeleted"):
                    issue_details.append(f"Enum removed: {enum_diff.get('deleted', [])}")
                elif "deleted" in enum_diff:
                    issue_details.append(f"Enum values removed: {enum_diff['deleted']}")

            if "type" in prop_diff:
                type_diff = prop_diff["type"]
                if "deleted" in type_diff:
                    issue_details.append(f"Type removed: {type_diff['deleted']}")
                if "added" in type_diff:
                    issue_details.append(f"Type added: {type_diff['added']}")

            if "listOfTypes" in prop_diff:
                lot_diff = prop_diff["listOfTypes"]
                if "added" in lot_diff and "null" in lot_diff["added"]:
                    issue_details.append("Nullable added (OpenAI non-nullable)")
                if "deleted" in lot_diff and "null" in lot_diff["deleted"]:
                    issue_details.append("Nullable removed (OpenAI nullable)")

            if "anyOf" in prop_diff or "oneOf" in prop_diff:
                union_diff = prop_diff.get("anyOf", prop_diff.get("oneOf", {}))
                if "added" in union_diff:
                    issue_details.append(f"Union variants added: {len(union_diff['added'])}")
                if "deleted" in union_diff:
                    issue_details.append(f"Union variants removed: {len(union_diff['deleted'])}")

            if "default" in prop_diff:
                default_diff = prop_diff["default"]
                if isinstance(default_diff, dict) and "from" in default_diff:
                    issue_details.append(f"Default changed: {default_diff['from']} -> {default_diff.get('to')}")

            if issue_details:
                result["issues"].append({"property": prop_path, "details": issue_details})

            nested = _extract_issues(prop_diff, prop_path)
            result["missing"].extend(nested["missing"])
            result["issues"].extend(nested["issues"])

    for key in ["schema", "items", "properties", "content", "responses", "parameters", "requestBody"]:
        if key in obj and isinstance(obj[key], dict):
            nested_path = f"{path}.{key}" if path and key not in ("schema", "items") else path
            nested = _extract_issues(obj[key], nested_path)
            result["missing"].extend(nested["missing"])
            result["issues"].extend(nested["issues"])

    return result


def _load_endpoint_categories(openai_spec: Path) -> dict[str, list[str]]:
    """Extract endpoint categories from OpenAI spec tags."""
    spec = _load_spec(openai_spec)
    categories: dict[str, list[str]] = defaultdict(list)
    for path, path_item in (spec.get("paths") or {}).items():
        if not isinstance(path_item, dict):
            continue
        for _, operation in path_item.items():
            if not isinstance(operation, dict):
                continue
            tags = operation.get("tags", [])
            if tags:
                category = tags[0]
                if path not in categories[category]:
                    categories[category].append(path)
    return dict(categories)


def _categorize_path(path: str, categories: dict[str, list[str]]) -> str:
    for category, paths in categories.items():
        for cat_path in paths:
            if path == cat_path or path.startswith(cat_path.rstrip("/") + "/"):
                return category
    return "Other"


def calculate_coverage(openai_spec: Path, guardrails_spec: Path, match_path: str | None = None) -> dict[str, Any]:
    """Calculate coverage metrics."""
    endpoint_categories = _load_endpoint_categories(openai_spec)
    diff = _run_oasdiff(openai_spec, guardrails_spec, match_path=match_path)

    paths_diff = diff.get("paths", {})
    deleted_paths = paths_diff.get("deleted", [])
    modified_paths = paths_diff.get("modified", {})

    missing_endpoints = sorted([p for p in deleted_paths if _categorize_path(p, endpoint_categories) != "Other"])
    implemented_endpoints = sorted(
        [p for p in modified_paths.keys() if _categorize_path(p, endpoint_categories) != "Other"]
    )

    categories: dict[str, dict[str, Any]] = {}

    for path in implemented_endpoints:
        category = _categorize_path(path, endpoint_categories)
        if category not in categories:
            categories[category] = {"endpoints": [], "total_issues": 0, "total_missing": 0}

        endpoint_diff = modified_paths[path]
        operations = endpoint_diff.get("operations", {}).get("modified", {})
        endpoint_data: dict[str, Any] = {"path": path, "operations": []}

        for method in sorted(operations.keys()):
            op_diff = operations[method]
            issues_data = _extract_issues(op_diff, method)
            sorted_missing = sorted(issues_data["missing"])
            sorted_issues = sorted(issues_data["issues"], key=lambda x: x["property"])

            endpoint_data["operations"].append(
                {
                    "method": method,
                    "missing_properties": sorted_missing,
                    "conformance_issues": sorted_issues,
                    "missing_count": len(sorted_missing),
                    "issues_count": len(sorted_issues),
                }
            )
            categories[category]["total_issues"] += len(sorted_issues)
            categories[category]["total_missing"] += len(sorted_missing)

        categories[category]["endpoints"].append(endpoint_data)

    total_issues = sum(c["total_issues"] for c in categories.values())
    total_missing = sum(c["total_missing"] for c in categories.values())
    total_endpoints = len(implemented_endpoints) + len(missing_endpoints)

    openai_spec_data = _load_spec(openai_spec)
    total_properties = 0
    category_properties: dict[str, int] = {}

    for cat_name, cat_data in categories.items():
        cat_paths = [ep["path"] for ep in cat_data["endpoints"]]
        cat_props = _count_endpoint_properties(openai_spec_data, cat_paths)
        cat_problems = cat_data["total_issues"] + cat_data["total_missing"]
        category_properties[cat_name] = max(cat_props, cat_problems)
        total_properties += category_properties[cat_name]

    total_problems = total_issues + total_missing
    total_properties = max(total_properties, total_problems)

    if total_properties > 0:
        overall_score = round((1 - total_problems / total_properties) * 100, 1)
    else:
        overall_score = 100.0 if total_problems == 0 else 0.0

    openai_version = openai_spec_data.get("info", {}).get("version", "unknown")

    report: dict[str, Any] = {
        "openai_spec": str(openai_spec),
        "openai_version": openai_version,
        "guardrails_spec": str(guardrails_spec),
        "summary": {
            "endpoints": {
                "implemented": len(implemented_endpoints),
                "total": total_endpoints,
                "missing": missing_endpoints,
            },
            "conformance": {
                "score": overall_score,
                "issues": total_issues,
                "missing_properties": total_missing,
                "total_problems": total_problems,
                "total_properties": total_properties,
            },
        },
        "categories": {},
    }

    for cat_name in sorted(categories.keys()):
        cat_data = categories[cat_name]
        cat_problems = cat_data["total_issues"] + cat_data["total_missing"]
        cat_total = category_properties.get(cat_name, cat_problems)
        if cat_total > 0:
            cat_score = round((1 - cat_problems / cat_total) * 100, 1)
        else:
            cat_score = 100.0 if cat_problems == 0 else 0.0

        report["categories"][cat_name] = {
            "score": cat_score,
            "issues": cat_data["total_issues"],
            "missing_properties": cat_data["total_missing"],
            "total_properties": cat_total,
            "endpoints": sorted(cat_data["endpoints"], key=lambda x: x["path"]),
        }

    return report


def print_summary(report: dict[str, Any], baseline_score: float | None = None, output_path: Path | None = None) -> None:
    """Print a human-readable summary."""
    summary = report["summary"]
    conf = summary["conformance"]
    ep = summary["endpoints"]
    openai_version = report.get("openai_version", "unknown")

    score_line = f"OpenAI API v{openai_version}: {conf['score']}%"
    if baseline_score is not None:
        delta = conf["score"] - baseline_score
        if delta != 0:
            sign = "+" if delta > 0 else ""
            score_line += f" ({sign}{delta:.1f} from {baseline_score}% baseline)"
        else:
            score_line += f" (baseline: {baseline_score}%)"
    score_line += f" — {ep['implemented']}/{ep['total']} endpoints, {conf['total_problems']} gaps"
    print(score_line)
    if ep["missing"]:
        print(f"  Not implemented: {', '.join(ep['missing'])}")
    if output_path:
        print(f"  Details: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="OpenAI API conformance analyzer for NeMo Guardrails")
    parser.add_argument(
        "--openai-spec",
        type=Path,
        default=Path("schemas/openai-spec.yml"),
        help="Path to vendored OpenAI spec",
    )
    parser.add_argument(
        "--guardrails-spec",
        type=Path,
        default=Path("fern/openapi.yml"),
        help="Path to NeMo Guardrails OpenAPI spec",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("schemas/openai-coverage.json"),
        help="Output path for coverage JSON",
    )
    parser.add_argument(
        "--match-path",
        type=str,
        default="/chat/completions",
        help="Regex to filter OpenAI spec paths (default: /chat/completions)",
    )
    parser.add_argument("--update", action="store_true", help="Update the coverage file")
    parser.add_argument("--quiet", action="store_true", help="Only output errors")
    parser.add_argument(
        "--check-regression",
        action="store_true",
        help="Fail if coverage score decreases compared to existing report",
    )

    args = parser.parse_args()

    previous_score: float | None = None
    if args.output.exists():
        try:
            with open(args.output) as f:
                previous_report = json.load(f)
                previous_score = previous_report.get("summary", {}).get("conformance", {}).get("score")
        except (json.JSONDecodeError, OSError) as e:
            if args.check_regression:
                print(f"Error: could not load previous report: {e}")
                sys.exit(1)

    try:
        report = calculate_coverage(args.openai_spec, args.guardrails_spec, match_path=args.match_path)
    except FileNotFoundError:
        print("Error: oasdiff not found. Install with: go install github.com/oasdiff/oasdiff@latest")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    new_score = report["summary"]["conformance"]["score"]

    if args.check_regression and previous_score is not None:
        if new_score < previous_score:
            print(f"Coverage regression detected: {previous_score}% -> {new_score}%")
            print(f"Coverage decreased by {previous_score - new_score:.1f} percentage points")
            print()
            print("To fix this, ensure your changes don't reduce OpenAI API conformance.")
            print("If this is intentional, update the coverage baseline with:")
            print(f"  python {__file__} --update")
            sys.exit(1)
        elif new_score > previous_score and not args.quiet:
            print(f"Coverage improved: {previous_score}% -> {new_score}% (+{new_score - previous_score:.1f}%)")

    if not args.quiet:
        print_summary(report, baseline_score=previous_score, output_path=args.output)

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
