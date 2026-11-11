"""Persist Renovate's transitive security fixes as reproducible constraints."""

import json
import os
import re
import shlex
import subprocess
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

BEGIN = "# BEGIN automated transitive security constraints"
END = "# END automated transitive security constraints"
CONSTRAINTS_FILE = Path("dependencies/cve-constraints.txt")
REQUIREMENTS_FILE = Path("requirements.txt")
ADVISORY = re.compile(r"\b(?:CVE-\d{4}-\d{4,}|GHSA-[a-z0-9-]+|PYSEC-\d{4}-\d+|TEST-ONLY-[a-z0-9-]+)\b")
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s;\\]+)", re.M)
COMPILE_COMMAND = re.compile(r"^#\s+(uv pip compile .+)$", re.M)


def merge_constraints(text: str, fixes: dict) -> str:
    records: dict[str, tuple[Version, set[str]]] = {}
    before, after = text, ""
    if BEGIN in text or END in text:
        if text.count(BEGIN) != 1 or text.count(END) != 1:
            raise ValueError("Malformed automated constraints block")

        before, block = text.split(BEGIN)
        block, after = block.split(END)
        references: set[str] = set()
        for line in block.splitlines():
            if not line.strip():
                continue
            if line.startswith("# Advisories: "):
                references = set(line.removeprefix("# Advisories: ").split(", "))
                continue

            match = re.fullmatch(r"([a-z0-9][a-z0-9.-]*)>=([^\s,]+)", line)
            if not match:
                raise ValueError(f"Unexpected content in automated block: {line!r}")

            name, version = match.groups()
            if name in records:
                raise ValueError(f"Duplicate automated constraint: {name}")
            records[name] = (Version(version), references)
            references = set()

    for name, (version, references) in fixes.items():
        old_version, old_references = records.get(name, (version, set()))
        records[name] = (max(old_version, version), old_references | references)

    lines = [BEGIN]
    for name, (version, references) in sorted(records.items()):
        if references:
            lines.append("# Advisories: " + ", ".join(sorted(references)))
        lines.append(f"{name}>={version}")
    lines.append(END)

    return before.rstrip("\n") + "\n\n" + "\n".join(lines) + "\n" + after.lstrip("\n")


def selected_fixes(upgrades: object) -> dict:
    if not isinstance(upgrades, list):
        raise ValueError("Expected a JSON array of Renovate upgrades")

    fixes: dict[str, tuple[Version, set[str]]] = {}
    for upgrade in upgrades:
        dep_types = upgrade.get("depTypes") or []
        if upgrade.get("isVulnerabilityAlert") is not True:
            continue
        if upgrade.get("depType") != "indirect" and "indirect" not in dep_types:
            continue

        name = upgrade.get("depName", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("Invalid package name in security upgrade")

        name = canonicalize_name(name)
        version = Version(upgrade["newVersion"])
        SpecifierSet(f">={version}")

        notes = upgrade.get("prBodyNotes") or []
        if not isinstance(notes, list) or not all(isinstance(note, str) for note in notes):
            raise ValueError("Expected advisory notes as an array of strings")

        references = set(ADVISORY.findall("\n".join(notes)))
        old_version, old_references = fixes.get(name, (version, set()))
        fixes[name] = (max(old_version, version), old_references | references)

    return fixes


def check_resolved(root: Path, fixes: dict) -> None:
    pins: dict[str, list[Version]] = {}
    for package, version in PIN.findall((root / REQUIREMENTS_FILE).read_text()):
        pins.setdefault(canonicalize_name(package), []).append(Version(version))

    for name, (version, _) in fixes.items():
        if name not in pins or any(pin < version for pin in pins[name]):
            raise ValueError(f"The generated requirements do not contain {name}>={version}")


def compile_requirements(root: Path) -> None:
    match = COMPILE_COMMAND.search((root / REQUIREMENTS_FILE).read_text())
    if not match:
        raise ValueError("requirements.txt does not record a uv pip compile command")

    command = shlex.split(match.group(1))
    if tuple(command[:3]) != ("uv", "pip", "compile"):
        raise ValueError("The recorded compile command must start with uv pip compile")
    if "--output-file=requirements.txt" not in command:
        raise ValueError("The recorded compile command must write requirements.txt")

    subprocess.run(command, cwd=root, check=True)


def persist(root: Path, upgrades: object) -> bool:
    fixes = selected_fixes(upgrades)
    if not fixes:
        return False

    check_resolved(root, fixes)
    original = {
        REQUIREMENTS_FILE: (root / REQUIREMENTS_FILE).read_bytes(),
        CONSTRAINTS_FILE: (root / CONSTRAINTS_FILE).read_bytes(),
    }
    updated = merge_constraints(original[CONSTRAINTS_FILE].decode(), fixes)
    if updated.encode() == original[CONSTRAINTS_FILE]:
        return False

    try:
        (root / CONSTRAINTS_FILE).write_text(updated)
        compile_requirements(root)
        check_resolved(root, fixes)
    except BaseException:
        for name, content in original.items():
            (root / name).write_bytes(content)
        raise

    return True


if __name__ == "__main__":
    data_file = Path(os.environ["RENOVATE_POST_UPGRADE_COMMAND_DATA_FILE"])
    changed = persist(Path.cwd(), json.loads(data_file.read_text()))
    print(
        "Persisted transitive security constraints and regenerated requirements"
        if changed
        else "No new transitive security constraints"
    )
