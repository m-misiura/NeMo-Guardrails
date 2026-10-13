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

"""Populate HuggingFace-convention cache directories from modelcar-sourced files.

Each modelcar's /models/ tree is staged at MODELCAR_DIR/{name}/ by
COPY --from instructions in the Dockerfile.  This script creates the
refs/main -> snapshot directory mapping that HF libraries expect.

Adding a new model:
  1. Add a COPY --from=modelcar-xxx line in the Dockerfile
  2. Add an install_hf_model call in main() below
  3. If the modelcar bundles extra formats, add a prune_modelcar call
"""

import os
import shutil
import sys
from pathlib import Path

MODELCAR_DIR = Path(os.environ.get("MODELCAR_DIR", "/tmp/modelcars"))
SNAPSHOT_ID = "modelcar"

PRUNE_FILES = [
    "pytorch_model.bin",
    "tf_model.h5",
    "rust_model.ot",
    "inference.py",
    "train_script.py",
    "requirements.txt",
    "README.md",
    "modelcard.md",
    "config.yaml",
    "burn_scars_config.yaml",
]

PRUNE_DIRS = ["openvino", "examples", "splits"]


def _dir_size_mb(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) // (1024 * 1024)


def prune_modelcar(src: Path) -> None:
    if not src.is_dir():
        return

    before = _dir_size_mb(src)

    for name in PRUNE_FILES:
        (src / name).unlink(missing_ok=True)

    for p in src.glob("*.pt"):
        p.unlink()

    for name in PRUNE_DIRS:
        shutil.rmtree(src / name, ignore_errors=True)

    onnx_dir = src / "onnx"
    if onnx_dir.is_dir():
        for p in onnx_dir.iterdir():
            if p.is_file() and p.name != "model.onnx":
                p.unlink()

    after = _dir_size_mb(src)
    print(f"Pruned {src}: {before}MB -> {after}MB")


def install_hf_model(src_dir: Path, model_id: str, cache_base: str, *, use_hub: bool = False) -> None:
    if not src_dir.is_dir():
        sys.exit(f"FATAL: source {src_dir} not found")

    org, repo = model_id.split("/", 1)
    cache_root = Path(cache_base) / "hub" if use_hub else Path(cache_base)

    model_dir = cache_root / f"models--{org}--{repo}"
    snapshot_dir = model_dir / "snapshots" / SNAPSHOT_ID

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "refs").mkdir(parents=True, exist_ok=True)
    (model_dir / "refs" / "main").write_text(SNAPSHOT_ID)

    shutil.copytree(src_dir, snapshot_dir, dirs_exist_ok=True)
    print(f"Installed {model_id} -> {cache_root}")


def install_nltk_data(src_dir: Path) -> None:
    if not src_dir.is_dir():
        sys.exit(f"FATAL: source {src_dir} not found")

    dest = Path(os.environ["NLTK_DATA"]) / "tokenizers" / "punkt_tab"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_dir, dest, dirs_exist_ok=True)
    print(f"Installed NLTK punkt_tab -> {dest}")


def main() -> None:
    print("Setting up models from modelcars...")

    prune_modelcar(MODELCAR_DIR / "minilm")
    prune_modelcar(MODELCAR_DIR / "snowflake")

    install_hf_model(
        MODELCAR_DIR / "minilm",
        "sentence-transformers/all-MiniLM-L6-v2",
        os.environ["HF_HOME"],
        use_hub=True,
    )
    install_hf_model(
        MODELCAR_DIR / "minilm",
        "sentence-transformers/all-MiniLM-L6-v2",
        os.environ["SENTENCE_TRANSFORMERS_HOME"],
    )
    install_hf_model(
        MODELCAR_DIR / "minilm",
        "qdrant/all-MiniLM-L6-v2-onnx",
        os.environ["FASTEMBED_CACHE_PATH"],
    )

    install_hf_model(
        MODELCAR_DIR / "snowflake",
        "RedHatAI/snowflake-arctic-embed-m-long",
        os.environ["HF_HOME"],
        use_hub=True,
    )

    install_nltk_data(MODELCAR_DIR / "nltk")

    print("Model setup complete")


if __name__ == "__main__":
    main()
