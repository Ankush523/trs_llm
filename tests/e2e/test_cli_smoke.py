from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CLISmokeTest(unittest.TestCase):
    def test_cli_smoke_pipeline(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root / "src")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_file = tmp / "smoke_math.jsonl"
            traces_file = tmp / "traces.jsonl"
            skills_file = tmp / "skills.jsonl"
            library_dir = tmp / "library"
            direct_dir = tmp / "direct"
            trs_dir = tmp / "trs"
            compare_dir = tmp / "compare"

            commands = [
                [
                    sys.executable,
                    "-m",
                    "trs.cli",
                    "datasets",
                    "prepare",
                    "--dataset",
                    "math_hendrycks",
                    "--source",
                    str(repo_root / "data/raw/smoke/math_hendrycks.jsonl"),
                    "--output",
                    str(dataset_file),
                ],
                [
                    sys.executable,
                    "-m",
                    "trs.cli",
                    "traces",
                    "generate",
                    "--dataset-file",
                    str(dataset_file),
                    "--model-config",
                    str(repo_root / "configs/models/generator.yaml"),
                    "--prompt-config",
                    str(repo_root / "configs/prompts/direct.yaml"),
                    "--output",
                    str(traces_file),
                ],
                [
                    sys.executable,
                    "-m",
                    "trs.cli",
                    "skills",
                    "distill",
                    "--dataset-file",
                    str(dataset_file),
                    "--traces-file",
                    str(traces_file),
                    "--model-config",
                    str(repo_root / "configs/models/summarizer.yaml"),
                    "--prompt-config",
                    str(repo_root / "configs/prompts/skill_extract.yaml"),
                    "--output",
                    str(skills_file),
                ],
                [
                    sys.executable,
                    "-m",
                    "trs.cli",
                    "library",
                    "build",
                    "--cards-file",
                    str(skills_file),
                    "--profile-config",
                    str(repo_root / "configs/retrieval/math_bm25.yaml"),
                    "--output-dir",
                    str(library_dir),
                ],
                [
                    sys.executable,
                    "-m",
                    "trs.cli",
                    "eval",
                    "run",
                    "--mode",
                    "direct",
                    "--dataset-file",
                    str(dataset_file),
                    "--model-config",
                    str(repo_root / "configs/models/inference.yaml"),
                    "--prompt-config",
                    str(repo_root / "configs/prompts/direct.yaml"),
                    "--output-dir",
                    str(direct_dir),
                ],
                [
                    sys.executable,
                    "-m",
                    "trs.cli",
                    "eval",
                    "run",
                    "--mode",
                    "trs",
                    "--dataset-file",
                    str(dataset_file),
                    "--model-config",
                    str(repo_root / "configs/models/inference.yaml"),
                    "--prompt-config",
                    str(repo_root / "configs/prompts/trs_infer.yaml"),
                    "--library-dir",
                    str(library_dir),
                    "--output-dir",
                    str(trs_dir),
                ],
                [
                    sys.executable,
                    "-m",
                    "trs.cli",
                    "compare",
                    "--baseline-summary",
                    str(direct_dir / "summary.json"),
                    "--candidate-summary",
                    str(trs_dir / "summary.json"),
                    "--candidate-results",
                    str(trs_dir / "results.json"),
                    "--output-dir",
                    str(compare_dir),
                ],
            ]
            for command in commands:
                completed = subprocess.run(command, cwd=repo_root, env=env, text=True, capture_output=True, check=False)
                self.assertEqual(completed.returncode, 0, msg=completed.stderr or completed.stdout)

            summary = json.loads((trs_dir / "summary.json").read_text(encoding="utf-8"))
            comparison = json.loads((compare_dir / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["accuracy"], 1.0)
            self.assertTrue((compare_dir / "comparison.md").exists())
            self.assertGreater(comparison["accuracy_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
