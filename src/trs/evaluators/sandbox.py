from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path


BANNED_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "pathlib",
    "shutil",
    "ctypes",
    "multiprocessing",
}


def _check_safety(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax-error: {exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in BANNED_IMPORTS:
                    return False, f"unsafe-import: {alias.name}"
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".")[0] in BANNED_IMPORTS:
                return False, f"unsafe-import: {module}"
    return True, ""


def run_python_code_with_tests(code: str, tests: list[str], timeout_s: int = 3) -> tuple[bool, str]:
    safe, reason = _check_safety(code)
    if not safe:
        return False, reason
    runner_payload = {
        "code": code,
        "tests": tests,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        runner_path = Path(tmpdir) / "runner.py"
        runner_path.write_text(
            (
                "import json\n"
                "payload = json.loads(open('payload.json', 'r', encoding='utf-8').read())\n"
                "namespace = {}\n"
                "exec(payload['code'], namespace, namespace)\n"
                "for test in payload['tests']:\n"
                "    exec(test, namespace, namespace)\n"
            ),
            encoding="utf-8",
        )
        payload_path = Path(tmpdir) / "payload.json"
        payload_path.write_text(json.dumps(runner_payload), encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(runner_path)],
                cwd=tmpdir,
                text=True,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or "execution-failed"
            return False, message
        return True, "passed"
