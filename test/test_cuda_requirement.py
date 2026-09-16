"""Behavioral contract for the opt-in CUDA pytest guard."""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
PROBE = Path(__file__).parent / "test_cuda_requirement_probe.py"


def _run_probe(
    tmp_path: Path, cuda_available: bool, *pytest_args: str
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    collected = tmp_path / "collected"
    ran = tmp_path / "ran"
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        "import os\n"
        "import torch\n"
        "from torch.utils import cpp_extension\n"
        "torch.cuda.is_available = lambda: os.environ['TIREX_CUDA_PROBE_AVAILABLE'] == '1'\n"
        "cpp_extension.include_paths = lambda *args, **kwargs: ['C:/cuda/include']\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["TIREX_CUDA_PROBE_AVAILABLE"] = "1" if cuda_available else "0"
    env["TIREX_CUDA_PROBE_COLLECTED"] = str(collected)
    env["TIREX_CUDA_PROBE_RAN"] = str(ran)
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    pythonpath = [str(tmp_path), str(REPO_ROOT)]
    if existing_pythonpath := env.get("PYTHONPATH"):
        pythonpath.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--disable-warnings", *pytest_args, str(PROBE)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    return result, collected, ran


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return f"{result.stdout}\n{result.stderr}"


def test_cpu_pytest_route_does_not_require_cuda(tmp_path):
    result, collected, ran = _run_probe(tmp_path, cuda_available=False)

    assert result.returncode == 0, _output(result)
    assert collected.exists()
    assert ran.exists()


def test_require_cuda_fails_before_collecting_when_cuda_is_unavailable(tmp_path):
    result, collected, ran = _run_probe(tmp_path, False, "--require-cuda")
    output = _output(result).lower()

    assert result.returncode != 0, output
    assert "cuda" in output
    assert any(term in output for term in ("unavailable", "not available", "required")), output
    assert not collected.exists(), output
    assert not ran.exists(), output


def test_require_cuda_allows_collection_when_cuda_is_available(tmp_path):
    result, collected, ran = _run_probe(tmp_path, True, "--require-cuda")

    assert result.returncode == 0, _output(result)
    assert collected.exists()
    assert ran.exists()
