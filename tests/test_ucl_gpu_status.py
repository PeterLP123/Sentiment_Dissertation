import runpy
from pathlib import Path

import pytest

_CHECKER = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "ucl_gpu_status.py"))
LAB_HOSTS = _CHECKER["LAB_HOSTS"]
classify_ssh_failure = _CHECKER["classify_ssh_failure"]
parse_nvidia_smi = _CHECKER["parse_nvidia_smi"]
ssh_command = _CHECKER["ssh_command"]


def test_inventory_matches_published_host_lists() -> None:
    # The page describes 25 lab105 PCs but currently publishes 23 hostnames.
    assert len(LAB_HOSTS["lab105"]) == 23
    # It describes 30 lab121 PCs but currently publishes 31 hostnames.
    assert len(LAB_HOSTS["lab121"]) == 31
    assert len(set(LAB_HOSTS["lab105"] + LAB_HOSTS["lab121"])) == 54


def test_parse_free_gpu() -> None:
    output = """__UCL_GPU_LIST__
0, GPU-aaaa, NVIDIA GeForce RTX 3090, 24576, 12, 0
__UCL_GPU_PROCESSES__
__UCL_GPU_END__
"""

    (gpu,) = parse_nvidia_smi(output)

    assert gpu.name == "NVIDIA GeForce RTX 3090"
    assert gpu.compute_processes == 0
    assert gpu.compute_memory_mib is None


def test_parse_taken_gpu_and_sum_compute_memory() -> None:
    output = """__UCL_GPU_LIST__
0, GPU-aaaa, NVIDIA GeForce RTX 3090, 24576, 4096, 45
__UCL_GPU_PROCESSES__
GPU-aaaa, 1234, 1024
GPU-aaaa, 5678, 2048
__UCL_GPU_END__
"""

    (gpu,) = parse_nvidia_smi(output)

    assert gpu.compute_processes == 2
    assert gpu.compute_memory_mib == 3072


def test_parse_rejects_unstructured_output() -> None:
    with pytest.raises(ValueError, match="markers are missing"):
        parse_nvidia_smi("NVIDIA-SMI has failed")


def test_parse_reports_driver_or_device_error() -> None:
    output = """__UCL_GPU_LIST__
No devices were found
__UCL_GPU_PROCESSES__
__UCL_GPU_END__
"""

    with pytest.raises(ValueError, match="no usable GPU"):
        parse_nvidia_smi(output)


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        ("Permission denied (publickey,password).", "AUTH"),
        ("Host key verification failed.", "HOST_KEY"),
        ("ssh: connect to host x port 22: Connection timed out", "OFFLINE"),
        ("channel 0: open failed: administratively prohibited: open failed", "OFFLINE"),
        ("bash: nvidia-smi: command not found", "ERROR"),
    ],
)
def test_classify_ssh_failure(stderr: str, expected: str) -> None:
    assert classify_ssh_failure(stderr)[0] == expected


def test_ssh_is_non_interactive_and_uses_jump_host() -> None:
    command = ssh_command("canada-l", "alice", "ucl-knuckles", 7)

    assert "BatchMode=yes" in command
    assert "StrictHostKeyChecking=accept-new" in command
    assert command[command.index("-J") + 1] == "ucl-knuckles"
    assert "alice@canada-l.cs.ucl.ac.uk" in command
