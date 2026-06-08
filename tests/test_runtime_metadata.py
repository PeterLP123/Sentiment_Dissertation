from sentiment_benchmark import runtime_metadata
from sentiment_benchmark.runtime_metadata import collect_run_environment, machine_identity


def test_machine_identity_prefers_configured_id(monkeypatch) -> None:
    monkeypatch.setenv("SENTIMENT_BENCH_MACHINE_ID", "Desktop 3070")
    monkeypatch.setenv("SENTIMENT_BENCH_MACHINE_LABEL", "office rig")

    identity = machine_identity()

    assert identity == {
        "id": "Desktop-3070",
        "label": "office rig",
        "id_source": "env:SENTIMENT_BENCH_MACHINE_ID",
    }


def test_collect_run_environment_includes_reproducibility_context(monkeypatch) -> None:
    monkeypatch.setenv("SENTIMENT_BENCH_MACHINE_ID", "test-machine")
    monkeypatch.setenv("SENTIMENT_BENCH_DB_BACKEND", "sqlite")

    environment = collect_run_environment()

    assert environment["machine"]["id"] == "test-machine"
    assert environment["package_version"] == "0.1.0"
    assert environment["database"]["backend"] == "sqlite"
    assert environment["python"]["version"]
    assert "git" in environment
    assert "timezone" in environment


def test_machine_identity_hashes_hostname_when_no_configured_id(monkeypatch) -> None:
    monkeypatch.delenv("SENTIMENT_BENCH_MACHINE_ID", raising=False)
    monkeypatch.setattr(runtime_metadata, "_windows_machine_guid", lambda: None)
    monkeypatch.setattr(runtime_metadata, "_linux_machine_id", lambda: None)
    monkeypatch.setattr(runtime_metadata.platform, "node", lambda: "research-box")

    identity = machine_identity()

    assert identity["id"] != "None"
    assert identity["id"].startswith("machine-")
    assert identity["label"] == "test-host"
    assert identity["id_source"] == "hostname_sha256"
