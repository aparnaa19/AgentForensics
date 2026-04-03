"""
Pytest configuration shared by all tests.

Redirects the fingerprint DB to a per-test tmp_path so that fingerprinting
in AlertManager.fire() does not pollute ~/.agentforensics/fingerprints.db
and every test starts with a clean, empty fingerprint store.
"""
import pytest


@pytest.fixture(autouse=True)
def _isolate_fingerprint_db(monkeypatch, tmp_path):
    """Give every test its own empty fingerprint DB."""
    monkeypatch.setenv("AF_FP_DB", str(tmp_path / "fingerprints.db"))
