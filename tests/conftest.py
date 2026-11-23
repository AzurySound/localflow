from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _ensure_qapplication(qapp):
    """Ensure pytest-qt initializes a QApplication before QCoreApplication-only tests."""
    return qapp
