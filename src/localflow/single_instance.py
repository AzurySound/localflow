"""Single-instance application guard using Qt QLocalServer / QLocalSocket."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket

DEFAULT_SERVER_NAME = "LocalFlow_SingleInstance_v1"


class SingleInstanceGuard(QObject):
    """Ensures only one LocalFlow GUI process runs; wakes the existing one on relaunch."""

    second_instance_activated = Signal()

    def __init__(self, server_name: str = DEFAULT_SERVER_NAME, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._server_name = server_name
        self._server: QLocalServer | None = None

    def try_acquire(self) -> bool:
        """Return True if this process is the primary instance, False if another is already active."""
        probe = QLocalSocket()
        probe.connectToServer(self._server_name)
        if probe.waitForConnected(250):
            probe.write(b"ACTIVATE")
            probe.flush()
            probe.waitForBytesWritten(250)
            probe.disconnectFromServer()
            return False

        # Clean up stale socket left after a crash
        QLocalServer.removeServer(self._server_name)
        server = QLocalServer(self)
        server.newConnection.connect(self._on_new_connection)
        if not server.listen(self._server_name):
            return True  # Fallback: never block launch if IPC cannot bind
        self._server = server
        return True

    def _on_new_connection(self) -> None:
        if self._server is None:
            return
        while self._server.hasPendingConnections():
            sock = self._server.nextPendingConnection()
            if sock is None:
                continue
            sock.waitForReadyRead(100)
            sock.readAll()
            sock.disconnectFromServer()
            sock.deleteLater()
            self.second_instance_activated.emit()

    def release(self) -> None:
        if self._server is not None:
            try:
                self._server.close()
            except Exception:
                pass
            self._server = None
