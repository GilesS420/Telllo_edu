"""
UDP/JSON link between the drone laptop and the Jetson.

Every datagram is one JSON object with a "type" field. See README for the
full protocol.
"""

import json
import socket
import threading


class JetsonLink:
    def __init__(self, on_message, listen_host, listen_port, jetson_host=None, jetson_port=9001):
        """
        Args:
            on_message: callback(dict) called (in the receive thread) for every message
            listen_host/listen_port: where we receive commands
            jetson_host: fixed Jetson IP, or None to reply to the last sender
            jetson_port: port on which the Jetson listens
        """
        self.on_message = on_message
        self.jetson_host = jetson_host
        self.jetson_port = jetson_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((listen_host, listen_port))
        self.sock.settimeout(0.5)
        self._running = True
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        print(f"📡 Waiting for Jetson commands on UDP {listen_host}:{listen_port}")

    def _recv_loop(self):
        while self._running:
            try:
                data, addr = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = json.loads(data.decode("utf-8"))
                if not isinstance(msg, dict) or "type" not in msg:
                    raise ValueError("message must be a JSON object with a 'type'")
            except ValueError as e:
                print(f"⚠️  Invalid message from {addr}: {e}")
                continue
            if self.jetson_host is None or self.jetson_host == addr[0]:
                self._last_sender = addr[0]
            try:
                self.on_message(msg)
            except Exception as e:
                print(f"⚠️  Error handling message {msg.get('type')}: {e}")
                self.send({"type": "error", "ref": msg.get("type"), "error": str(e)})

    @property
    def target(self):
        host = self.jetson_host or getattr(self, "_last_sender", None)
        return (host, self.jetson_port) if host else None

    def send(self, msg):
        """Send a dict to the Jetson. Silently dropped if no Jetson is known yet."""
        target = self.target
        if target is None:
            return False
        data = json.dumps(msg).encode("utf-8")
        with self._lock:
            try:
                self.sock.sendto(data, target)
                return True
            except OSError as e:
                print(f"⚠️  Could not send to Jetson {target}: {e}")
                return False

    def close(self):
        self._running = False
        self.sock.close()
