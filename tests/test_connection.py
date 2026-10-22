import unittest
from unittest.mock import Mock, patch

from djitellopy import Tello

from drone import config as cfg
from drone.connection import ConfiguredTello, network_hint


class ConnectionTests(unittest.TestCase):
    def client(self, responses, state=None):
        client = Mock()
        client.address = ("192.168.10.1", 8889)
        client.send_command_with_return.side_effect = responses
        client.get_current_state.return_value = state or {"bat": 80}
        return client

    @patch("drone.connection.network_hint", return_value="local IP: 10.2.172.127")
    def test_no_response_is_bounded_and_explains_network(self, _hint):
        client = self.client(["timeout"] * cfg.CONNECT_ATTEMPTS)
        with self.assertRaisesRegex(ConnectionError, "10.2.172.127"):
            ConfiguredTello.connect(client)
        self.assertEqual(client.send_command_with_return.call_count, cfg.CONNECT_ATTEMPTS)
        for call in client.send_command_with_return.call_args_list:
            self.assertEqual(call.args, ("command",))
            self.assertEqual(call.kwargs, {"timeout": cfg.CONNECT_TIMEOUT})
        client.get_current_state.assert_not_called()

    @patch("drone.connection.network_hint", return_value="network")
    def test_retry_then_success(self, _hint):
        client = self.client(["timeout", "ok"])
        ConfiguredTello.connect(client)
        self.assertEqual(client.send_command_with_return.call_count, 2)

    @patch("drone.connection.network_hint", return_value="network")
    def test_acknowledgement_does_not_hide_missing_telemetry(self, _hint):
        client = self.client(["ok"])
        client.get_current_state.return_value = {}
        with patch("drone.connection.time.monotonic", side_effect=[0, 10]):
            with self.assertRaisesRegex(ConnectionError, "8890"):
                ConfiguredTello.connect(client)

    @patch("drone.connection.network_hint", return_value="network")
    def test_keyboard_interrupt_is_not_retried(self, _hint):
        client = self.client([KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            ConfiguredTello.connect(client)
        self.assertEqual(client.send_command_with_return.call_count, 1)

    def test_timeout_forwarding_through_library_decorators(self):
        client = ConfiguredTello.__new__(ConfiguredTello)
        # No sockets are opened by __new__; prevent library destructor cleanup.
        with patch.object(ConfiguredTello, "end"), \
             patch.object(Tello, "send_command_with_return", return_value="ok") as send, \
             patch.object(Tello, "send_control_command", return_value=True) as control:
            client.send_command_with_return("battery?")
            send.assert_called_once_with("battery?", timeout=cfg.RESPONSE_TIMEOUT)
            client.send_command_with_return("flip f", timeout=5)
            send.assert_called_with("flip f", timeout=5)
            client.send_control_command("go 100 0 0 30")
            control.assert_called_once_with("go 100 0 0 30", timeout=cfg.RESPONSE_TIMEOUT)
            del client

    def test_wrong_subnet_hint_uses_os_route_without_sending_packet(self):
        with patch("drone.connection.socket.socket") as factory:
            sock = factory.return_value.__enter__.return_value
            sock.getsockname.return_value = ("10.2.172.127", 12345)
            hint = network_hint("192.168.10.1")
            self.assertIn("TELLO-", hint)
            self.assertIn("10.2.172.127", hint)
            sock.sendto.assert_not_called()


if __name__ == "__main__":
    unittest.main()
