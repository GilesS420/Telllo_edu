"""Bounded SDK connection with diagnostics, without sending flight commands."""

import socket
import time

from djitellopy import Tello

from . import config as cfg


def network_hint(host):
    """Ask the OS which local address it would use; no packet is transmitted."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect((host, 8889))
            local_ip = sock.getsockname()[0]
    except OSError as exc:
        return f"Geen netwerkroute naar {host}: {exc}"
    hint = f"Drone: {host}:8889 (UDP). Lokaal IP: {local_ip}."
    if host == "192.168.10.1" and not local_ip.startswith("192.168.10."):
        hint += ("\nHet gebruikte netwerk is niet het directe Tello-netwerk. "
                 "Verbind met de Wi-Fi van de drone (TELLO-...), ook zonder internet.")
    return hint


class ConfiguredTello(Tello):
    # Explicit forwarding works with djitellopy's decorated methods. Changing
    # Tello.RESPONSE_TIMEOUT does not change their bound default arguments.
    def send_command_with_return(self, command, timeout=None):
        return super().send_command_with_return(
            command, timeout=cfg.RESPONSE_TIMEOUT if timeout is None else timeout)

    def send_control_command(self, command, timeout=None):
        return super().send_control_command(
            command, timeout=cfg.RESPONSE_TIMEOUT if timeout is None else timeout)

    def connect(self, wait_for_state=True):
        hint = network_hint(self.address[0])
        print(hint)
        response = "geen antwoord"
        for attempt in range(cfg.CONNECT_ATTEMPTS):
            print(f"Tello verbinden ({attempt + 1}/{cfg.CONNECT_ATTEMPTS})...")
            response = self.send_command_with_return("command", timeout=cfg.CONNECT_TIMEOUT)
            if str(response).strip().lower() == "ok":
                break
        else:
            raise ConnectionError(
                f"Geen SDK-antwoord van de drone na {cfg.CONNECT_ATTEMPTS} pogingen.\n"
                f"{hint}\nAntwoord: {response}\n"
                "Controleer of de drone aanstaat en of een andere Tello-app gesloten is.")

        if wait_for_state:
            deadline = time.monotonic() + cfg.CONNECT_STATE_TIMEOUT
            while not self.get_current_state():
                if time.monotonic() >= deadline:
                    raise ConnectionError(
                        f"Drone antwoordt op UDP 8889, maar telemetrie op UDP 8890 ontbreekt.\n"
                        f"{hint}\nControleer de Windows-firewalltoegang voor deze Python-installatie.")
                time.sleep(0.05)
