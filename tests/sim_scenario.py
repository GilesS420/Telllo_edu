"""End-to-end simulated flight: python -X utf8 -m tests.sim_scenario."""

import math
import os
import tempfile
import threading
import time

from drone.app import DroneApp
from drone import config as cfg
from drone import sim as sim_world
from drone.sim import FakeTello


def main(local_reference=False):
    # Match the reported flight: take off from the H, rather than put a
    # clipped marker at the edge of the initial camera footprint.
    original_pad = sim_world.SIM_HELIPAD
    sim_world.SIM_HELIPAD = (0, 0, *original_pad[2:])
    tello = FakeTello(drift=True, wind_cms=1.5, seed=4,
                      camera_turn_deg=cfg.CAM_ROTATE_DEG)
    app = DroneApp(tello)
    complete = threading.Event()
    original = app.handle_command
    errors = []
    log_directory = tempfile.TemporaryDirectory()
    original_record_path = cfg.FLIP_RECORD_PATH
    cfg.FLIP_RECORD_PATH = os.path.join(log_directory.name, "last_flip.json")

    def tracked(cmd):
        try:
            original(cmd)
        except Exception as exc:
            errors.append(exc)
            raise
        finally:
            complete.set()

    app.handle_command = tracked

    def command(msg):
        complete.clear()
        app.submit(msg)
        if not complete.wait(30):
            raise AssertionError(f"command timed out: {msg}")
        if errors:
            raise AssertionError(errors)

    try:
        app.start()
        command({"type": "takeoff"})
        time.sleep(1)
        print("TAKEOFF", "truth", tello.truth(), "estimate", app.pose.get(),
              "valid", app.pose.position_valid, "tracking", app.pose.measured)
        assert app.pose.measured and app.pose.position_valid, "takeoff tracking lost"
        assert math.dist(tello.truth()[:2], app.pose.get()[:2]) < 15
        deadline = time.monotonic() + 3
        while app.seen_helipad() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert app.seen_helipad() is not None, "H was not detected"
        before = tello.truth()
        if local_reference:
            app.pose.position_valid = False  # reproduce a lost origin with fresh local tracking
        flip_started = time.monotonic()
        command({"type": "flip", "dir": "f", "pose": app.pose.get()[:3]})
        print("FLIP DURATION", time.monotonic() - flip_started,
              "RECOVERY", app.last_flip_record.get("recovery_s"))
        print("FLIP", "before", before, "truth", tello.truth(), "estimate", app.pose.get(),
              "valid", app.pose.position_valid, "tracking", app.pose.measured)
        assert app.pose.measured and app.pose.local_reference_valid, "pre-flip reference not recovered"
        assert app.pose.position_valid == (not local_reference), "global validity was invented"
        assert app.last_flip_record["status"] == "confirmed", "return was not measured as stable"
        assert math.dist(tello.truth()[:2], app.pose.get()[:2]) < 15
        assert math.dist(tello.truth()[:2], before[:2]) < 15, "physical return failed"
        command({"type": "helipad_land"})
        print("H LAND", "truth", tello.truth(), "H", sim_world.SIM_HELIPAD[:2])
        assert not app.airborne
        assert math.dist(tello.truth()[:2], sim_world.SIM_HELIPAD[:2]) < 15, "landed away from H"
        print("SIM SCENARIO PASS")
    finally:
        if hasattr(app, "mission_thread"):
            app.shutdown()
        if hasattr(app, "odometry"):
            app.odometry.stop()
        tello.end()
        cfg.FLIP_RECORD_PATH = original_record_path
        sim_world.SIM_HELIPAD = original_pad
        log_directory.cleanup()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-reference", action="store_true")
    main(local_reference=parser.parse_args().local_reference)
