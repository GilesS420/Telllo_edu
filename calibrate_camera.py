"""Manual, non-flying camera-axis proposal. Run: python calibrate_camera.py.

Importing this module never connects to hardware. Only video/telemetry and SDK
mode are used; do not reuse DroneApp or Tello.end(), which can command flight.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from drone import config as cfg
from drone.camera_calibration import (CalibrationError, infer_camera_transform,
                                      validate_capture_pair)
from drone.downcam import DownCam, cm_per_pixel
from drone.odometry import Odometry


WORKSPACE = Path(__file__).resolve().parent


def _new_frame(reader, previous=None, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = reader.frame
        if frame is not None and frame is not previous and frame.size:
            return frame
        time.sleep(0.02)
    raise CalibrationError("Aucune nouvelle image : vérifier le flux de la caméra inférieure.")


def _warmup_camera(reader, camera):
    # prepare(), rather than Odometry._gray(), initializes the automatic crop.
    previous = None
    for _ in range(10):
        previous = _new_frame(reader, previous)
        camera.prepare(previous.copy())
    if cfg.AUTO_CROP and cfg.DOWNCAM_CROP is None and camera.crop is None:
        raise CalibrationError("Impossible de fixer le recadrage : éclairer un sol texturé.")
    return previous


def _capture(tello, reader, camera):
    # Let video catch up after the person has stopped moving, then use a new
    # frame and the current cached sensor packet. No flight command is sent.
    time.sleep(max(0.35, cfg.FRAME_LATENCY_S + 0.15))
    frame = _new_frame(reader, reader.frame)
    state = dict(tello.get_current_state())
    try:
        sample = {"height_cm": float(state["tof"]), "yaw_deg": float(state["yaw"]),
                  "pitch_deg": float(state["pitch"]), "roll_deg": float(state["roll"])}
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise CalibrationError("Télémétrie ToF/yaw/pitch/roll indisponible.") from exc
    validate_capture_pair(sample, sample)
    gray = camera.cropped(camera.to_gray(frame.copy()))
    if gray.shape[1] > 320:
        gray = cv2.resize(gray, (320, int(round(gray.shape[0] * 320 / gray.shape[1]))),
                          interpolation=cv2.INTER_AREA)
    sample["img"] = gray.astype(np.float32)
    return sample


def _measure_shift(reference, current):
    validate_capture_pair(reference, current)
    if reference["img"].shape != current["img"].shape:
        raise CalibrationError("Le format ou recadrage de l'image a changé.")
    # Reuse exactly the existing registration without starting odometry or any
    # worker thread, and without applying the current camera transform.
    registration = Odometry.__new__(Odometry)
    registration._window = None
    ref = {"img": reference["img"], "height": reference["height_cm"],
           "yaw": reference["yaw_deg"]}
    shift, quality = registration._compare(ref, current["img"],
                                            current["height_cm"], current["yaw_deg"])
    method = "phase"
    diagnostics = {"phase_quality": float(quality) if math.isfinite(quality) else None}
    if not math.isfinite(quality) or quality < max(0.2, cfg.VO_MIN_RESPONSE):
        shift, features = registration._feature_shift(
            ref, current["img"], current["height_cm"], current["yaw_deg"])
        diagnostics.update(features)
        method = "features"
        if shift is None:
            raise CalibrationError("Images non concordantes ou sol trop peu texturé : "
                                   + str(features.get("reason", "échec de recalage")))
        rotation = float(features.get("residual_rotation_deg", 0))
        scale = float(features.get("residual_scale", 1))
        if (not all(math.isfinite(value) for value in (rotation, scale)) or scale <= 0
                or abs(rotation) > 5 or max(scale, 1 / scale) > 1.15):
            raise CalibrationError("Le recalage révèle une rotation ou variation d'échelle excessive.")
    scale = cm_per_pixel(reference["img"].shape[1],
                         (reference["height_cm"] + current["height_cm"]) / 2)
    return {"shift_px": [float(value) for value in shift], "cm_per_pixel": scale,
            "reference": {k: v for k, v in reference.items() if k != "img"},
            "current": {k: v for k, v in current.items() if k != "img"},
            "registration": method, "diagnostics": diagnostics}


def _capture_movement(tello, reader, camera, direction):
    input(f"\nNouvelle référence pour le mouvement {direction}. "
          "Tenir le drone immobile et horizontal, puis appuyer sur Entrée. ")
    reference = _capture(tello, reader, camera)
    input(f"Déplacer À LA MAIN d'environ 20 cm {direction}, sans tourner ni changer "
          "de hauteur. S'immobiliser, puis appuyer sur Entrée. ")
    current = _capture(tello, reader, camera)
    measured = _measure_shift(reference, current)
    print(f"Mesure acceptée ({measured['registration']}), "
          f"décalage image : {measured['shift_px']} px.")
    return measured


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Proposer les axes de la caméra inférieure par deux déplacements à la main, moteurs arrêtés.",
        epilog="Ne modifie pas la configuration. Proposition : flight_logs/camera_calibration.json.")
    parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    from drone.connection import ConfiguredTello

    print("Proposition de calibration MANUELLE de l'orientation de la caméra, SANS VOL.")
    print("Fermer la GUI et les autres applications Tello. Moteurs arrêtés pendant tout le test.")
    print("Tenir le drone horizontal à 60–120 cm au-dessus d'un sol fixe, bien éclairé et texturé.")
    print("Les deux mouvements se font à la main. Aucun réglage ne sera appliqué automatiquement.")
    input("Drone prêt, moteurs arrêtés et Wi-Fi Tello connecté : appuyer sur Entrée. ")
    tello = ConfiguredTello()
    reader = None
    streaming = False
    try:
        tello.connect()
        # Mark the attempt before sending: cleanup also covers a lost streamon
        # acknowledgement when streaming actually started on the drone.
        streaming = True
        tello.streamon()
        response = tello.send_command_with_return("downvision 1")
        if str(response).strip().lower() != "ok":
            raise CalibrationError(f"Caméra inférieure indisponible : {response}")
        reader = tello.get_frame_read()
        time.sleep(0.5)
        camera = DownCam()
        _warmup_camera(reader, camera)
        forward = _capture_movement(tello, reader, camera, "vers le nez du drone")
        print("Repositionner librement le drone : la mesure gauche utilisera une référence indépendante.")
        left = _capture_movement(tello, reader, camera, "vers la gauche du drone")
        fit = infer_camera_transform(
            forward["shift_px"], left["shift_px"],
            forward_cm_per_pixel=forward["cm_per_pixel"],
            left_cm_per_pixel=left["cm_per_pixel"])
        settings = {key: fit[key] for key in ("CAM_ROTATE_DEG", "CAM_FORWARD_SIGN", "CAM_LEFT_SIGN")}
        proposal = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "procedure": "manual_motors_stopped_forward_then_left",
            "applied": False,
            "note": "Proposition d'axes/signes uniquement ; échelle optique et comportement en vol non validés.",
            "proposed_config": settings,
            "current_config": {key: getattr(cfg, key) for key in settings},
            "fit": {key: value for key, value in fit.items() if key not in settings},
            "measurements": {"forward": forward, "left": left},
        }
        output = WORKSPACE / "flight_logs" / "camera_calibration.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(proposal, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")
        print(f"\nProposition à examiner : {output}")
        print("Après vérification, reporter manuellement ces valeurs dans drone/config.py :")
        for key, value in settings.items():
            print(f"{key} = {value}")
        print("Aucun réglage appliqué. Le sens des axes est proposé par ce test manuel, sans validation en vol.")
        return 0
    except Exception as exc:
        print(f"Calibration refusée : {exc}\nAucune nouvelle proposition enregistrée.")
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nTest interrompu. Aucune nouvelle proposition enregistrée.")
        return 1
    finally:
        if streaming:
            try:
                tello.streamoff()
            except Exception as exc:
                print(f"Arrêt du flux vidéo impossible : {exc}")
        if reader is not None:
            try:
                reader.stop()
            except Exception as exc:
                print(f"Arrêt du décodeur vidéo impossible : {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
