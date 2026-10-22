"""Pure checks and axis inference for a manual, motors-stopped camera test.

The observations are image shifts of a fixed floor while someone moves the
drone forward, then left. No existing camera signs or rotation enter the fit.
"""

import math


class CalibrationError(ValueError):
    """The observations cannot support a camera-axis proposal."""


def validate_capture_pair(reference, current):
    """Reject unsuitable height/attitude samples before image registration.

    Both dictionaries contain height_cm, yaw_deg, pitch_deg and roll_deg.
    The narrow height range belongs to this manual procedure, not flight VO.
    """
    for sample in (reference, current):
        try:
            values = [float(sample[key]) for key in
                      ("height_cm", "yaw_deg", "pitch_deg", "roll_deg")]
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise CalibrationError("Télémétrie absente ou invalide.") from exc
        if not all(math.isfinite(value) for value in values):
            raise CalibrationError("Télémétrie non finie.")
        height, _, pitch, roll = values
        if not 60 <= height <= 120:
            raise CalibrationError("Hauteur ToF invalide : tenir le drone entre 60 et 120 cm.")
        if max(abs(pitch), abs(roll)) > 5:
            raise CalibrationError("Drone incliné : le tenir horizontal (±5 degrés).")
    yaw_change = (float(current["yaw_deg"]) - float(reference["yaw_deg"]) + 180) % 360 - 180
    ratio = float(current["height_cm"]) / float(reference["height_cm"])
    if abs(yaw_change) > 5:
        raise CalibrationError("Rotation supérieure à 5 degrés : refaire le déplacement sans tourner.")
    if abs(ratio - 1) > 0.15:
        raise CalibrationError("Variation de hauteur supérieure à 15 % : recommencer à hauteur constante.")


def infer_camera_transform(forward_shift, left_shift, *,
                           forward_cm_per_pixel, left_cm_per_pixel):
    """Return the unique signed quarter-turn mapping the shifts to body axes.

    Shifts are (image_x, image_y), with the same convention as
    Odometry._compare. Raw centimetre vectors are (shift_y, shift_x) * scale.
    Each candidate is R(rotation) @ diag(1, left_sign), representing all eight
    possible signed quarter-turns without equivalent duplicate settings.
    Raises CalibrationError instead of guessing when the movements disagree.
    """
    vectors = []
    for shift, scale in ((forward_shift, forward_cm_per_pixel),
                         (left_shift, left_cm_per_pixel)):
        try:
            sx, sy = (float(value) for value in shift)
            scale = float(scale)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CalibrationError("Déplacement d'image invalide.") from exc
        if not all(math.isfinite(value) for value in (sx, sy, scale)) or scale <= 0:
            raise CalibrationError("Déplacement ou échelle d'image invalide.")
        length_px = math.hypot(sx, sy)
        length_cm = length_px * scale
        if length_px < 4 or not 8 <= length_cm <= 40:
            raise CalibrationError("Déplacement insuffisant ou excessif : viser environ 20 cm.")
        vectors.append((sy / length_px, sx / length_px))

    forward, left = vectors
    dot = sum(a * b for a, b in zip(forward, left))
    if abs(dot) > 0.20:
        raise CalibrationError("Les deux déplacements ne sont pas assez perpendiculaires.")

    candidates = []
    for rotation, (cosine, sine) in zip((0, 90, 180, 270),
                                       ((1, 0), (0, 1), (-1, 0), (0, -1))):
        for left_sign in (1, -1):
            # Dot products with the intended forward=(1,0), left=(0,1).
            forward_cos = cosine * forward[0] - sine * left_sign * forward[1]
            left_cos = sine * left[0] + cosine * left_sign * left[1]
            if forward_cos > 0.92 and left_cos > 0.92:
                candidates.append({
                    "CAM_ROTATE_DEG": rotation,
                    "CAM_FORWARD_SIGN": 1,
                    "CAM_LEFT_SIGN": left_sign,
                    "forward_alignment_cos": forward_cos,
                    "left_alignment_cos": left_cos,
                    "movement_orthogonality_cos": dot,
                })
    if len(candidates) != 1:
        raise CalibrationError("Axes ambigus ou mouvements obliques : refaire les deux mesures.")
    return candidates[0]
