"""Conservative registration of a partly overlapping post-flip floor image."""

import math

import cv2
import numpy as np

from . import config as cfg


def _distinct_translation(reference, current):
    """Reject equally strong, separated phase peaks, such as repeated floor tiles.

    Do this without a window: its envelope can make a periodic image appear to
    have a unique match near the centre even when several shifts are identical.
    """
    first = np.fft.rfft2(reference - reference.mean())
    second = np.fft.rfft2(current - current.mean())
    cross = first * np.conj(second)
    magnitude = np.abs(cross)
    strongest = float(magnitude.max())
    if strongest == 0:
        return False, 1.0
    spectrum = np.divide(cross, magnitude, out=np.zeros_like(cross),
                         where=magnitude > strongest * 1e-8)
    peaks = np.fft.irfft2(spectrum, s=reference.shape).real
    y, x = np.unravel_index(np.argmax(peaks), peaks.shape)
    h, w = peaks.shape
    dy = np.abs(np.arange(h) - y)
    dx = np.abs(np.arange(w) - x)
    far = (np.minimum(dy, h - dy)[:, None] > 8) | (np.minimum(dx, w - dx)[None, :] > 8)
    ratio = float(peaks[far].max() / peaks[y, x]) if peaks[y, x] > 0 else 1.0
    return ratio < 0.8, ratio


def recover_translation(ref_img, g, expected_scale, yaw_delta):
    """Return ``(shift, phase_quality, diagnostics)`` for reliable real overlap.

    ``expected_scale`` is current height / reference height; ``yaw_delta`` uses
    the same positive rotation as Odometry's sensor normalization. The returned
    shift is in reference-image pixels, ready for ``Odometry._position``.
    A caller must still confirm independent frames before trusting a recovery.
    This helper never changes the images, tracking state, or ordinary odometry.
    """
    info = {"method": "matched_overlap_translation", "expected_scale": float(expected_scale)}
    if (ref_img.shape != g.shape or ref_img.ndim != 2 or min(g.shape) < 40
            or not math.isfinite(expected_scale) or not 0.45 < expected_scale < 2.2
            or not math.isfinite(yaw_delta) or abs(yaw_delta) > 35):
        return None, 0.0, dict(info, reason="invalid_registration_geometry")
    reference = np.asarray(ref_img, dtype=np.float32)
    current = np.asarray(g, dtype=np.float32)
    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        return None, 0.0, dict(info, reason="invalid_registration_image")
    if min(float(reference.std()), float(current.std())) < 3:
        return None, 0.0, dict(info, reason="not_enough_textured_overlap")

    h, w = reference.shape
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    scales = sorted(set([1.0] + [float(expected_scale * factor)
                                for factor in np.linspace(0.9, 1.1, 9)]))
    candidates = []
    best_rejected = dict(info, reason="no_reliable_overlap")
    min_response = max(0.2, cfg.VO_MIN_RESPONSE)
    min_correlation = max(0.6, cfg.VO_MIN_CORRELATION)
    support = np.ones((h, w), np.float32)
    for scale in scales:
        transform = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), yaw_delta, scale)
        normalized = cv2.warpAffine(current, transform, (w, h))
        shift, quality = cv2.phaseCorrelate(reference.copy(), normalized.copy(), window)
        sx, sy = shift
        diag = dict(info, registration_scale=scale, phase_quality=float(quality))
        if (not all(math.isfinite(v) for v in (sx, sy, quality))
                or abs(sx) >= w / 2 or abs(sy) >= h / 2 or quality < min_response):
            continue

        undo_shift = np.float32([[1, 0, -sx], [0, 1, -sy]])
        aligned = cv2.warpAffine(normalized, undo_shift, (w, h))
        # Preserve only pixels genuinely supplied by the current sensor image;
        # rotation, scale and translation can each introduce black padding.
        valid = cv2.warpAffine(support, transform, (w, h))
        valid = cv2.warpAffine(valid, undo_shift, (w, h)) >= 0.999
        valid[:8] = valid[-8:] = False
        valid[:, :8] = valid[:, -8:] = False
        overlap = float(valid.mean())
        diag["overlap_fraction"] = overlap
        if overlap < 0.35 or min(float(reference[valid].std()), float(aligned[valid].std())) < 3:
            best_rejected = dict(diag, reason="not_enough_textured_overlap")
            continue
        correlation = float(np.corrcoef(reference[valid], aligned[valid])[0, 1])
        diag["correlation"] = correlation
        if not math.isfinite(correlation) or correlation < min_correlation:
            best_rejected = dict(diag, reason="registered_images_disagree")
            continue
        distinct, ratio = _distinct_translation(reference, normalized)
        diag["secondary_peak_ratio"] = ratio
        if not distinct:
            best_rejected = dict(diag, reason="ambiguous_repeated_texture")
            continue
        candidates.append((correlation, quality, shift, diag))

    if not candidates:
        return None, 0.0, best_rejected
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    correlation, quality, shift, diag = candidates[0]
    # Similar image agreement must not support substantially different positions.
    if any(other[0] >= correlation - 0.03 and math.dist(shift, other[2]) > 10
           for other in candidates[1:]):
        return None, 0.0, dict(diag, reason="ambiguous_scale_or_translation")
    return tuple(float(v) for v in shift), float(quality), dict(diag, reason="matched_overlap_translation")
