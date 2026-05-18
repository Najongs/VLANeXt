"""Vision-based trocar axis estimator (handoff v5 prototype).

Given a tool_camera frame + KP-predicted (u, v) center, fit an ellipse to the
trocar opening and recover the 3D axis (normal direction) in world frame.

Geometry:
  Trocar entry = circular ring of radius R (mm) lying in a plane with normal n.
  When projected through a pinhole camera, it appears as an ellipse.
    semi-major axis  a = R               (in image, along direction perpendicular
                                          to the tilt projection)
    semi-minor axis  b = R * cos(θ)      (along the tilt-projection direction)
    cos(θ) = b / a   where θ = angle between trocar normal and optical axis.

  In image: let φ = angle of the MINOR-axis direction (image x-right = 0, y-down = π/2).
    Tilt projection in image = (cos φ, sin φ).

  Map image → camera frame (MuJoCo convention, same as KeypointInferencer.predict_world_seed):
    image right  =>  cam -x       (project_to_2d flips u: u = -fx * px/pz)
    image down   =>  cam +y
    image deeper =>  cam +z       (cam looks along -Z, so +z = into scene)

  Trocar axis direction (entry→depth, AWAY from camera) in cam frame:
    axis_cam = ( sin θ · (-cos φ),   # image x-right component flipped → -cos φ
                 sin θ ·   sin φ,
                 cos θ )
  Then axis_world = cam_mat @ axis_cam.

This module is intentionally framework-light: no torch, no model loading.
"""
import numpy as np
import cv2


def fit_trocar_ellipse(img_rgb_uint8, kp_uv_norm, roi_px=64,
                        canny_lo=30, canny_hi=90, min_contour_pts=20,
                        expected_radius_px=None, intensity_max=120):
    """Fit ellipse to trocar opening in ROI around predicted KP center.

    Args:
        img_rgb_uint8: HxWx3 numpy (the tool_camera frame; will be resized to 256 if needed).
        kp_uv_norm: (u, v) in [0,1] normalized image coords (predicted).
        roi_px: ROI half-size in pixels at 256-resolution.

    Returns:
        dict with keys:
            center_px: (cx, cy) ellipse center in 256-image pixels (None on failure)
            axes_px: (minor, major) semi-axis lengths in pixels
            angle_deg: ellipse rotation (OpenCV convention: deg from horizontal of major axis)
            confidence: 0..1 heuristic (contour size / ROI area)
        Returns None on detection failure.
    """
    import PIL.Image as PImg
    pil = PImg.fromarray(img_rgb_uint8).convert("RGB")
    if pil.size != (256, 256):
        pil = pil.resize((256, 256), PImg.LANCZOS)
    img = np.array(pil)

    H, W = img.shape[:2]
    u_px = int(round(kp_uv_norm[0] * W))
    v_px = int(round(kp_uv_norm[1] * H))
    x0 = max(0, u_px - roi_px); x1 = min(W, u_px + roi_px)
    y0 = max(0, v_px - roi_px); y1 = min(H, v_px + roi_px)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None

    roi = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
    # Intensity mask: trocar interior is dark grey/black; exclude bright iris/sclera.
    dark_mask = (gray_blur <= intensity_max).astype(np.uint8) * 255
    # Edges only inside dark region.
    edges = cv2.Canny(gray_blur, canny_lo, canny_hi)
    edges = cv2.bitwise_and(edges, edges, mask=dark_mask)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    best = None
    best_score = -1.0
    roi_cx = (x1 - x0) / 2.0
    roi_cy = (y1 - y0) / 2.0
    for c in contours:
        if len(c) < min_contour_pts:
            continue
        try:
            ell = cv2.fitEllipse(c)
        except Exception:
            continue
        (cx, cy), (w, h), ang = ell
        if w <= 1 or h <= 1:
            continue
        # Prefer larger ellipse closer to ROI center.
        center_dist = np.hypot(cx - roi_cx, cy - roi_cy)
        size = max(w, h)
        # Reject ellipses that touch border (likely spurious).
        if cx - max(w, h) / 2 < 1 or cy - max(w, h) / 2 < 1: continue
        if cx + max(w, h) / 2 > (x1 - x0) - 1 or cy + max(w, h) / 2 > (y1 - y0) - 1: continue
        # Filter by expected size if provided: reject ellipses way larger/smaller than expected.
        if expected_radius_px is not None:
            er = float(expected_radius_px)
            r_eff = size / 2.0  # max semi-axis
            if r_eff < 0.4 * er or r_eff > 2.0 * er:
                continue
        # Score: bigger & more central is better.
        score = size - 0.5 * center_dist
        if score > best_score:
            best_score = score
            best = (cx, cy, w, h, ang, len(c))

    if best is None:
        return None
    cx_roi, cy_roi, w_axis, h_axis, ang_deg, n_pts = best
    # OpenCV fitEllipse returns FULL axes lengths (not semi); convert.
    semi_w = w_axis / 2.0
    semi_h = h_axis / 2.0
    if semi_w <= semi_h:
        semi_minor, semi_major = semi_w, semi_h
        # Angle returned is for the FIRST axis (width). Major axis perpendicular to it.
        major_angle_deg = ang_deg + 90.0
    else:
        semi_minor, semi_major = semi_h, semi_w
        major_angle_deg = ang_deg
    # Wrap to [0, 180)
    major_angle_deg = major_angle_deg % 180.0

    return {
        "center_px": (x0 + cx_roi, y0 + cy_roi),
        "axes_px": (float(semi_minor), float(semi_major)),
        "major_angle_deg": float(major_angle_deg),
        "n_contour_pts": int(n_pts),
        "confidence": min(1.0, n_pts / 200.0),
    }


def ellipse_to_axis_cam(semi_minor, semi_major, major_angle_deg):
    """Recover 3D trocar axis direction in CAMERA frame from ellipse params.

    Returns:
        axis_cam: unit 3-vec (cam frame), pointing entry→depth (AWAY from camera).
        tilt_deg: angle between trocar normal and optical axis.
    """
    ratio = float(np.clip(semi_minor / max(semi_major, 1e-6), 0.0, 1.0))
    tilt = float(np.arccos(ratio))
    # Minor-axis direction in image = major + 90°, in degrees from image x-right.
    minor_angle_deg = (major_angle_deg + 90.0) % 180.0
    phi = np.deg2rad(minor_angle_deg)
    # image (cos φ, sin φ) → cam (-cos φ, sin φ, 0); add depth component +cos θ.
    axis_cam = np.array([
        np.sin(tilt) * (-np.cos(phi)),
        np.sin(tilt) *   np.sin(phi),
        np.cos(tilt),
    ], dtype=np.float64)
    axis_cam = axis_cam / (np.linalg.norm(axis_cam) + 1e-10)
    return axis_cam, float(np.rad2deg(tilt))


def estimate_trocar_axis_world(img_rgb_uint8, kp_uv_norm, cam_mat,
                                roi_px=64, prior_axis_world=None,
                                expected_radius_px=None, intensity_max=120):
    """End-to-end: image + KP center + camera extrinsics → trocar axis in WORLD frame.

    Args:
        img_rgb_uint8: tool_camera frame (uint8 HxWx3).
        kp_uv_norm: predicted (u, v) in [0,1].
        cam_mat: 3x3 cam-to-world rotation (env.data.cam_xmat reshape).
        prior_axis_world: optional 3-vec. If provided and ellipse-derived axis has
            negative dot with it, flip sign (handles 2-fold ambiguity of normal).

    Returns dict:
        {"axis_world": 3-vec or None, "tilt_deg": float, "ellipse": {...} | None,
         "ok": bool}
    """
    ell = fit_trocar_ellipse(img_rgb_uint8, kp_uv_norm, roi_px=roi_px,
                              expected_radius_px=expected_radius_px,
                              intensity_max=intensity_max)
    if ell is None:
        return {"axis_world": None, "tilt_deg": None, "ellipse": None, "ok": False}
    smin, smaj = ell["axes_px"]
    axis_cam, tilt_deg = ellipse_to_axis_cam(smin, smaj, ell["major_angle_deg"])
    axis_world = (cam_mat @ axis_cam).astype(np.float64)
    axis_world = axis_world / (np.linalg.norm(axis_world) + 1e-10)
    if prior_axis_world is not None:
        prior = np.asarray(prior_axis_world, dtype=np.float64)
        if np.dot(axis_world, prior) < 0:
            axis_world = -axis_world
    return {
        "axis_world": axis_world,
        "tilt_deg": tilt_deg,
        "ellipse": ell,
        "ok": True,
    }
