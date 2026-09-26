"""Check the reference -> video registration with evidence that does not come from SIFT.

For each video median background (made with register.median_frame from ~60 frames):
  1. SIFT + RANSAC homography and similarity (register.estimate), with and without
     the static-pixel mask.
  2. Named fixed landmarks (signs, kerb corners, a manhole cover, ...) picked once on
     the reference and found in each video by normalized cross-correlation of
     gradient images. The NCC position is compared with the position predicted by
     the homography and by the similarity. Weak NCC matches are reported but not
     used for the verdict.
  3. Signal heads: reference head centres mapped with the homography vs the traffic
     light boxes YOLO found independently in each video (eda.json).
  4. Displacement field |H(p) - p| over a grid, so a "shift" is never quoted as one
     number when the transform includes rotation and scale.
Writes registration.json and before/after edge overlays.

  python tools/scene/verify_registration.py --bg-dir work/bg --eda-dir reports/eda \
      --out reports/eda --stability work/stability.json
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.scene.register import REF_PATH, estimate, load_ref, to_video  # noqa: E402

REF_STEM = "C3897"
# reference landmarks, 1080p pixel of the reference median (x2 for 4K)
LANDMARKS = {
    "ped_crossing_sign": (1228, 370), "arrow_sign": (1280, 512), "gantry_diamond_sign": (585, 312),
    "gantry_post_foot": (240, 684), "middle_island_apex": (633, 670),
    "lower_island_right_vertex": (950, 877), "left_island_left_vertex": (118, 932),
    "median_nose_kerb": (1178, 512), "round_island_kerb": (1240, 562), "left_concrete_bin": (150, 722),
    "manhole_cover": (1740, 668), "building_OO_sign": (955, 130), "street_lamp_head": (298, 46),
    "white_tree_base": (1355, 300)}
# signal heads in the reference, 4K px (YOLO traffic-light boxes on a 4K C3897 frame, tiled)
REF_HEADS = {"A": [517, 964, 554, 1038], "B": [549, 957, 594, 1034], "C": [2244, 719, 2281, 779],
             "D": [2291, 719, 2340, 827], "E": [1442, 493, 1502, 615], "F": [799, 530, 835, 640],
             "G": [3782, 708, 3816, 753]}
NCC_MIN, NCC_MARGIN = 0.70, 0.20


def _grad(img):
    g = cv2.createCLAHE(3.0, (8, 8)).apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(np.float32)
    return cv2.GaussianBlur(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), 1.0)


def ncc_find(R, I, p, half=24, search=110):
    """Position of reference patch around p (1080p px) in image I; subpixel."""
    x, y = p
    t = R[y - half:y + half, x - half:x + half]
    x0, y0 = max(0, x - half - search), max(0, y - half - search)
    res = cv2.matchTemplate(I[y0:y + half + search, x0:x + half + search], t, cv2.TM_CCOEFF_NORMED)
    _, mx, _, (j, i) = cv2.minMaxLoc(res)
    r2 = res.copy()
    cv2.circle(r2, (j, i), 8, -1, -1)
    dx = dy = 0.0
    if 0 < j < res.shape[1] - 1:
        a, b, c = res[i, j - 1], res[i, j], res[i, j + 1]
        dx = 0.5 * (a - c) / (a - 2 * b + c + 1e-9)
    if 0 < i < res.shape[0] - 1:
        a, b, c = res[i - 1, j], res[i, j], res[i + 1, j]
        dy = 0.5 * (a - c) / (a - 2 * b + c + 1e-9)
    return (x0 + j + dx + half, y0 + i + dy + half), float(mx), float(r2.max())


def overlay(ref, img, r, path):
    """Top: reference edges (magenta) on the raw video median. Bottom: on the video
    median warped into the reference frame with H_video_to_ref."""
    s = 1920 / 3840
    S = np.diag([s, s, 1.0])
    Hw = S @ np.array(r["H_video_to_ref"]) @ np.linalg.inv(S)
    warped = cv2.warpPerspective(img, Hw, (ref.shape[1], ref.shape[0]))
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), (3, 3), 0), 60, 150) > 0
    panels = []
    for base, label in ((img, "before: raw video median"), (warped, "after: warped with H_video_to_ref")):
        v = (base * 0.75).astype(np.uint8)
        v[edges] = (255, 0, 255)
        v = cv2.resize(v, (1280, 720), interpolation=cv2.INTER_AREA)
        cv2.putText(v, label + " | magenta = reference (C3897) edges", (12, 30), 0, 0.8, (255, 255, 255), 2)
        panels.append(v)
    cv2.imwrite(path, np.vstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 72])


def disp_stats(r):
    xs, ys = np.meshgrid(np.linspace(100, 3740, 20), np.linspace(100, 2060, 12))
    P = np.column_stack([xs.ravel(), ys.ravel()])
    d = to_video(P, r) - P
    m = np.linalg.norm(d, axis=1)
    junction = np.array([[2000, 1100]])
    dj = (to_video(junction, r) - junction)[0]
    return {"grid_px": "20 x 12 points over the 4K frame",
            "min": round(float(m.min()), 1), "median": round(float(np.median(m)), 1), "max": round(float(m.max()), 1),
            "at_junction_2000_1100": [round(float(dj[0]), 1), round(float(dj[1]), 1)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bg-dir", required=True, help="folder with <STEM>_median.png and <STEM>_static_mask.png")
    ap.add_argument("--eda-dir", required=True, help="reports/eda (for YOLO traffic light boxes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stems", default="C3896,C3902,C3905")
    ap.add_argument("--stability", help="stability.json from the server run (optional)")
    ap.add_argument("--drift", help="JSON {stem: [[frame, max_disp_px_vs_own_median], ...]} (optional)")
    a = ap.parse_args()
    ref = load_ref()
    Rg = _grad(ref)
    ref_mask = cv2.imread(os.path.join(a.bg_dir, f"{REF_STEM}_static_mask.png"), 0)
    os.makedirs(os.path.join(a.out, "registration"), exist_ok=True)
    result = {"reference": {"stem": REF_STEM, "image": os.path.relpath(REF_PATH),
                            "made_from": "median of 60 frames spread over the 1080p proxy",
                            "coords": "all points and matrices in 4K px (3840x2160)"},
              "landmarks_ref_4k": {k: [2 * v[0], 2 * v[1]] for k, v in LANDMARKS.items()},
              "heads_ref_4k": REF_HEADS,
              "gates": {"landmark_ncc_min": NCC_MIN, "landmark_ncc_margin_over_2nd_peak": NCC_MARGIN},
              "videos": {}}
    regs = {}
    for stem in a.stems.split(","):
        img = cv2.imread(os.path.join(a.bg_dir, f"{stem}_median.png"))
        msk = cv2.imread(os.path.join(a.bg_dir, f"{stem}_static_mask.png"), 0)
        r = estimate(ref, img)
        rm = estimate(ref, img, ref_mask, msk)
        regs[stem] = r
        A = np.array(r["S_ref_to_video"])
        Ig = _grad(img)
        lms = []
        for name, p in LANDMARKS.items():
            q, ncc, second = ncc_find(Rg, Ig, p)
            q4 = np.array(q) * 2
            p4 = np.array(p, float) * 2
            qh = to_video([p4], r)[0]
            qs = A[:, :2] @ p4 + A[:, 2]
            qm = to_video([p4], rm)[0]
            lms.append({"name": name, "ref": p4.tolist(), "ncc_found": np.round(q4, 1).tolist(),
                        "ncc": round(ncc, 3), "ncc_second_peak": round(second, 3),
                        "used": bool(ncc >= NCC_MIN and ncc - second >= NCC_MARGIN),
                        "err_homography_px": round(float(np.linalg.norm(q4 - qh)), 2),
                        "err_similarity_px": round(float(np.linalg.norm(q4 - qs)), 2),
                        "err_homography_masked_px": round(float(np.linalg.norm(q4 - qm)), 2)})
        used = [x for x in lms if x["used"]]

        def agg(key):
            v = np.array([x[key] for x in used])
            return {"n": int(len(v)), "median": round(float(np.median(v)), 2), "max": round(float(v.max()), 2)} if len(v) else None

        # heads: mapped reference centre vs nearest YOLO box centre found in this video
        det = json.load(open(os.path.join(a.eda_dir, stem, "eda.json")))["extra"]["traffic_lights_detected"]
        dc = np.array([[(d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2] for d in det])
        heads = {}
        for h, b in REF_HEADS.items():
            c = to_video([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)], r)[0]
            dd = np.linalg.norm(dc - c, axis=1)
            j = int(dd.argmin())
            heads[h] = {"mapped_centre": np.round(c, 1).tolist(),
                        "nearest_yolo_box": det[j]["box"] if dd[j] < 60 else None,
                        "err_px": round(float(dd[j]), 1) if dd[j] < 60 else None}
        overlay(ref, img, r, os.path.join(a.out, "registration", f"{stem}_overlay.jpg"))
        result["videos"][stem] = {
            "homography": {k: r[k] for k in ("H_ref_to_video", "H_video_to_ref")},
            "similarity_fit": {k: r[k] for k in ("S_ref_to_video", "S_video_to_ref", "similarity")},
            "sift": {k: r[k] for k in ("matches", "inliers_h", "inliers_s", "resid_h_px", "resid_s_px", "inlier_spread")},
            "sift_masked": {"similarity": rm["similarity"], "inliers_h": rm["inliers_h"],
                            "resid_h_px": rm["resid_h_px"]},
            "displacement_ref_to_video_px": disp_stats(r),
            "landmarks": lms,
            "landmark_error_summary": {"homography": agg("err_homography_px"),
                                       "similarity": agg("err_similarity_px"),
                                       "homography_masked": agg("err_homography_masked_px")},
            "heads_vs_yolo": heads}
        s = result["videos"][stem]
        print(stem, r["similarity"], s["displacement_ref_to_video_px"], s["landmark_error_summary"], flush=True)
    # C3905 relative to C3902 (both evening): are they the same set-up?
    if "C3905" in regs and "C3902" in regs:
        H = np.array(regs["C3905"]["H_ref_to_video"]) @ np.array(regs["C3902"]["H_video_to_ref"])
        xs, ys = np.meshgrid(np.linspace(100, 3740, 20), np.linspace(100, 2060, 12))
        P = np.column_stack([xs.ravel(), ys.ravel()])
        Q = cv2.perspectiveTransform(P.reshape(-1, 1, 2), H).reshape(-1, 2)
        m = np.linalg.norm(Q - P, axis=1)
        result["C3902_to_C3905_displacement_px"] = {"min": round(float(m.min()), 1), "median": round(float(np.median(m)), 1),
                                                   "max": round(float(m.max()), 1)}
    if a.stability and os.path.exists(a.stability):
        st = json.load(open(a.stability))
        result["within_video_stability"] = {
            k: {"max_disp_single_frame_vs_own_median_px": max(p["max_disp_px"] for p in v["frames_vs_own_median"]),
                "frames_checked": len(v["frames_vs_own_median"])} for k, v in st.items()}
        result["runtime"] = {k: v["runtime_s"] for k, v in st.items()}
        result["runtime_note"] = ("seconds on the 2-core server (decode = reading frames, match = median + mask + "
                                  "SIFT + RANSAC). proxy_n15 = 15 frames spread over the 1080p proxy; "
                                  "original_4k_n10_span10 = 10 frames from the first 10 s of the 4K original.")
    if a.drift and os.path.exists(a.drift):
        dr = json.load(open(a.drift))
        fps = 30000 / 1001
        res = {}
        for k, pts in dr.items():
            pts = sorted(pts)
            n_early = [p for p in pts if p[0] < 3000]
            settle = next((p[0] for i, p in enumerate(n_early) if all(q[1] < 2.0 for q in n_early[i:])), None)
            res[k] = {"samples_frame_px": pts,
                      "first_frame_px": pts[0][1] if pts else None,
                      "settled_below_2px_from_s": None if settle is None else round(settle / fps, 1)}
        result["early_clip_drift"] = res
        result["early_clip_drift_note"] = ("max displacement (4K px, 5 test points) of single frames vs the "
                                           "video's own median. The camera settles after recording starts; "
                                           "gyro spikes in the rtmd track at 0 to 2 s and in the last seconds "
                                           "match the button presses.")
    with open(os.path.join(a.out, "registration.json"), "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
