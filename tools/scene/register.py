"""Align any recording of the (not quite) fixed camera to the reference frame.

The reference frame is the median background of C3897 (tools/scene/reference_C3897.jpg,
1920x1080, a clean frame with moving traffic removed). All transforms returned here
are in ORIGINAL 4K pixel coordinates (3840x2160) on both sides, so hand-drawn zones
in reference 4K pixels can be moved into a video with H_ref_to_video, and detections
in a video can be moved into the reference with H_video_to_ref.

Method: median of N frames of the video (removes moving objects) -> SIFT on the
1080p image -> Lowe ratio test (0.75) -> RANSAC homography and RANSAC similarity.
Optionally (--mask) pixels that change a lot across the N frames are masked out
before SIFT. On the four sample videos the mask did not improve the independent
landmark check (reports/eda/registration.json) and cost inliers under dusk light,
so it is off by default. Use the homography: the similarity fit is off by up to
~11 px at the frame edges because the view angle changed between recordings.

The camera settles for up to ~30 s after recording starts (C3896: 30 px at frame 0,
below 2 px after 27 s). Register from frames after that, or re-register in a
sliding window if early frames matter.

Python use
    from tools.scene.register import register_video, to_video, to_ref
    r = register_video("C3902.MP4")              # dict, see estimate()
    zone_in_video = to_video(zone_ref_pts, r)    # (N, 2) 4K px
    det_in_ref = to_ref(det_pts, r)

CLI
    python tools/scene/register.py --video proxies/C3902_1080p.mp4 [--frames 15] [--out r.json]

Runtime (measured on the 2-core server, reports/eda/registration.json "runtime"):
15 frames spread over a 1080p proxy: 10 to 15 s decode + 3 to 4.5 s matching.
10 frames from the first 10 s of a 4K original (--span 10): 20 to 24 s decode +
3 to 4 s matching; that result differed from the whole-clip proxy result by 3.8 and
5.3 px (C3905, C3902), consistent with the camera still settling early in the clip.
Decoding dominates, so pass frames the model already decoded to register_frames().
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF_PATH = os.path.join(HERE, "reference_C3897.jpg")
FULL = (3840, 2160)   # coordinate system of all returned transforms
WORK_W = 1920         # SIFT runs at this width


# ------------------------------------------------------------------ frames
def sample_frames(video, n=15, span=None, work_w=WORK_W):
    """n frames spread evenly over the clip (or over its first `span` seconds,
    decoded sequentially, which is much faster on long-GOP 4K files).
    Returns (frames as uint8 BGR at work_w wide, list of frame indices)."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise IOError(f"cannot open {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    last = total - 2 if span is None else min(total - 2, int(span * fps))
    want = np.unique(np.linspace(0, max(last, 0), n).astype(int)).tolist()
    out, got = [], []
    if span is None:
        for fi in want:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if ok:
                out.append(_resize(fr, work_w))
                got.append(fi)
    else:
        wanted, idx = set(want), -1
        while idx < last:
            if not cap.grab():
                break
            idx += 1
            if idx in wanted:
                ok, fr = cap.retrieve()
                if ok:
                    out.append(_resize(fr, work_w))
                    got.append(idx)
    cap.release()
    if len(out) < 3:
        raise IOError(f"only {len(out)} frames decoded from {video}")
    return out, got


def _resize(img, w):
    if img.shape[1] == w:
        return img
    h = int(round(img.shape[0] * w / img.shape[1]))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def median_frame(frames):
    """Per-pixel median; chunked by rows so 60 x 1080p frames fit in little memory."""
    stack_h = frames[0].shape[0]
    bg = np.empty_like(frames[0])
    for r0 in range(0, stack_h, 90):
        bg[r0:r0 + 90] = np.median(np.stack([f[r0:r0 + 90] for f in frames]), axis=0).astype(np.uint8)
    return bg


def static_mask(frames, bg=None, thr=12.0, frac=0.25):
    """255 where the pixel is static: its gray value differs from the median by more
    than `thr` in fewer than `frac` of the frames. Moving traffic, pedestrians and
    waving leaves become 0. Dilated a little so SIFT does not sit on their edges."""
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.int16) for f in frames]
    m = cv2.cvtColor(bg if bg is not None else median_frame(frames), cv2.COLOR_BGR2GRAY).astype(np.int16)
    busy = np.zeros(m.shape, np.float32)
    for x in g:
        busy += (np.abs(x - m) > thr)
    busy /= len(g)
    mask = (busy < frac).astype(np.uint8) * 255
    return cv2.erode(mask, np.ones((9, 9), np.uint8))


# ------------------------------------------------------------------ matching
def load_ref(path=REF_PATH):
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"reference image not found: {path}")
    return _resize(img, WORK_W)


def _sift(img, mask=None, nfeat=8000):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)  # evens out day / dusk contrast
    return cv2.SIFT_create(nfeatures=nfeat).detectAndCompute(gray, mask)


def estimate(ref, img, ref_mask=None, img_mask=None, ratio=0.75, ransac_px=2.0):
    """Transforms between reference and `img` (both BGR, same work width).

    Returns a dict (all matrices in FULL 4K pixel coordinates):
      H_ref_to_video, H_video_to_ref      3x3 homographies
      S_ref_to_video, S_video_to_ref      2x3 similarity (rotation, uniform scale, shift)
      similarity: {tx, ty, rot_deg, scale} of S_ref_to_video
      matches, inliers_h, inliers_s       counts after ratio test / RANSAC
      resid_h_px, resid_s_px              median / p90 inlier reprojection error (4K px)
      inlier_spread                       fraction of a 4x4 image grid with >= 3 inliers
    """
    s = FULL[0] / ref.shape[1]
    k1, d1 = _sift(ref, ref_mask)
    k2, d2 = _sift(img, img_mask)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        raise RuntimeError("too few SIFT features")
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m for m, n2 in (p for p in knn if len(p) == 2) if m.distance < ratio * n2.distance]
    if len(good) < 12:
        raise RuntimeError(f"only {len(good)} ratio-test matches")
    p1 = np.float32([k1[m.queryIdx].pt for m in good]) * s
    p2 = np.float32([k2[m.trainIdx].pt for m in good]) * s
    H, mh = cv2.findHomography(p1, p2, cv2.RANSAC, ransac_px * s, maxIters=5000, confidence=0.999)
    A, ma = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=ransac_px * s,
                                        maxIters=5000, confidence=0.999)
    if H is None or A is None:
        raise RuntimeError("RANSAC failed")
    mh, ma = mh.ravel().astype(bool), ma.ravel().astype(bool)
    # refine on inliers only (least squares)
    H, _ = cv2.findHomography(p1[mh], p2[mh], 0)
    A, _ = cv2.estimateAffinePartial2D(p1[ma], p2[ma], method=cv2.LMEDS)

    def err_h(Hm, a, b):
        q = cv2.perspectiveTransform(a.reshape(-1, 1, 2), Hm).reshape(-1, 2)
        return np.linalg.norm(q - b, axis=1)

    eh = err_h(H, p1[mh], p2[mh])
    ea = np.linalg.norm(p1[ma] @ A[:, :2].T + A[:, 2] - p2[ma], axis=1)
    grid = np.zeros((4, 4), int)
    for x, y in p1[mh]:
        grid[min(3, int(4 * y / FULL[1])), min(3, int(4 * x / FULL[0]))] += 1
    sc = float(np.hypot(A[0, 0], A[1, 0]))
    Ainv = cv2.invertAffineTransform(A)
    return {
        "H_ref_to_video": H.tolist(),
        "H_video_to_ref": np.linalg.inv(H).tolist(),
        "S_ref_to_video": A.tolist(),
        "S_video_to_ref": Ainv.tolist(),
        "similarity": {"tx": round(float(A[0, 2]), 2), "ty": round(float(A[1, 2]), 2),
                       "rot_deg": round(float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))), 3),
                       "scale": round(sc, 5)},
        "matches": int(len(good)), "inliers_h": int(mh.sum()), "inliers_s": int(ma.sum()),
        "resid_h_px": {"median": round(float(np.median(eh)), 2), "p90": round(float(np.percentile(eh, 90)), 2)},
        "resid_s_px": {"median": round(float(np.median(ea)), 2), "p90": round(float(np.percentile(ea, 90)), 2)},
        "inlier_spread": round(float((grid >= 3).mean()), 3),
        "coords": "4K pixels (3840x2160) on both sides",
    }


def register_frames(frames, ref=None, use_mask=False):
    """frames: list of BGR frames of one video (any size, same camera setup)."""
    ref = load_ref() if ref is None else _resize(ref, WORK_W)
    frames = [_resize(f, WORK_W) for f in frames]
    bg = median_frame(frames) if len(frames) > 1 else frames[0]
    mask = static_mask(frames, bg) if (use_mask and len(frames) >= 5) else None
    r = estimate(ref, bg, img_mask=mask)
    r["n_frames"] = len(frames)
    r["masked_static_fraction"] = None if mask is None else round(float((mask > 0).mean()), 3)
    return r, bg


def register_video(video, ref=None, n=15, span=None, use_mask=False):
    t0 = time.time()
    frames, idx = sample_frames(video, n=n, span=span)
    t1 = time.time()
    r, _ = register_frames(frames, ref, use_mask)
    r["frame_indices"] = idx
    r["runtime_s"] = {"decode": round(t1 - t0, 2), "match": round(time.time() - t1, 2)}
    return r


# ------------------------------------------------------------------ point helpers
def _apply(H, pts):
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(H, np.float64)).reshape(-1, 2)


def to_video(pts_ref, r):
    """Reference 4K px -> this video's 4K px."""
    return _apply(r["H_ref_to_video"], pts_ref)


def to_ref(pts_video, r):
    """This video's 4K px -> reference 4K px."""
    return _apply(r["H_video_to_ref"], pts_video)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True, help="video file (proxy or original)")
    ap.add_argument("--ref", default=REF_PATH, help="reference image (default: C3897 median)")
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--span", type=float, default=None, help="only use the first SPAN seconds")
    ap.add_argument("--mask", action="store_true", help="mask moving pixels before SIFT (off by default)")
    ap.add_argument("--out", help="write the result JSON here")
    a = ap.parse_args()
    r = register_video(a.video, cv2.imread(a.ref), a.frames, a.span, a.mask)
    txt = json.dumps(r, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt)
    s = r["similarity"]
    print(f"{os.path.basename(a.video)}: shift ({s['tx']:+.1f}, {s['ty']:+.1f}) px, rot {s['rot_deg']:+.2f} deg, "
          f"scale {s['scale']:.4f}; inliers H {r['inliers_h']} (resid median {r['resid_h_px']['median']} px); "
          f"time {r['runtime_s']}", file=sys.stderr)
    if not a.out:
        print(txt)


if __name__ == "__main__":
    main()
