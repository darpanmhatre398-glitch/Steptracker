"""
Step tracker: matches live frames (camera or video file) against reference images
of the unit after each step, and reports "Step N complete".

refs/ layout (state AFTER each step; names sort alphabetically = procedure order):
    refs/00_start/          images and/or videos of the unit before any step
    refs/01_remove_cover/   images and/or videos of the unit after step 1
    refs/02_remove_screws/  ...

Usage:
    python track.py --refs refs --source 0                # webcam
    python track.py --refs refs --source recording.mp4    # video file
    python track.py --refs refs --source 0 --roi          # pick a crop region first
"""
import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv"}


class Embedder:
    def __init__(self, model_name, device):
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def __call__(self, frames_bgr):
        imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
        inputs = self.proc(images=imgs, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        # Per-patch features keep spatial layout, which matters for a fixed camera:
        # dot product of two outputs = mean cosine similarity over patches.
        patches = torch.nn.functional.normalize(out.last_hidden_state[:, 1:], dim=2)
        return (patches.flatten(1) / patches.shape[1] ** 0.5).cpu().numpy()


def crop(frame, roi):
    if roi is None:
        return frame
    x, y, w, h = roi
    return frame[y:y + h, x:x + w]


def load_reference_frames(folder, roi, max_video_frames=30):
    frames = []
    for p in sorted(folder.iterdir()):
        ext = p.suffix.lower()
        if ext in IMG_EXT:
            img = cv2.imread(str(p))
            if img is not None:
                frames.append(crop(img, roi))
        elif ext in VID_EXT:
            cap = cv2.VideoCapture(str(p))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in np.linspace(0, max(n - 1, 0), min(max_video_frames, max(n, 1))).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, img = cap.read()
                if ok:
                    frames.append(crop(img, roi))
            cap.release()
    return frames


def pick_roi(source):
    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read a frame to select the ROI")
    r = cv2.selectROI("Select the unit area, then press ENTER", frame, showCrosshair=False)
    cv2.destroyAllWindows()
    return tuple(int(v) for v in r) if r[2] and r[3] else None


class StepTracker:
    """Advances only when the next state is confidently matched for several consecutive frames."""

    def __init__(self, names, window, margin, confirm_frames):
        self.names = names
        self.current = 0                      # index of the last confirmed state
        self.window = deque(maxlen=window)    # smoothing over recent similarity vectors
        self.margin = margin
        self.confirm = confirm_frames
        self.streak = {}                      # candidate state -> consecutive frames matched
        self.events = []

    def update(self, sims):
        self.window.append(sims)
        avg = np.mean(self.window, axis=0)
        order = np.argsort(avg)[::-1]
        best = int(order[0])
        second = float(avg[order[1]]) if len(order) > 1 else -1.0
        confident = float(avg[best]) - second >= self.margin

        self.streak = {best: self.streak.get(best, 0) + 1} if confident else {}
        status = f"state: {self.names[self.current]}"
        if best != self.current and self.streak.get(best, 0) >= self.confirm:
            if best == self.current + 1:
                self.current = best
                self.events.append(("complete", best))
                status = f"STEP {best} COMPLETE: {self.names[best]}"
            elif best > self.current + 1:
                skipped = list(range(self.current + 1, best))
                self.current = best
                self.events.append(("skipped", skipped))
                status = f"WARNING: skipped step(s) {skipped}, now at {self.names[best]}"
            else:
                self.events.append(("regressed", best))
                self.current = best
                status = f"WARNING: went back to {self.names[best]}"
            self.streak = {}
        return status, avg, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="refs")
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--roi", action="store_true", help="select the unit region with the mouse")
    ap.add_argument("--window", type=int, default=5, help="frames of smoothing")
    ap.add_argument("--margin", type=float, default=0.02, help="min similarity lead over 2nd best state")
    ap.add_argument("--confirm", type=int, default=8, help="consecutive frames needed to confirm a step")
    ap.add_argument("--every", type=int, default=2, help="process every Nth frame")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    roi = pick_roi(source) if args.roi else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed = Embedder(args.model, device)

    folders = sorted(p for p in Path(args.refs).iterdir() if p.is_dir())
    if len(folders) < 2:
        raise SystemExit(f"Need at least 2 state folders in {args.refs}/ (start + step 1)")
    names, protos = [], []
    for f in folders:
        frames = load_reference_frames(f, roi)
        if not frames:
            raise SystemExit(f"No images/videos found in {f}")
        e = embed(frames)
        names.append(f.name)
        protos.append(e)
        print(f"loaded {f.name}: {len(frames)} reference frames")
    # score a frame against a state by its best-matching reference frame
    tracker = StepTracker(names, args.window, args.margin, args.confirm)

    cap = cv2.VideoCapture(source)
    i = 0
    status = f"state: {names[0]}"
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % args.every:
            continue
        e = embed([crop(frame, roi)])[0]
        sims = np.array([float((p @ e).max()) for p in protos])
        new_status, avg, best = tracker.update(sims)
        if new_status != f"state: {names[tracker.current]}":
            print(f"[{time.time() - t0:6.1f}s] {new_status}")
        status = new_status

        if not args.no_display:
            vis = frame.copy()
            cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            for k, n in enumerate(names):
                color = (0, 255, 0) if k <= tracker.current else (180, 180, 180)
                cv2.putText(vis, f"{n}: {avg[k]:.2f}", (10, 65 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            cv2.imshow("step tracker (q to quit)", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()
    print(f"Final state: {names[tracker.current]}  ({tracker.current}/{len(names) - 1} steps)")


if __name__ == "__main__":
    main()
