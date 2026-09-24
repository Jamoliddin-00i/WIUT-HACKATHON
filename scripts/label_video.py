"""Keyboard-driven local video annotator. Uses Tkinter, not OpenCV HighGUI."""

from __future__ import annotations

import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog

import cv2
from PIL import Image, ImageTk


CLASSES = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle",
    "fire_smoke",
]

CONTROLS = """Controls
SPACE = pause/play
A / D = back/forward 1 second
Z / X = previous/next frame
S = mark event START
E = mark event END + choose class
R = remove last event
Q = save and quit"""


class VideoLabeler:
    def __init__(self, video_path: Path, output_path: Path, smoke: bool = False):
        self.video_path = video_path
        self.output_path = output_path
        self.smoke = smoke
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.fps <= 0 or self.frame_count <= 0:
            self.capture.release()
            raise RuntimeError(f"Invalid video metadata: {video_path}")
        self.duration = self.frame_count / self.fps

        if output_path.exists():
            self.labels = json.loads(output_path.read_text(encoding="utf-8"))
        else:
            self.labels = {}
        self.labels.setdefault(video_path.name, {
            "duration": self.duration,
            "fps": self.fps,
            "events": [],
        })
        self.events = self.labels[video_path.name]["events"]

        self.root = tk.Tk()
        self.root.title(f"WIUT Labeler — {video_path.name}")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<KeyPress>", self.on_key)
        self.image_label = tk.Label(self.root)
        self.image_label.pack()
        self.status = tk.StringVar()
        tk.Label(self.root, textvariable=self.status, anchor="w").pack(fill="x")

        screen_width = max(640, self.root.winfo_screenwidth() - 80)
        screen_height = max(360, self.root.winfo_screenheight() - 140)
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = min(1.0, 1280 / width, 720 / height,
                    screen_width / width, screen_height / height)
        self.display_size = (max(1, int(width * scale)),
                             max(1, int(height * scale)))

        self.frame_index = 0
        self.frame = None
        self.paused = True
        self.start_time = None
        self.closed = False
        self.load_frame(0)
        self.root.after(30, self.tick)
        if smoke:
            self.root.after(1000, self.close)

    def load_frame(self, index: int) -> bool:
        index = min(max(0, index), self.frame_count - 1)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        if not ok:
            return False
        self.frame_index = index
        self.frame = frame
        self.render()
        return True

    def render(self) -> None:
        frame = cv2.resize(self.frame, self.display_size,
                           interpolation=cv2.INTER_AREA)
        t_sec = self.frame_index / self.fps
        cv2.putText(frame, f"{self.video_path.name}  {t_sec:.2f}s / {self.duration:.2f}s",
                    (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255),
                    2, cv2.LINE_AA)
        if self.start_time is not None:
            cv2.putText(frame, f"START: {self.start_time:.2f}s", (16, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255),
                        2, cv2.LINE_AA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb), master=self.root)
        self.image_label.configure(image=photo)
        self.image_label.image = photo
        self.status.set(f"{'PAUSED' if self.paused else 'PLAYING'}  |  "
                        f"{len(self.events)} events  |  Space play/pause  |  "
                        "A/D ±1s  Z/X ±1 frame  S start  E end  R undo  Q save")

    def tick(self) -> None:
        if self.closed:
            return
        if not self.paused:
            if self.frame_index + 1 >= self.frame_count:
                self.paused = True
                self.render()
            else:
                self.load_frame(self.frame_index + 1)
        self.root.after(30, self.tick)

    def save(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(self.labels, indent=2) + "\n",
                             encoding="utf-8")
        temp_path.replace(self.output_path)

    def on_key(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        if key == "space":
            self.paused = not self.paused
            self.render()
        elif key in ("a", "d", "z", "x"):
            self.paused = True
            offset = {"a": -round(self.fps), "d": round(self.fps),
                      "z": -1, "x": 1}[key]
            self.load_frame(self.frame_index + offset)
        elif key == "s":
            self.start_time = self.frame_index / self.fps
            print(f"START marked: {self.start_time:.3f}s")
            self.render()
        elif key == "e":
            self.finish_event()
        elif key == "r":
            if self.events:
                removed = self.events.pop()
                self.save()
                print("Removed:", removed)
                self.render()
        elif key == "q":
            self.close()

    def finish_event(self) -> None:
        if self.start_time is None:
            messagebox.showinfo("No start", "Press S to mark the event start first.",
                                parent=self.root)
            return
        end_time = self.frame_index / self.fps
        if end_time <= self.start_time:
            messagebox.showinfo("Invalid interval", "The end must be after the start.",
                                parent=self.root)
            return
        class_list = "\n".join(f"{i:2}. {name}"
                               for i, name in enumerate(CLASSES, 1))
        choice = simpledialog.askinteger(
            "Event class", f"Choose event class:\n\n{class_list}",
            parent=self.root, minvalue=1, maxvalue=len(CLASSES))
        if choice is None:
            return
        event = [round(self.start_time, 3), round(end_time, 3),
                 CLASSES[choice - 1]]
        self.events.append(event)
        self.events.sort(key=lambda item: item[0])
        self.save()
        self.start_time = None
        print("Added:", event)
        self.render()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if not self.smoke:
            self.save()
            print(f"Saved to {self.output_path}")
        self.capture.release()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "dev_labels.json")
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    print(CONTROLS)
    VideoLabeler(args.video, args.out, args.smoke).run()


if __name__ == "__main__":
    main()
