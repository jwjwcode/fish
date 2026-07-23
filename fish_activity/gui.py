"""Tkinter GUI for live fish-feeding activity preview."""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - user-facing startup guard
    raise SystemExit(
        "Missing dependency: "
        f"{exc.name}\nInstall dependencies with: python3 -m pip install -r requirements.txt"
    ) from exc

from fish_activity.detectors.unsupervised import (
    UnsupervisedSplashDetector,
    compute_flow_magnitude,
    create_flow_estimator,
)
from fish_activity.excel_io import write_score_workbook
from fish_activity.pipeline_v1 import PROJECT_ROOT, parse_args
from fish_activity.render import (
    bottom_panel_layout,
    overlay_visuals,
    resize_frame,
    roi_bounds,
)
from fish_activity.scoring import average_score_history, score_flow
from fish_activity.video_io import is_stream_source, open_writer, source_name


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "tune_less_bubble.json"
VIDEO_EXTENSIONS = (
    ("Video files", "*.mp4 *.avi *.mov *.mkv"),
    ("All files", "*.*"),
)


@dataclass
class LiveSettings:
    threshold: float
    seg_weight: float
    flow_weight: float


@dataclass
class PreviewUpdate:
    frame_bgr: np.ndarray
    source_frame: int
    time_s: float
    seg_score: float
    flow_score: float
    total_activity: float
    prev10_total: float
    threshold: float
    status: str
    processing_fps: float


@dataclass
class WorkerDone:
    message: str
    excel_path: str | None = None
    video_path: str | None = None
    final_score: float | None = None


@dataclass
class WorkerError:
    message: str


GuiMessage = PreviewUpdate | WorkerDone | WorkerError


def _put_message(output_queue: queue.Queue[GuiMessage], message: GuiMessage) -> None:
    output_queue.put(message)


def _fit_for_display(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return frame
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 0.999:
        return frame
    target_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)


def _bgr_to_photo(frame: np.ndarray) -> tk.PhotoImage:
    ok, png_data = cv2.imencode(".png", frame)
    if ok:
        return tk.PhotoImage(data=png_data.tobytes(), format="png")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    return tk.PhotoImage(data=header + rgb.tobytes(), format="PPM")


def _build_preview_args(
    video_path: str,
    config_path: str,
    resize_width: int,
) -> object:
    argv = [
        video_path,
        "--frame-step",
        "1",
        "--resize-width",
        str(resize_width),
        "--headless",
        "on",
        "--decision-mode",
        "off",
        "--progress-interval",
        "0",
    ]
    if config_path:
        argv.extend(["--config", config_path])
    return parse_args(argv)


def _read_config_defaults(config_path: str) -> tuple[float, float]:
    argv = ["_preview_defaults.mp4", "--headless", "on"]
    if config_path:
        argv.extend(["--config", config_path])
    args = parse_args(argv)
    return float(args.seg_weight), float(args.flow_weight)


class PreviewWorker(threading.Thread):
    def __init__(
        self,
        args: object,
        settings: LiveSettings,
        settings_lock: threading.Lock,
        output_queue: queue.Queue[GuiMessage],
        stop_event: threading.Event,
        pause_event: threading.Event,
        real_time: bool,
        excel_path: Path | None,
        video_path: Path | None,
    ) -> None:
        super().__init__(daemon=True)
        self.args = args
        self.settings = settings
        self.settings_lock = settings_lock
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.real_time = real_time
        self.excel_path = excel_path
        self.video_path = video_path

    def run(self) -> None:
        try:
            self._run()
        except SystemExit as exc:
            _put_message(self.output_queue, WorkerError(str(exc)))
        except Exception as exc:  # pragma: no cover - protects GUI event loop
            _put_message(self.output_queue, WorkerError(f"{type(exc).__name__}: {exc}"))

    def _run(self) -> None:
        input_source = str(self.args.input)
        if not is_stream_source(input_source) and not Path(input_source).exists():
            raise SystemExit(f"Input video does not exist: {input_source}")

        cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            raise SystemExit(f"Could not open input video/source: {input_source}")

        writer: cv2.VideoWriter | None = None
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps <= 0:
                fps = 25.0

            detector = UnsupervisedSplashDetector(self.args)
            flow_estimator, flow_method = create_flow_estimator(self.args.flow_method)
            prev_gray_blur: np.ndarray | None = None
            score_history: deque[dict[str, float]] = deque(maxlen=10)
            compact_score_rows: list[dict[str, float]] = []
            score_sum = 0.0
            score_count = 0
            processed_index = 0
            processed_start = time.monotonic()
            playback_start = processed_start
            pause_started: float | None = None

            while not self.stop_event.is_set():
                if self.pause_event.is_set():
                    if pause_started is None:
                        pause_started = time.monotonic()
                    time.sleep(0.05)
                    continue
                if pause_started is not None:
                    playback_start += time.monotonic() - pause_started
                    pause_started = None

                ok, frame = cap.read()
                if not ok:
                    break

                source_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                if source_frame_index < 0:
                    source_frame_index = processed_index

                frame = resize_frame(frame, self.args.resize_width)
                height, width = frame.shape[:2]
                x0, x1 = roi_bounds(width)
                roi = frame[:, x0:x1]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

                if flow_method == "none":
                    raw_flow_mag = np.zeros_like(gray_blur, dtype=np.float32)
                else:
                    raw_flow_mag = compute_flow_magnitude(
                        gray_blur,
                        prev_gray_blur,
                        flow_estimator,
                    )

                detector_result = detector.compute(
                    roi,
                    gray_blur,
                    prev_gray_blur,
                    raw_flow_mag,
                    processed_index,
                )
                mask = detector_result.mask
                seg_activity_pct = detector_result.activity_score

                if flow_method == "none":
                    flow_mag = raw_flow_mag
                    flow_activity = 0.0
                else:
                    flow_mask = mask if self.args.flow_mask == "segmentation" else None
                    flow_mag, flow_activity, _raw_flow_activity = score_flow(
                        raw_flow_mag,
                        self.args.flow_percentile,
                        flow_mask,
                        self.args.flow_min_mask_pixels,
                    )

                with self.settings_lock:
                    threshold = self.settings.threshold
                    seg_weight = self.settings.seg_weight
                    flow_weight = self.settings.flow_weight

                seg_score = seg_weight * seg_activity_pct
                flow_score = flow_weight * flow_activity
                total_activity = seg_score + flow_score
                score_sum += total_activity
                score_count += 1
                score_averages = average_score_history(score_history)
                prev10_total = score_averages["total_activity_prev10_avg"]
                status = "ACTIVE" if total_activity >= threshold else "LOW"
                time_s = max(0.0, source_frame_index / fps)
                compact_score_rows.append(
                    {
                        "time_s": time_s,
                        "current_segmentation_score": seg_score,
                        "current_optical_flow_score": flow_score,
                        "current_total_activity": total_activity,
                        "previous_10_total_activity_average": prev10_total,
                    }
                )

                panel_height, preview_width, text_area_width = bottom_panel_layout(
                    width,
                    height,
                    self.args.preview_width,
                )
                annotated = overlay_visuals(
                    frame,
                    x0,
                    x1,
                    mask,
                    flow_mag,
                    {
                        "seg_score": seg_score,
                        "flow_score": flow_score,
                        "total_activity": total_activity,
                        "total_activity_prev10_avg": prev10_total,
                    },
                    time_s,
                    preview_width,
                    panel_height,
                    text_area_width,
                )
                if self.video_path is not None and writer is None:
                    annotated_height, annotated_width = annotated.shape[:2]
                    writer = open_writer(
                        self.video_path,
                        fps,
                        (annotated_width, annotated_height),
                        self.args.codec,
                    )
                if writer is not None:
                    writer.write(annotated)

                elapsed = max(time.monotonic() - processed_start, 1e-6)
                processing_fps = float(processed_index + 1) / elapsed
                _put_message(
                    self.output_queue,
                    PreviewUpdate(
                        frame_bgr=annotated,
                        source_frame=source_frame_index,
                        time_s=time_s,
                        seg_score=seg_score,
                        flow_score=flow_score,
                        total_activity=total_activity,
                        prev10_total=prev10_total,
                        threshold=threshold,
                        status=status,
                        processing_fps=processing_fps,
                    ),
                )

                score_history.append(
                    {
                        "segmentation_score": seg_score,
                        "optical_flow_score": flow_score,
                        "total_activity": total_activity,
                    }
                )
                prev_gray_blur = gray_blur
                processed_index += 1

                if self.real_time:
                    target_elapsed = processed_index / fps
                    while not self.stop_event.is_set():
                        if self.pause_event.is_set():
                            break
                        remaining = target_elapsed - (time.monotonic() - playback_start)
                        if remaining <= 0:
                            break
                        time.sleep(min(remaining, 0.02))
            final_score = score_sum / float(score_count) if score_count else None
            if self.excel_path is not None:
                write_score_workbook(
                    self.excel_path,
                    compact_score_rows,
                    final_score,
                )
        finally:
            if writer is not None:
                writer.release()
            cap.release()

        final_score = score_sum / float(score_count) if score_count else None
        excel_path_text = str(self.excel_path) if self.excel_path is not None else None
        video_path_text = str(self.video_path) if self.video_path is not None else None
        saved_parts = []
        if excel_path_text is not None:
            saved_parts.append(f"Excel: {excel_path_text}")
        if video_path_text is not None:
            saved_parts.append(f"video: {video_path_text}")
        saved_text = f". Saved {'; '.join(saved_parts)}" if saved_parts else ""
        if self.stop_event.is_set():
            message = f"Stopped{saved_text}"
            _put_message(
                self.output_queue,
                WorkerDone(message, excel_path_text, video_path_text, final_score),
            )
        else:
            message = f"Finished video{saved_text}"
            _put_message(
                self.output_queue,
                WorkerDone(message, excel_path_text, video_path_text, final_score),
            )


class FeedingActivityGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Fish Feeding Activity V1")
        self.geometry("1420x900")
        self.minsize(980, 640)

        self.video_path = tk.StringVar()
        self.config_path = tk.StringVar(value=self._default_config_path())
        self.threshold = tk.DoubleVar(value=1.0)
        self.seg_weight = tk.DoubleVar(value=1.0)
        self.flow_weight = tk.DoubleVar(value=0.03)
        self.resize_width = tk.IntVar(value=1280)
        self.real_time = tk.BooleanVar(value=True)
        self.save_excel = tk.BooleanVar(value=True)
        self.save_video = tk.BooleanVar(value=False)
        self.excel_path = tk.StringVar()
        self.output_video_path = tk.StringVar()
        self.status_text = tk.StringVar(value="Select a video and press Start.")
        self.score_text = tk.StringVar(value="Seg 0.000 | Flow 0.000 | Total 0.000")
        self.time_text = tk.StringVar(value="Time 0.0s | Frame 0")

        self.settings = LiveSettings(
            threshold=self.threshold.get(),
            seg_weight=self.seg_weight.get(),
            flow_weight=self.flow_weight.get(),
        )
        self.settings_lock = threading.Lock()
        self.output_queue: queue.Queue[GuiMessage] = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.worker: PreviewWorker | None = None
        self.current_photo: tk.PhotoImage | None = None

        self._build_ui()
        self._load_selected_config_defaults(show_errors=False)
        self._install_setting_traces()
        self.after(30, self._poll_worker)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _default_config_path(self) -> str:
        if DEFAULT_CONFIG.exists():
            return str(DEFAULT_CONFIG)
        configs = sorted((PROJECT_ROOT / "configs").glob("*.json"))
        return str(configs[0]) if configs else ""

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        controls = ttk.Frame(root, width=360)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        controls.grid_propagate(False)
        controls.columnconfigure(0, weight=1)

        preview_area = ttk.Frame(root)
        preview_area.grid(row=0, column=1, sticky="nsew")
        preview_area.columnconfigure(0, weight=1)
        preview_area.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(preview_area, anchor=tk.CENTER)
        self.image_label.grid(row=0, column=0, sticky="nsew")

        self._add_path_controls(controls)
        self._add_output_controls(controls)
        self._add_numeric_controls(controls)
        self._add_action_controls(controls)
        self._add_status_controls(controls)

    def _add_path_controls(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Video").grid(row=0, column=0, sticky="w")
        video_row = ttk.Frame(parent)
        video_row.grid(row=1, column=0, sticky="ew", pady=(2, 10))
        video_row.columnconfigure(0, weight=1)
        ttk.Entry(video_row, textvariable=self.video_path).grid(
            row=0,
            column=0,
            sticky="ew",
        )
        ttk.Button(video_row, text="Browse", command=self._browse_video).grid(
            row=0,
            column=1,
            padx=(6, 0),
        )

        ttk.Label(parent, text="Config").grid(row=2, column=0, sticky="w")
        config_row = ttk.Frame(parent)
        config_row.grid(row=3, column=0, sticky="ew", pady=(2, 10))
        config_row.columnconfigure(0, weight=1)
        config_values = [
            str(path) for path in sorted((PROJECT_ROOT / "configs").glob("*.json"))
        ]
        self.config_combo = ttk.Combobox(
            config_row,
            textvariable=self.config_path,
            values=config_values,
        )
        self.config_combo.grid(row=0, column=0, sticky="ew")
        self.config_combo.bind("<<ComboboxSelected>>", self._on_config_selected)
        ttk.Button(config_row, text="Browse", command=self._browse_config).grid(
            row=0,
            column=1,
            padx=(6, 0),
        )

    def _add_output_controls(self, parent: ttk.Frame) -> None:
        output = ttk.LabelFrame(parent, text="Output", padding=10)
        output.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        output.columnconfigure(0, weight=1)

        ttk.Checkbutton(
            output,
            text="Save Excel",
            variable=self.save_excel,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Entry(output, textvariable=self.excel_path).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(output, text="Browse", command=self._browse_excel).grid(
            row=1,
            column=1,
            padx=(6, 0),
            pady=(8, 0),
        )

        ttk.Checkbutton(
            output,
            text="Save Video",
            variable=self.save_video,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Entry(output, textvariable=self.output_video_path).grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(output, text="Browse", command=self._browse_output_video).grid(
            row=3,
            column=1,
            padx=(6, 0),
            pady=(8, 0),
        )

    def _add_numeric_controls(self, parent: ttk.Frame) -> None:
        numeric = ttk.LabelFrame(parent, text="Live Parameters", padding=10)
        numeric.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        numeric.columnconfigure(1, weight=1)

        self._add_entry(numeric, 0, "Threshold", self.threshold)
        self._add_entry(numeric, 1, "Seg weight", self.seg_weight)
        self._add_entry(numeric, 2, "Flow weight", self.flow_weight)
        self._add_entry(numeric, 3, "Resize width", self.resize_width)
        ttk.Checkbutton(
            numeric,
            text="Real-time playback",
            variable=self.real_time,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Button(
            numeric,
            text="Load Config Weights",
            command=lambda: self._load_selected_config_defaults(show_errors=True),
        ).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def _add_entry(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.Variable,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=variable, width=12).grid(
            row=row,
            column=1,
            sticky="ew",
            pady=4,
        )

    def _add_action_controls(self, parent: ttk.Frame) -> None:
        actions = ttk.Frame(parent)
        actions.grid(row=6, column=0, sticky="ew", pady=(0, 10))
        for column in range(3):
            actions.columnconfigure(column, weight=1)
        self.start_button = ttk.Button(actions, text="Start", command=self._start)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.pause_button = ttk.Button(
            actions,
            text="Pause",
            command=self._toggle_pause,
            state=tk.DISABLED,
        )
        self.pause_button.grid(row=0, column=1, sticky="ew", padx=4)
        self.stop_button = ttk.Button(
            actions,
            text="Stop",
            command=self._stop,
            state=tk.DISABLED,
        )
        self.stop_button.grid(row=0, column=2, sticky="ew", padx=(4, 0))

    def _add_status_controls(self, parent: ttk.Frame) -> None:
        status = ttk.LabelFrame(parent, text="Current Scores", padding=10)
        status.grid(row=7, column=0, sticky="ew")
        ttk.Label(status, textvariable=self.score_text, justify=tk.LEFT).grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(status, textvariable=self.time_text, justify=tk.LEFT).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(6, 0),
        )
        ttk.Label(parent, textvariable=self.status_text, wraplength=310).grid(
            row=8,
            column=0,
            sticky="ew",
            pady=(10, 0),
        )

    def _install_setting_traces(self) -> None:
        for variable in (self.threshold, self.seg_weight, self.flow_weight):
            variable.trace_add("write", lambda *_args: self._update_live_settings())

    def _update_live_settings(self) -> None:
        try:
            threshold = max(0.0, float(self.threshold.get()))
            seg_weight = max(0.0, float(self.seg_weight.get()))
            flow_weight = max(0.0, float(self.flow_weight.get()))
        except (tk.TclError, ValueError):
            return
        with self.settings_lock:
            self.settings.threshold = threshold
            self.settings.seg_weight = seg_weight
            self.settings.flow_weight = flow_weight

    def _browse_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=VIDEO_EXTENSIONS,
            initialdir=str(PROJECT_ROOT),
        )
        if path:
            self.video_path.set(path)
            self.excel_path.set(str(self._default_output_path(path, ".xlsx")))
            self.output_video_path.set(str(self._default_output_path(path, ".mp4")))

    def _browse_config(self) -> None:
        path = filedialog.askopenfilename(
            title="Select config",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
            initialdir=str(PROJECT_ROOT / "configs"),
        )
        if path:
            self.config_path.set(path)
            self._load_selected_config_defaults(show_errors=True)

    def _browse_excel(self) -> None:
        initial_file = Path(self.excel_path.get()).name if self.excel_path.get() else ""
        initial_dir = PROJECT_ROOT / "results" / "gui"
        initial_dir.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="Save Excel",
            defaultextension=".xlsx",
            filetypes=(("Excel workbook", "*.xlsx"), ("All files", "*.*")),
            initialdir=str(initial_dir),
            initialfile=initial_file,
        )
        if path:
            self.excel_path.set(path)
            self.save_excel.set(True)

    def _browse_output_video(self) -> None:
        initial_file = (
            Path(self.output_video_path.get()).name
            if self.output_video_path.get()
            else ""
        )
        initial_dir = PROJECT_ROOT / "results" / "gui"
        initial_dir.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="Save processed video",
            defaultextension=".mp4",
            filetypes=(("MP4 video", "*.mp4"), ("All files", "*.*")),
            initialdir=str(initial_dir),
            initialfile=initial_file,
        )
        if path:
            self.output_video_path.set(path)
            self.save_video.set(True)

    def _default_output_path(self, video_path: str, suffix: str) -> Path:
        stem = source_name(video_path)
        return PROJECT_ROOT / "results" / "gui" / f"{stem}_activity_v1{suffix}"

    def _on_config_selected(self, _event: object) -> None:
        self._load_selected_config_defaults(show_errors=True)

    def _load_selected_config_defaults(self, show_errors: bool) -> None:
        try:
            seg_weight, flow_weight = _read_config_defaults(self.config_path.get())
        except SystemExit as exc:
            if show_errors:
                messagebox.showerror("Config error", str(exc))
            return
        self.seg_weight.set(seg_weight)
        self.flow_weight.set(flow_weight)
        self._update_live_settings()

    def _start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return

        video_path = self.video_path.get().strip()
        if not video_path:
            messagebox.showerror("Missing video", "Select a video first.")
            return

        try:
            resize_width = max(0, int(self.resize_width.get()))
            excel_path = None
            if self.save_excel.get():
                excel_path_text = self.excel_path.get().strip()
                if not excel_path_text:
                    excel_path = self._default_output_path(video_path, ".xlsx")
                    self.excel_path.set(str(excel_path))
                else:
                    excel_path = Path(excel_path_text)
            output_video_path = None
            if self.save_video.get():
                output_video_text = self.output_video_path.get().strip()
                if not output_video_text:
                    output_video_path = self._default_output_path(video_path, ".mp4")
                    self.output_video_path.set(str(output_video_path))
                else:
                    output_video_path = Path(output_video_text)
            args = _build_preview_args(
                video_path,
                self.config_path.get().strip(),
                resize_width,
            )
        except (SystemExit, tk.TclError, ValueError) as exc:
            messagebox.showerror("Start error", str(exc))
            return

        self._update_live_settings()
        self.output_queue = queue.Queue(maxsize=4)
        self.stop_event.clear()
        self.pause_event.clear()
        self.worker = PreviewWorker(
            args=args,
            settings=self.settings,
            settings_lock=self.settings_lock,
            output_queue=self.output_queue,
            stop_event=self.stop_event,
            pause_event=self.pause_event,
            real_time=bool(self.real_time.get()),
            excel_path=excel_path,
            video_path=output_video_path,
        )
        self.worker.start()
        self.start_button.configure(state=tk.DISABLED)
        self.pause_button.configure(state=tk.NORMAL, text="Pause")
        self.stop_button.configure(state=tk.NORMAL)
        self.status_text.set("Processing...")

    def _toggle_pause(self) -> None:
        if self.worker is None or not self.worker.is_alive():
            return
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.pause_button.configure(text="Pause")
            self.status_text.set("Processing...")
        else:
            self.pause_event.set()
            self.pause_button.configure(text="Resume")
            self.status_text.set("Paused")

    def _stop(self) -> None:
        self.stop_event.set()
        self.pause_event.clear()
        self.status_text.set("Stopping...")

    def _poll_worker(self) -> None:
        try:
            message = self.output_queue.get_nowait()
        except queue.Empty:
            self.after(30, self._poll_worker)
            return

        if isinstance(message, PreviewUpdate):
            self._show_preview(message)
        else:
            self._handle_terminal_message(message)

        delay_ms = 1 if not self.output_queue.empty() else 30
        self.after(delay_ms, self._poll_worker)

    def _show_preview(self, update: PreviewUpdate) -> None:
        max_width = max(480, self.image_label.winfo_width() - 8)
        max_height = max(360, self.image_label.winfo_height() - 8)
        display_frame = _fit_for_display(update.frame_bgr, max_width, max_height)
        self.current_photo = _bgr_to_photo(display_frame)
        self.image_label.configure(image=self.current_photo)

        self.score_text.set(
            f"Seg {update.seg_score:.3f} | Flow {update.flow_score:.3f} | "
            f"Total {update.total_activity:.3f} | Prev10 {update.prev10_total:.3f}"
        )
        self.time_text.set(
            f"Time {update.time_s:.1f}s | Frame {update.source_frame} | "
            f"Threshold {update.threshold:.3f} | {update.status} | "
            f"{update.processing_fps:.1f} fps"
        )

    def _handle_terminal_message(self, message: WorkerDone | WorkerError) -> None:
        if isinstance(message, WorkerError):
            self.status_text.set(f"Error: {message.message}")
            messagebox.showerror("Processing error", message.message)
        else:
            self.status_text.set(message.message)
            if message.final_score is not None:
                self.score_text.set(
                    f"Final average total activity: {message.final_score:.6f}"
                )
        self.start_button.configure(state=tk.NORMAL)
        self.pause_button.configure(state=tk.DISABLED, text="Pause")
        self.stop_button.configure(state=tk.DISABLED)

    def _on_close(self) -> None:
        self.stop_event.set()
        self.destroy()


def main() -> int:
    app = FeedingActivityGui()
    app.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
