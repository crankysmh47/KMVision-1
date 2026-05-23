"""
Interactive labeling tool for real-world Kaplan-Meier charts.

Usage:
    python real_dataset/labeler.py
    python real_dataset/labeler.py --type km
    python real_dataset/labeler.py --type km --image chart_001_km.png

Keyboard shortcuts:
    D          Discard image (move to discarded/)
    S / Ctrl+S Save label and go to next
    Right      Next image (without saving)
    Left       Previous image
    Ctrl+Enter Save label and go to next
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from PIL import Image, ImageTk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import (  # noqa: E402
    LABELING_STATE_FILE,
    chart_number,
    discarded_dir,
    ensure_dirs,
    flatten_nested_images,
    images_dir,
    inbox_dir,
    label_path_for_image,
    labels_dir,
    next_curated_name,
    unlabeled_queue,
)


DEFAULT_TIME_POINTS = [0, 6, 12, 18, 24, 30, 36, 48, 60]


def default_km_label() -> dict[str, Any]:
    return {
        "chart_type": "kaplan_meier",
        "axes": {
            "x": {"label": "Time (months)", "max_value": 0.0},
            "y": {"label": "Survival Probability", "max_value": 1.0},
        },
        "arms": [],
        "at_risk_table": {
            "time_points": DEFAULT_TIME_POINTS.copy(),
            "rows": [],
        },
    }


class KMChartLabeler(tk.Tk):
    def __init__(self, chart_type: str = "km", start_image: str | None = None) -> None:
        super().__init__()
        self.chart_type = chart_type
        ensure_dirs(chart_type)
        flatten_nested_images(chart_type)

        self.title(f"KMVision Labeler — {chart_type.upper()}")
        self.geometry("1400x900")
        self.minsize(1100, 700)

        self.queue = unlabeled_queue(chart_type)
        self.index = 0
        self.photo: ImageTk.PhotoImage | None = None
        self.current_image_path: str | None = None
        self.draft = default_km_label()
        self.state = self._load_state()

        self.arm_rows: list[dict[str, Any]] = []
        self.time_entries: list[ttk.Entry] = []
        self.risk_rows: list[dict[str, Any]] = []

        self._build_ui()
        self._bind_shortcuts()

        if start_image:
            for i, path in enumerate(self.queue):
                if path.name == start_image:
                    self.index = i
                    break

        if not self.queue:
            messagebox.showinfo(
                "Nothing to label",
                "No unlabeled images found.\n\n"
                f"Drop PNGs into:\n  {inbox_dir(chart_type)}\n  {images_dir(chart_type)}\n\n"
                "Or run collection first:\n  python real_dataset/run_collection.py",
            )
        else:
            self.load_current()

    def _load_state(self) -> dict[str, Any]:
        if LABELING_STATE_FILE.exists():
            with open(LABELING_STATE_FILE, encoding="utf-8") as handle:
                return json.load(handle)
        return {"last_index": 0, "drafts": {}}

    def _save_state(self) -> None:
        self.state["last_index"] = self.index
        if self.current_image_path:
            self.state.setdefault("drafts", {})[os.path.basename(self.current_image_path)] = self.draft
        with open(LABELING_STATE_FILE, "w", encoding="utf-8") as handle:
            json.dump(self.state, handle, indent=2)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        image_frame = ttk.Frame(self, padding=8)
        image_frame.grid(row=0, column=0, sticky="nsew")
        image_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)

        self.image_label = ttk.Label(image_frame, anchor="center")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(image_frame, textvariable=self.status_var).grid(row=1, column=0, sticky="w", pady=(6, 0))

        form_outer = ttk.Frame(self, padding=8)
        form_outer.grid(row=0, column=1, sticky="nsew")
        form_outer.rowconfigure(0, weight=1)
        form_outer.columnconfigure(0, weight=1)

        canvas = tk.Canvas(form_outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(form_outer, orient="vertical", command=canvas.yview)
        self.form = ttk.Frame(canvas)
        self.form.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.form, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self._build_axes_section()
        self._build_arms_section()
        self._build_risk_table_section()
        self._build_advanced_section()

        toolbar = ttk.Frame(self, padding=8)
        toolbar.grid(row=1, column=0, columnspan=2, sticky="ew")
        for i in range(6):
            toolbar.columnconfigure(i, weight=1)

        ttk.Button(toolbar, text="← Prev", command=self.prev_image).grid(row=0, column=0, sticky="ew", padx=4)
        ttk.Button(toolbar, text="Discard (D)", command=self.discard_current).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(toolbar, text="Skip →", command=self.next_image).grid(row=0, column=2, sticky="ew", padx=4)
        ttk.Button(toolbar, text="Save & Next (S)", command=self.save_and_next).grid(row=0, column=3, sticky="ew", padx=4)
        ttk.Button(toolbar, text="+ Arm", command=self.add_arm_row).grid(row=0, column=4, sticky="ew", padx=4)
        ttk.Button(toolbar, text="+ Risk row", command=self.add_risk_row).grid(row=0, column=5, sticky="ew", padx=4)

    def _build_axes_section(self) -> None:
        section = ttk.LabelFrame(self.form, text="Axes", padding=8)
        section.pack(fill="x", pady=(0, 8))

        self.x_label_var = tk.StringVar(value="Time (months)")
        self.x_max_var = tk.StringVar(value="")
        self.y_label_var = tk.StringVar(value="Survival Probability")
        self.y_max_var = tk.StringVar(value="1.0")

        ttk.Label(section, text="X label").grid(row=0, column=0, sticky="w")
        ttk.Entry(section, textvariable=self.x_label_var, width=28).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Label(section, text="X max").grid(row=1, column=0, sticky="w")
        ttk.Entry(section, textvariable=self.x_max_var, width=12).grid(row=1, column=1, sticky="w", padx=4)

        ttk.Label(section, text="Y label").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(section, textvariable=self.y_label_var, width=28).grid(row=2, column=1, sticky="ew", padx=4, pady=(6, 0))
        ttk.Label(section, text="Y max").grid(row=3, column=0, sticky="w")
        ttk.Entry(section, textvariable=self.y_max_var, width=12).grid(row=3, column=1, sticky="w", padx=4)
        section.columnconfigure(1, weight=1)

    def _build_arms_section(self) -> None:
        self.arms_frame = ttk.LabelFrame(self.form, text="Survival arms (curve metadata)", padding=8)
        self.arms_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(
            self.arms_frame,
            text="Add one row per curve. Coordinates / censoring can stay empty if you only label the risk table.",
            wraplength=420,
        ).pack(anchor="w", pady=(0, 6))

    def _build_risk_table_section(self) -> None:
        self.risk_frame = ttk.LabelFrame(self.form, text="Number at risk table", padding=8)
        self.risk_frame.pack(fill="x", pady=(0, 8))

        controls = ttk.Frame(self.risk_frame)
        controls.pack(fill="x", pady=(0, 6))
        ttk.Button(controls, text="+ time column", command=self.add_time_column).pack(side="left", padx=(0, 4))
        ttk.Button(controls, text="- time column", command=self.remove_time_column).pack(side="left")

        self.time_row = ttk.Frame(self.risk_frame)
        self.time_row.pack(fill="x", pady=(0, 4))
        ttk.Label(self.time_row, text="Time →", width=14).pack(side="left")

        self.risk_rows_container = ttk.Frame(self.risk_frame)
        self.risk_rows_container.pack(fill="x")

        self._rebuild_time_columns(DEFAULT_TIME_POINTS)
        self.add_risk_row()
        self.add_risk_row()

    def _build_advanced_section(self) -> None:
        section = ttk.LabelFrame(self.form, text="Advanced (optional JSON per arm)", padding=8)
        section.pack(fill="both", expand=True)

        ttk.Label(
            section,
            text="Paste coordinates as [[time, prob], ...] and censoring ticks as comma-separated times.",
            wraplength=420,
        ).pack(anchor="w")

        self.advanced_text = tk.Text(section, height=8, wrap="word")
        self.advanced_text.pack(fill="both", expand=True, pady=(6, 0))
        self.advanced_text.insert(
            "1.0",
            '# Example for arm 0:\n# coordinates: [[0, 1.0], [12, 0.8], [24, 0.6]]\n# censoring_ticks: 12, 18, 30\n',
        )

    def _bind_shortcuts(self) -> None:
        self.bind("<d>", lambda _event: self.discard_current())
        self.bind("<D>", lambda _event: self.discard_current())
        self.bind("<s>", lambda _event: self.save_and_next())
        self.bind("<S>", lambda _event: self.save_and_next())
        self.bind("<Control-s>", lambda _event: self.save_and_next())
        self.bind("<Control-Return>", lambda _event: self.save_and_next())
        self.bind("<Right>", lambda _event: self.next_image())
        self.bind("<Left>", lambda _event: self.prev_image())

    def _rebuild_time_columns(self, time_points: list[float | int]) -> None:
        for widget in self.time_row.winfo_children()[1:]:
            widget.destroy()
        self.time_entries.clear()

        for value in time_points:
            entry = ttk.Entry(self.time_row, width=6)
            entry.insert(0, str(value))
            entry.pack(side="left", padx=2)
            self.time_entries.append(entry)

        for row in self.risk_rows:
            self._sync_risk_count_entries(row)

    def add_time_column(self) -> None:
        values = self._read_time_points()
        values.append(values[-1] + 6 if values else 0)
        self._rebuild_time_columns(values)

    def remove_time_column(self) -> None:
        values = self._read_time_points()
        if len(values) <= 1:
            return
        self._rebuild_time_columns(values[:-1])

    def _sync_risk_count_entries(self, row: dict[str, Any]) -> None:
        for widget in row["counts_frame"].winfo_children():
            widget.destroy()
        row["count_entries"] = []
        for _ in self.time_entries:
            entry = ttk.Entry(row["counts_frame"], width=6)
            entry.pack(side="left", padx=2)
            row["count_entries"].append(entry)

    def add_risk_row(self, label: str = "", counts: list[str] | None = None) -> None:
        row_frame = ttk.Frame(self.risk_rows_container)
        row_frame.pack(fill="x", pady=2)

        label_entry = ttk.Entry(row_frame, width=14)
        label_entry.insert(0, label)
        label_entry.pack(side="left")

        counts_frame = ttk.Frame(row_frame)
        counts_frame.pack(side="left", padx=(4, 0))

        remove_btn = ttk.Button(row_frame, text="×", width=3, command=lambda: self._remove_risk_row(row))
        remove_btn.pack(side="right")

        row = {
            "frame": row_frame,
            "label_entry": label_entry,
            "counts_frame": counts_frame,
            "count_entries": [],
            "remove_btn": remove_btn,
        }
        self._sync_risk_count_entries(row)
        if counts:
            for entry, value in zip(row["count_entries"], counts):
                entry.insert(0, value)
        self.risk_rows.append(row)

    def _remove_risk_row(self, row: dict[str, Any]) -> None:
        row["frame"].destroy()
        self.risk_rows.remove(row)

    def add_arm_row(
        self,
        label: str = "",
        coordinates: list[list[float]] | None = None,
        censoring: list[float] | None = None,
    ) -> None:
        frame = ttk.Frame(self.arms_frame)
        frame.pack(fill="x", pady=2)

        label_var = tk.StringVar(value=label)
        ttk.Entry(frame, textvariable=label_var, width=22).pack(side="left")
        coord_var = tk.StringVar(value=json.dumps(coordinates or []))
        ttk.Entry(frame, textvariable=coord_var, width=34).pack(side="left", padx=4)
        censor_var = tk.StringVar(value=", ".join(str(v) for v in (censoring or [])))
        ttk.Entry(frame, textvariable=censor_var, width=16).pack(side="left", padx=4)
        ttk.Button(frame, text="×", width=3, command=lambda: self._remove_arm_row(row)).pack(side="right")

        row = {"frame": frame, "label_var": label_var, "coord_var": coord_var, "censor_var": censor_var}
        self.arm_rows.append(row)

    def _remove_arm_row(self, row: dict[str, Any]) -> None:
        row["frame"].destroy()
        self.arm_rows.remove(row)

    def load_current(self) -> None:
        if not self.queue:
            return

        if self.index >= len(self.queue):
            self.index = len(self.queue) - 1
        if self.index < 0:
            self.index = 0

        image_path = self.queue[self.index]
        self.current_image_path = str(image_path)

        image = Image.open(image_path).convert("RGB")
        max_w, max_h = 820, 760
        image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo)

        basename = image_path.name
        draft = self.state.get("drafts", {}).get(basename)
        if draft:
            self._apply_draft(draft)
        else:
            self._reset_form()

        remaining = len(self.queue) - self.index
        self.status_var.set(
            f"{basename}  —  {self.index + 1}/{len(self.queue)} in queue  ({remaining} remaining)"
        )

    def _reset_form(self) -> None:
        self.x_label_var.set("Time (months)")
        self.x_max_var.set("")
        self.y_label_var.set("Survival Probability")
        self.y_max_var.set("1.0")

        for row in list(self.arm_rows):
            self._remove_arm_row(row)
        for row in list(self.risk_rows):
            self._remove_risk_row(row)

        self._rebuild_time_columns(DEFAULT_TIME_POINTS.copy())
        self.add_risk_row()
        self.add_risk_row()

    def _apply_draft(self, draft: dict[str, Any]) -> None:
        axes = draft.get("axes", {})
        self.x_label_var.set(axes.get("x", {}).get("label", "Time (months)"))
        self.x_max_var.set(str(axes.get("x", {}).get("max_value", "")))
        self.y_label_var.set(axes.get("y", {}).get("label", "Survival Probability"))
        self.y_max_var.set(str(axes.get("y", {}).get("max_value", "1.0")))

        for row in list(self.arm_rows):
            self._remove_arm_row(row)
        for arm in draft.get("arms", []):
            self.add_arm_row(
                label=arm.get("treatment_label", ""),
                coordinates=arm.get("coordinates", []),
                censoring=arm.get("censoring_ticks", []),
            )

        risk = draft.get("at_risk_table", {})
        for row in list(self.risk_rows):
            self._remove_risk_row(row)
        self._rebuild_time_columns(risk.get("time_points", DEFAULT_TIME_POINTS.copy()))
        rows = risk.get("rows", [])
        if not rows:
            self.add_risk_row()
            self.add_risk_row()
        else:
            for risk_row in rows:
                counts = [str(value) for value in risk_row.get("counts", [])]
                self.add_risk_row(label=risk_row.get("treatment_label", ""), counts=counts)

    def _read_time_points(self) -> list[float]:
        values: list[float] = []
        for entry in self.time_entries:
            text = entry.get().strip()
            if not text:
                continue
            values.append(float(text))
        return values

    def _parse_float(self, text: str, field_name: str) -> float:
        try:
            return float(text.strip())
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a number, got {text!r}") from exc

    def _collect_label(self) -> dict[str, Any]:
        x_max_text = self.x_max_var.get().strip()
        if not x_max_text:
            time_points = self._read_time_points()
            x_max_text = str(max(time_points) if time_points else 0.0)

        label: dict[str, Any] = {
            "chart_type": "kaplan_meier",
            "axes": {
                "x": {
                    "label": self.x_label_var.get().strip() or "Time (months)",
                    "max_value": self._parse_float(x_max_text, "X max"),
                },
                "y": {
                    "label": self.y_label_var.get().strip() or "Survival Probability",
                    "max_value": self._parse_float(self.y_max_var.get() or "1.0", "Y max"),
                },
            },
            "arms": [],
            "at_risk_table": {"time_points": self._read_time_points(), "rows": []},
        }

        for row in self.arm_rows:
            treatment = row["label_var"].get().strip()
            if not treatment:
                continue
            coords_raw = row["coord_var"].get().strip() or "[]"
            censor_raw = row["censor_var"].get().strip()
            coordinates = json.loads(coords_raw)
            censoring = [float(part.strip()) for part in censor_raw.split(",") if part.strip()]
            label["arms"].append(
                {
                    "treatment_label": treatment,
                    "coordinates": coordinates,
                    "censoring_ticks": censoring,
                }
            )

        for row in self.risk_rows:
            treatment = row["label_entry"].get().strip()
            counts: list[int] = []
            for entry in row["count_entries"]:
                text = entry.get().strip()
                counts.append(int(text) if text else 0)
            if not treatment and not any(counts):
                continue
            label["at_risk_table"]["rows"].append(
                {"treatment_label": treatment, "counts": counts}
            )

        if not label["arms"] and label["at_risk_table"]["rows"]:
            for risk_row in label["at_risk_table"]["rows"]:
                label["arms"].append(
                    {
                        "treatment_label": risk_row["treatment_label"],
                        "coordinates": [],
                        "censoring_ticks": [],
                    }
                )

        return label

    def save_and_next(self) -> None:
        if not self.current_image_path:
            return
        try:
            label = self._collect_label()
        except (ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("Invalid label", str(exc))
            return

        if not label["at_risk_table"]["rows"] and not label["arms"]:
            messagebox.showerror("Missing data", "Add at least one arm or one at-risk table row before saving.")
            return

        image_path = self.current_image_path
        basename = os.path.basename(image_path)
        accepted_dir = images_dir(self.chart_type)
        accepted_dir.mkdir(parents=True, exist_ok=True)

        from_inbox = str(inbox_dir(self.chart_type)) in os.path.abspath(image_path)
        if from_inbox or basename.startswith("raw_"):
            curated_name = next_curated_name(self.chart_type)
        else:
            curated_name = basename

        label_path = labels_dir(self.chart_type) / f"{Path(curated_name).stem}.json"
        label_path.parent.mkdir(parents=True, exist_ok=True)

        with open(label_path, "w", encoding="utf-8") as handle:
            json.dump(label, handle, indent=2)

        target = accepted_dir / curated_name
        if os.path.abspath(image_path) != os.path.abspath(target):
            shutil.move(image_path, target)

        self.state.setdefault("drafts", {}).pop(basename, None)
        self._save_state()

        self.queue.pop(self.index)
        if self.index >= len(self.queue) and self.queue:
            self.index = len(self.queue) - 1

        if self.queue:
            self.load_current()
        else:
            messagebox.showinfo("Done", "All images in the queue are labeled.")
            self.status_var.set("Queue complete")

    def discard_current(self) -> None:
        if not self.current_image_path:
            return
        if not messagebox.askyesno("Discard image", "Move this image to discarded/ ?"):
            return

        basename = os.path.basename(self.current_image_path)
        target_dir = discarded_dir(self.chart_type)
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(self.current_image_path, target_dir / basename)

        self.state.setdefault("drafts", {}).pop(basename, None)
        self._save_state()

        self.queue.pop(self.index)
        if self.index >= len(self.queue) and self.queue:
            self.index = len(self.queue) - 1

        if self.queue:
            self.load_current()
        else:
            self.status_var.set("Queue complete")

    def next_image(self) -> None:
        if not self.queue:
            return
        self.draft = self._collect_label() if self.current_image_path else default_km_label()
        if self.current_image_path:
            self.state.setdefault("drafts", {})[os.path.basename(self.current_image_path)] = self.draft
            self._save_state()
        self.index = min(self.index + 1, len(self.queue) - 1)
        self.load_current()

    def prev_image(self) -> None:
        if not self.queue:
            return
        self.index = max(self.index - 1, 0)
        self.load_current()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive KM chart labeler")
    parser.add_argument("--type", default="km", choices=["km", "forest", "wf"], help="Chart type folder")
    parser.add_argument("--image", default=None, help="Start at a specific filename")
    args = parser.parse_args()

    if args.type != "km":
        messagebox.showerror("Unsupported", "Forest/waterfall labelers are not implemented yet. Use --type km.")
        return

    app = KMChartLabeler(chart_type=args.type, start_image=args.image)
    app.mainloop()


if __name__ == "__main__":
    main()
