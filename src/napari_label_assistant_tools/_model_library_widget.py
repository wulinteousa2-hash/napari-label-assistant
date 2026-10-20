"""Workspace-wide model catalog, training history, and selected-run analysis."""
from pathlib import Path
from contextlib import suppress
import json

from qtpy.QtCore import Qt, QByteArray, QRectF, QItemSelectionModel
from qtpy.QtGui import QPainter, QImage
from qtpy.QtSvg import QSvgRenderer
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QPushButton, QLineEdit, QPlainTextEdit, QCheckBox, QComboBox, QDoubleSpinBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QAbstractItemView, QFileDialog,
    QGroupBox, QScrollArea, QLayout)

from ._model_project import project, model_project_events, RUNTIME
from ._model_library import (catalog, comparison_key, comparison_note, annotate,
    export_csv, run_curve, chart_svg, weight_statistics, number, FIELDS)


class _ValueItem(QTableWidgetItem):
    def __lt__(self, other):
        a, b = self.data(Qt.UserRole), other.data(Qt.UserRole)
        if a is None and isinstance(b, (int, float)):
            return True
        if b is None and isinstance(a, (int, float)):
            return False
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a < b
        return self.text().casefold() < other.text().casefold()


class _Chart(QWidget):
    def __init__(self):
        super().__init__()
        self.svg = chart_svg([])
        self.setMinimumHeight(230)

    def paintEvent(self, event):
        painter = QPainter(self)
        renderer = QSvgRenderer(QByteArray(self.svg.encode()))
        renderer.render(painter, QRectF(self.rect()))
        painter.end()


class ModelLibraryWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        viewer._label_assistant_model_library = self
        self.rows = []
        self.comparison_ids = []
        self._editing_key = None
        self.weight_worker = None
        self.closed = False
        layout = QVBoxLayout(self)
        note = QLabel("Browse all models and training runs in this workspace. Save the workspace to keep names, notes, tags, and checkpoints.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.current_label = QLabel()
        self.current_label.setWordWrap(True)
        layout.addWidget(self.current_label)
        filters = QFormLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search model, architecture, dataset, tags, notes…")
        self.task_filter = QComboBox()
        self.dataset_filter = QComboBox()
        self.max_parameters = QDoubleSpinBox()
        self.max_parameters.setRange(0, 100000)
        self.max_parameters.setDecimals(3)
        self.max_parameters.setSuffix(" M")
        self.max_parameters.setSpecialValueText("Any size")
        self.max_parameters.setToolTip("Maximum parameter count in millions. Zero includes all sizes; unknown sizes are excluded when a limit is set.")
        self.favorites = QCheckBox("Favorites only")
        self.comparable = QCheckBox("Match selected dataset and split")
        self.comparable.setToolTip("Uses recorded dataset snapshot and validation settings. Unknown provenance is excluded.")
        filters.addRow("Search", self.search)
        filters.addRow("Task", self.task_filter)
        filters.addRow("Dataset", self.dataset_filter)
        filters.addRow("Maximum parameters", self.max_parameters)
        filters.addRow(self.favorites)
        filters.addRow(self.comparable)
        layout.addLayout(filters)
        more_filters = QCheckBox("More filters")
        layout.addWidget(more_filters)
        def show_filters(visible):
            for control in (self.max_parameters, self.favorites, self.comparable):
                control.setVisible(visible)
                label = filters.labelForField(control)
                if label is not None:
                    label.setVisible(visible)
        more_filters.toggled.connect(show_filters)
        show_filters(False)
        self._comparison_anchor = None
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        models_page = QWidget()
        model_layout = QVBoxLayout(models_page)
        self.model_columns = ("model_name", "dice", "parameters", "dataset", "task", "dataset_id", "provider", "architecture", "run", "favorite", "status")
        self.models = self._table(self.model_columns, multiple=False)
        self.models.sortItems(0, Qt.AscendingOrder)
        model_layout.addWidget(self.models, 1)
        editor = QGroupBox("Model and selected-run documentation")
        form = QFormLayout(editor)
        self.name = QLineEdit()
        self.dataset_name = QLineEdit()
        self.tags = QLineEdit()
        self.tags.setPlaceholderText("Comma-separated: lightweight, candidate, production…")
        self.favorite = QCheckBox("Favorite model")
        self.notes = QPlainTextEdit()
        self.notes.setMaximumHeight(75)
        self.notes.setPlaceholderText("What changed, observations, intended use, limitations…")
        self.save_notes = QPushButton("Save documentation")
        for label, widget in (("Model name", self.name), ("Dataset name", self.dataset_name), ("Tags", self.tags), ("Run notes", self.notes)):
            form.addRow(label, widget)
        form.addRow(self.favorite)
        form.addRow(self.save_notes)
        self.documentation_toggle = QCheckBox("Edit selected model / run documentation")
        layout.addWidget(self.documentation_toggle)
        layout.addWidget(editor)
        editor.hide()
        self.documentation_toggle.toggled.connect(editor.setVisible)
        self.tabs.addTab(models_page, "Models")

        history_page = QWidget()
        history_layout = QVBoxLayout(history_page)
        self.history = self._table(FIELDS, multiple=True)
        self.history.sortItems(FIELDS.index("date"), Qt.DescendingOrder)
        for position, field in enumerate(("model_name", "dice", "iou", "parameters", "run", "date", "dataset", "dataset_id")):
            self.history.horizontalHeader().moveSection(self.history.horizontalHeader().visualIndex(FIELDS.index(field)), position)
        history_layout.addWidget(self.history, 1)
        self.all_columns = QCheckBox("Show all history columns")
        history_layout.addWidget(self.all_columns)
        self.compare_button = QPushButton("Compare selected runs →")
        history_layout.addWidget(self.compare_button)
        self.tabs.addTab(history_page, "Training History")

        compare_page = QWidget()
        compare_layout = QVBoxLayout(compare_page)
        compare_layout.setSizeConstraint(QLayout.SetMinimumSize)
        compare_page.setMinimumHeight(570)
        self.comparison_note = QLabel("Select runs in Training History, then choose Compare selected runs.")
        self.comparison_note.setWordWrap(True)
        compare_layout.addWidget(self.comparison_note)
        chart_form = QFormLayout()
        self.chart_kind = QComboBox()
        for label, key in (("Validation metric", "metrics"), ("Score versus parameter count", "size"),
                           ("Results over time", "time"), ("Learning curves", "curves")):
            self.chart_kind.addItem(label, key)
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["dice", "iou", "precision", "recall", "loss"])
        chart_form.addRow("Chart", self.chart_kind)
        chart_form.addRow("Metric", self.metric_combo)
        compare_layout.addLayout(chart_form)
        self.chart = _Chart()
        compare_layout.addWidget(self.chart, 1)
        self.legend = QPlainTextEdit()
        self.legend.setReadOnly(True)
        self.legend.setMaximumHeight(95)
        compare_layout.addWidget(self.legend)
        self.export_chart = QPushButton("Export chart…")
        compare_layout.addWidget(self.export_chart)
        compare_scroll = QScrollArea()
        compare_scroll.setWidgetResizable(True)
        compare_scroll.setWidget(compare_page)
        self.tabs.addTab(compare_scroll, "Compare Runs")

        actions = QHBoxLayout()
        analysis_actions = QHBoxLayout()
        self.select_button = QPushButton("Select checkpoint")
        self.inspect_button = QPushButton("Inspect loaded weights…")
        self.export_button = QPushButton("Export history CSV…")
        self.refresh_button = QPushButton("Refresh")
        self.import_button = QPushButton("Import checkpoint…")
        for button in (self.select_button, self.import_button, self.refresh_button):
            actions.addWidget(button)
        for button in (self.inspect_button, self.export_button):
            analysis_actions.addWidget(button)
        layout.addLayout(actions)
        layout.addLayout(analysis_actions)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(130)
        show_details = QCheckBox("Show selected-run details")
        layout.addWidget(show_details)
        layout.addWidget(self.details)
        self.details.hide()
        show_details.toggled.connect(self.details.setVisible)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.search.textChanged.connect(self.render_tables)
        self.task_filter.currentIndexChanged.connect(self.render_tables)
        self.dataset_filter.currentIndexChanged.connect(self.render_tables)
        self.max_parameters.valueChanged.connect(self.render_tables)
        self.favorites.toggled.connect(self.render_tables)
        self.comparable.toggled.connect(self._toggle_comparable)
        self.models.itemSelectionChanged.connect(self.selection_changed)
        self.history.itemSelectionChanged.connect(self.selection_changed)
        self.tabs.currentChanged.connect(self.selection_changed)
        self.all_columns.toggled.connect(self._columns)
        self.save_notes.clicked.connect(lambda: self.run(self._save_documentation))
        self.compare_button.clicked.connect(self.compare_selected)
        self.chart_kind.currentIndexChanged.connect(lambda: self.run(self.draw_chart))
        self.metric_combo.currentIndexChanged.connect(lambda: self.run(self.draw_chart))
        self.export_chart.clicked.connect(lambda: self.run(self._export_chart))
        self.export_button.clicked.connect(lambda: self.run(self._export_csv))
        self.select_button.clicked.connect(lambda: self.run(self._select_checkpoint))
        self.import_button.clicked.connect(lambda: self.run(lambda: self._trainer()._load()))
        self.inspect_button.clicked.connect(lambda: self.run(self._inspect_weights))
        self.refresh_button.clicked.connect(self.refresh)
        self.events = model_project_events(viewer)
        self.events.changed.connect(self.refresh)
        self.destroyed.connect(self.shutdown)
        self.refresh()
        self._columns(False)

    def _table(self, columns, multiple):
        table = QTableWidget(0, len(columns))
        table.setHorizontalHeaderLabels(["Dataset revision" if c == "dataset_id" else c.replace("_", " ").title() for c in columns])
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.ExtendedSelection if multiple else QAbstractItemView.SingleSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSortingEnabled(True)
        table.setMinimumHeight(150)
        for index, field in enumerate(columns):
            table.setColumnWidth(index, 170 if field == "model_name" else 75 if field in ("dice", "iou", "precision", "recall", "loss") else 110)
        return table

    def run(self, action):
        try:
            action()
        except Exception as error:
            self.status.setText(str(error))

    def refresh(self, *_args):
        if self.closed:
            return
        self.rows = catalog(self.viewer)
        state = project(self.viewer)
        current = next((r for r in self.rows if r["model_id"] == state.get("current_model") and r["revision"] == state.get("current_checkpoint")), None)
        self.current_label.setText(f"Current checkpoint: {current['model_name']} / run {current['run']}" if current else "No current checkpoint selected for training or inference.")
        for combo, entries in ((self.task_filter, [(t["name"], t["id"]) for t in state["tasks"]]),
                               (self.dataset_filter, sorted({(r["dataset"], r["dataset_id"]) for r in self.rows if r["dataset_id"]}))):
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("All", None)
            for label, key in entries:
                combo.addItem(f"{label} [{key[:10]}]" if combo is self.dataset_filter else label, key)
            combo.setCurrentIndex(max(0, combo.findData(current)))
            combo.blockSignals(False)
        self.render_tables()
        self.run(self.draw_chart)

    def filtered_rows(self):
        query = self.search.text().strip().casefold()
        limit = self.max_parameters.value() * 1_000_000
        rows = []
        for row in self.rows:
            if self.task_filter.currentData() and row["task_id"] != self.task_filter.currentData():
                continue
            if self.dataset_filter.currentData() and row["dataset_id"] != self.dataset_filter.currentData():
                continue
            if self.favorites.isChecked() and not row["favorite"]:
                continue
            if limit and (not number(row["parameters"]) or row["parameters"] > limit):
                continue
            if self.comparable.isChecked() and (self._comparison_anchor is None or comparison_key(row) != self._comparison_anchor):
                continue
            if query and query not in " ".join(str(row.get(k, "")) for k in FIELDS).casefold():
                continue
            rows.append(row)
        return rows

    def selected_rows(self, table=None):
        table = table or (self.history if self.tabs.currentIndex() in (1, 2) else self.models)
        identifiers = {table.item(index.row(), 0).data(Qt.UserRole + 1) for index in table.selectionModel().selectedRows()}
        return [row for row in self.rows if row["run_id"] in identifiers]

    def _fill(self, table, columns, rows):
        selected = {r["run_id"] for r in self.selected_rows(table)}
        table.blockSignals(True)
        table.setSortingEnabled(False)
        table.clearSelection()
        table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            for col, key in enumerate(columns):
                value = row.get(key)
                text = "—" if value is None or value == "" else f"{value:.5g}" if isinstance(value, float) else str(value)
                if key == "dataset_id" and value:
                    text = str(value)[:12]
                item = _ValueItem(text)
                item.setData(Qt.UserRole, value)
                item.setToolTip(str(value) if value is not None else "Not recorded")
                if col == 0:
                    item.setData(Qt.UserRole + 1, row["run_id"])
                table.setItem(index, col, item)
        table.setSortingEnabled(True)
        for index in range(table.rowCount()):
            if table.item(index, 0).data(Qt.UserRole + 1) in selected:
                table.selectionModel().select(table.model().index(index, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)
        table.blockSignals(False)
        if not selected and table.rowCount():
            table.selectRow(0)

    def render_tables(self, *_args):
        rows = self.filtered_rows()
        latest = {}
        for row in rows:
            if row["model_id"] not in latest or latest[row["model_id"]]["run"] < row["run"]:
                latest[row["model_id"]] = row
        self._fill(self.models, self.model_columns, list(latest.values()))
        self._fill(self.history, FIELDS, rows)
        self.status.setText(f"{len(latest)} models · {len(rows)} recorded runs. Scores are recorded validation results; parameter count is not a speed benchmark.")
        self.selection_changed()
        if self.tabs.currentIndex() == 2:
            self.run(self.draw_chart)

    def _columns(self, all_columns):
        default = {"model_name", "run", "date", "task", "dataset", "dataset_id", "provider", "width", "parameters", "dice", "iou", "validation", "epochs", "seconds", "status"}
        for index, key in enumerate(FIELDS):
            self.history.setColumnHidden(index, not all_columns and key not in default)

    def selection_changed(self, *_args):
        selected = self.selected_rows()
        row = selected[0] if selected else None
        for button in (self.save_notes, self.select_button, self.inspect_button):
            button.setEnabled(row is not None)
        if row is None:
            self.details.setPlainText("No model selected. Train a model or import a checkpoint in Model Training to populate this workspace library.")
            self._editing_key = None
            return
        model = next(m for m in project(self.viewer)["models"] if m["id"] == row["model_id"])
        self.details.setPlainText(json.dumps({"model": row["model_name"], "run": row["run"], "dataset_id": row["dataset_id"] or "Not recorded", "checkpoint": row["checkpoint"] or "Not saved", "status": row["status"], "history": row["history"]}, indent=2))
        if self._editing_key != row["run_id"]:
            self._editing_key = row["run_id"]
            self.name.setText(row["model_name"])
            self.dataset_name.setText(row["dataset"])
            self.dataset_name.setEnabled(bool(row["dataset_id"]))
            self.notes.setPlainText(row["notes"])
            self.tags.setText(row["tags"])
            self.favorite.setChecked(model.get("favorite", False))

    def _toggle_comparable(self, enabled):
        selected = self.selected_rows()
        self._comparison_anchor = comparison_key(selected[0]) if selected else None
        if enabled and self._comparison_anchor is None:
            self.comparable.setChecked(False)
            self.status.setText("Select a run with a recorded dataset snapshot and validation settings first.")
            return
        self.render_tables()

    def _save_documentation(self):
        selected = self.selected_rows()
        if not selected:
            raise ValueError("Select a model or training run first.")
        annotate(self.viewer, selected[0], name=self.name.text(), dataset_name=self.dataset_name.text(),
                 notes=self.notes.toPlainText(), tags=self.tags.text(), favorite=self.favorite.isChecked())
        trainer = getattr(self.viewer, "_label_assistant_training_widget", None)
        if trainer is not None and not trainer.closed:
            trainer.models_panel.refresh()
        self.status.setText("Documentation updated. Save the workspace to persist these changes.")

    def compare_selected(self):
        selected = self.selected_rows(self.history)
        if len(selected) < 2:
            self.status.setText("Select at least two rows in Training History (Ctrl/Shift-click).")
            return
        self.comparison_ids = [r["run_id"] for r in selected]
        self.tabs.setCurrentIndex(2)
        self.run(self.draw_chart)

    def draw_chart(self):
        rows = self.comparison_rows()
        metric, kind = self.metric_combo.currentText(), self.chart_kind.currentData()
        self.comparison_note.setText(comparison_note(rows, metric))
        missing = []
        if kind == "curves":
            from ._training_report import learning_svg
            points = []
            for row in rows:
                curve = run_curve(self.viewer, row)
                if not curve:
                    missing.append(row["model_name"])
                points.extend({**p, "metrics": {metric: p["metrics"][metric]}} for p in curve if number(p["metrics"].get(metric)))
            colors = ("#147d64", "#2468ac", "#b65b18", "#884eb5", "#ad3f69")
            series_colors = {p["series"]: colors[i % len(colors)] for i, p in enumerate({p["series"]: p for p in points}.values())}
            self.chart.svg = learning_svg(points, title=f"Selected-run validation {metric}", loss=metric == "loss", series_colors=series_colors)
        else:
            self.chart.svg = chart_svg(rows, kind=kind, metric=metric)
        legend_rows = rows if kind == "curves" else [r for r in rows if number(r.get(metric)) and (kind != "size" or number(r.get("parameters")))]
        if kind == "time":
            legend_rows = sorted(legend_rows, key=lambda r: r.get("date", ""))
        self.legend.setPlainText("\n".join(f"{i}. {r['model_name']} / run {r['run']} · {r['date'] or 'date unknown'} · {r['dataset']}" for i, r in enumerate(legend_rows, 1)) +
                                 ("\nCurves unavailable (history sidecar or in-memory checkpoint needed): " + ", ".join(missing) if missing else ""))
        self.chart.update()

    def comparison_rows(self):
        return [r for r in self.filtered_rows() if r["run_id"] in self.comparison_ids]

    def _export_csv(self):
        rows = self.comparison_rows() if self.tabs.currentIndex() == 2 else self.filtered_rows()
        path, _ = QFileDialog.getSaveFileName(self, "Export model training history", "model-training-history.csv", "CSV (*.csv)")
        if path:
            export_csv(rows, path)
            self.status.setText(f"Exported {len(rows)} runs to {path}.")

    def _export_chart(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export comparison chart", "model-comparison.svg", "SVG (*.svg);;PNG (*.png)")
        if not path:
            return
        self.draw_chart()
        if Path(path).suffix.lower() == ".png":
            renderer = QSvgRenderer(QByteArray(self.chart.svg.encode()))
            image = QImage(renderer.defaultSize(), QImage.Format_ARGB32)
            image.fill(Qt.white)
            painter = QPainter(image)
            renderer.render(painter)
            painter.end()
            if not image.save(path):
                raise ValueError("Could not write the PNG chart.")
        else:
            Path(path).write_text(self.chart.svg, encoding="utf-8")
        self.status.setText(f"Chart exported to {path}.")

    def _select_checkpoint(self):
        rows = self.selected_rows()
        if len(rows) != 1:
            raise ValueError("Select one checkpoint to use for training or inference.")
        trainer = self._trainer()
        if trainer.worker is not None:
            raise ValueError("Wait for the current training operation to finish.")
        row = rows[0]
        panel = trainer.models_panel
        panel.refresh()
        panel.task_combo.setCurrentIndex(max(0, panel.task_combo.findData(row["task_id"])))
        panel.models.setCurrentIndex(panel.models.findData(row["model_id"]))
        panel.run_combo.setCurrentIndex(max(0, panel.run_combo.findData(row["revision"])))
        panel.use_selected()
        self.status.setText("Checkpoint selection requested. Model Training reports loading progress and readiness.")

    def _trainer(self):
        trainer = getattr(self.viewer, "_label_assistant_training_widget", None)
        if trainer is None or trainer.closed:
            from ._quick_train_widget import quick_train_widget
            trainer = quick_train_widget(self.viewer)
            self.viewer.window.add_dock_widget(trainer, name="Model Training")
        return trainer

    def _inspect_weights(self):
        if self.weight_worker is not None:
            raise ValueError("Weight inspection is already running.")
        rows = self.selected_rows()
        if len(rows) != 1:
            raise ValueError("Select one loaded checkpoint to inspect.")
        trainer = getattr(self.viewer, "_label_assistant_training_widget", None)
        if trainer is not None and trainer.worker is not None:
            raise ValueError("Wait for training to finish before inspecting weights.")
        row = rows[0]
        model = getattr(self.viewer, RUNTIME, {}).get(row["model_id"] + ":" + row["revision"])
        if model is None:
            raise ValueError("Select this checkpoint first to load its weights. Browsing the library does not load networks.")
        from superqt.utils import create_worker
        worker = create_worker(weight_statistics, model, _start_thread=False,
                               _connect={"errored": lambda error: self.status.setText(str(error)) if not self.closed else None})
        self.weight_worker = worker
        worker.returned.connect(self._show_weights)
        worker.finished.connect(lambda: setattr(self, "weight_worker", None))
        worker.start()
        self.status.setText("Inspecting bounded samples of loaded weights…")

    def _show_weights(self, rows):
        if self.closed:
            return
        self.weight_window = QWidget(self, Qt.Window)
        self.weight_window.setWindowTitle("Loaded model weights")
        layout = QVBoxLayout(self.weight_window)
        note = QLabel("Statistics sample at most 65,536 values per tensor. Nonfinite counts refer to that sample. Weight statistics do not measure prediction quality.")
        note.setWordWrap(True)
        layout.addWidget(note)
        columns = ("layer", "shape", "parameters", "sampled", "nonfinite_sampled", "mean", "std", "min", "max")
        table = self._table(columns, multiple=False)
        self._fill(table, columns, [{**r, "run_id": str(i)} for i, r in enumerate(rows)])
        layout.addWidget(table)
        export = QPushButton("Export weight statistics CSV…")
        def save():
            import csv
            path, _ = QFileDialog.getSaveFileName(self.weight_window, "Export weight statistics", "weight-statistics.csv", "CSV (*.csv)")
            if path:
                with Path(path).open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(rows)
        export.clicked.connect(lambda: self.run(save))
        layout.addWidget(export)
        self.weight_window.resize(850, 500)
        self.weight_window.show()

    def shutdown(self, *_args):
        self.closed = True
        if getattr(self.viewer, "_label_assistant_model_library", None) is self:
            self.viewer._label_assistant_model_library = None
        with suppress(Exception):
            self.events.changed.disconnect(self.refresh)


def model_library_widget(viewer=None, **kwargs):
    if viewer is None:
        import napari
        viewer = napari.current_viewer()
    if viewer is None:
        raise RuntimeError("No active napari viewer found.")
    return ModelLibraryWidget(viewer)
