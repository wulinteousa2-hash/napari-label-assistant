from __future__ import annotations

from pathlib import Path

from napari import current_viewer
from napari.viewer import Viewer
from qtpy.QtCore import QSettings, Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QMenu,
    QListWidgetItem,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QVBoxLayout,
    QWidget,
)

from .workspace import load_workspace, save_workspace

SETTINGS_ORG = "napari-label-assistant"
SETTINGS_APP = "label-assistant"
RECENT_WORKSPACES_KEY = "workspace/recent_manifests"
LAST_WORKSPACE_KEY = "workspace/last_manifest"
COMPLETE_PACKAGE_KEY = "workspace/complete_package"
OPTIMIZE_LARGE_IMAGES_KEY = "workspace/optimize_large_images"
RECENT_LIMIT = 10
VIEWER_WORKSPACE_PATH_ATTR = "_label_assistant_workspace_path"
VIEWER_WORKSPACE_LAYER_IDS_ATTR = "_label_assistant_workspace_layer_ids"


class WorkspaceManagerWidget(QWidget):
    """Manage Label Assistant sessions with lazy, writable mask data."""

    def __init__(
        self,
        napari_viewer: Viewer | None = None,
        viewer: Viewer | None = None,
        *,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__()
        self.viewer = napari_viewer or viewer or current_viewer()
        self.settings = settings or QSettings(SETTINGS_ORG, SETTINGS_APP)
        last = str(
            self.settings.value(LAST_WORKSPACE_KEY, "", type=str) or ""
        ).strip()
        self._last_workspace_path = Path(last).expanduser() if last else None
        active_path = str(
            getattr(self.viewer, VIEWER_WORKSPACE_PATH_ATTR, "") or ""
        ).strip()
        active_layer_ids = set(
            getattr(self.viewer, VIEWER_WORKSPACE_LAYER_IDS_ATTR, ()) or ()
        )
        current_layer_ids = {
            id(layer) for layer in list(getattr(self.viewer, "layers", ()))
        }
        if active_path and active_layer_ids.intersection(current_layer_ids):
            self.workspace_path: Path | None = Path(active_path).expanduser()
            self._project_layer_ids = active_layer_ids
        else:
            # The most recently used path is only a file-dialog convenience.
            # It must never make unrelated viewer contents eligible for Save.
            self.workspace_path = None
            self._project_layer_ids: set[int] = set()

        self.storage_state_label = QLabel()
        self.storage_state_label.setWordWrap(True)
        self.current_label = QLabel()
        self.current_label.setWordWrap(True)
        self.status_label = QLabel(
            "Choose save options, then save your workspace."
        )
        self.status_label.setWordWrap(True)
        self.optimize_images_check = QCheckBox(
            "Optimize large images"
        )
        self.optimize_images_check.setChecked(
            bool(
                self.settings.value(
                    OPTIMIZE_LARGE_IMAGES_KEY, False, type=bool
                )
            )
        )
        self.optimize_images_check.setToolTip(
            "Create multiscale copies of large images for smoother zooming and navigation. "
            "Applies to single-scale 2D images larger than 32,768 pixels on either axis. "
            "Existing pyramids are preserved. Reopen the workspace to use new optimized copies. "
            "This option can be combined with Include all data (portable)."
        )
        self.optimize_images_note = QLabel(
            "Improve zooming and navigation for large images."
        )
        self.optimize_images_note.setWordWrap(True)
        self.optimize_images_check.toggled.connect(
            lambda checked: self.settings.setValue(
                OPTIMIZE_LARGE_IMAGES_KEY, bool(checked)
            )
        )
        self.storage_mode_combo = QComboBox()
        self.storage_mode_combo.addItem("Link to source data", False)
        self.storage_mode_combo.addItem("Include all data (portable)", True)
        self.storage_mode_combo.setCurrentIndex(
            1 if self.settings.value(COMPLETE_PACKAGE_KEY, False, type=bool) else 0
        )
        self.storage_mode_combo.setToolTip(
            "Link to source data keeps saves lightweight and requires access to linked files. "
            "Include all data copies loaded images and masks beside the workspace file. "
            "Keep that file and its data folder together when sharing or archiving. "
            "Large datasets take additional time and disk space to copy. "
            "This choice applies to the next save."
        )
        self.storage_mode_combo.currentIndexChanged.connect(self._update_storage_mode)
        self.package_note = QLabel()
        self.package_note.setWordWrap(True)
        self._update_storage_mode()
        self.recent_list = QListWidget()
        self.recent_list.setMinimumHeight(120)
        self.recent_list.itemDoubleClicked.connect(
            lambda _item: self.open_selected_recent()
        )

        new_button = QPushButton("New")
        open_button = QPushButton("Open")
        save_button = QPushButton("Save")
        save_as_button = QPushButton("Save As")
        recent_button = QPushButton("Open Recent")
        self.open_recent_button = recent_button
        self.recent_menu = QMenu(recent_button)
        recent_button.setMenu(self.recent_menu)

        new_button.setToolTip(
            "Start a new workspace by removing the current layers from the viewer. "
            "Existing files on disk are preserved."
        )
        open_button.setToolTip(
            "Restore layers, annotations, and viewing settings from a saved workspace. "
            "Linked source data must be accessible unless the workspace includes copies."
        )
        save_button.setToolTip(
            "Save the current layers, annotations, and viewing settings using the options below. "
            "An unsaved workspace will prompt for a destination."
        )
        save_as_button.setToolTip(
            "Choose a new name or location for the workspace. Saving to a new destination "
            "creates independent mask copies. Choose Include all data (portable) "
            "to include image data as well."
        )
        recent_button.setToolTip(
            "Choose a recently used workspace from the menu. "
            "You can also double-click a recent workspace to open it."
        )

        new_button.clicked.connect(self.new_workspace)
        open_button.clicked.connect(self.open_workspace)
        save_button.clicked.connect(self.save)
        save_as_button.clicked.connect(self.save_as)

        first_row = QHBoxLayout()
        first_row.addWidget(new_button)
        first_row.addWidget(open_button)
        first_row.addWidget(recent_button)
        first_row.addWidget(save_button)
        first_row.addWidget(save_as_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(QLabel("Workspace files"))
        layout.addWidget(self.current_label)
        layout.addWidget(self.storage_state_label)
        layout.addLayout(first_row)
        layout.addWidget(QLabel("Data storage for next save"))
        layout.addWidget(self.storage_mode_combo)
        layout.addWidget(self.package_note)
        layout.addWidget(self.optimize_images_check)
        layout.addWidget(self.optimize_images_note)
        layout.addWidget(QLabel("Recent workspaces"))
        layout.addWidget(self.recent_list)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        removed = getattr(
            getattr(getattr(self.viewer, "layers", None), "events", None),
            "removed",
            None,
        )
        if removed is not None:
            removed.connect(self._on_layer_removed)

        self._refresh_current_label()
        self._refresh_recent_list()

    def new_workspace(self) -> None:
        if self.viewer is None:
            self._set_status("No napari viewer is available.")
            return
        if self._has_pending_local_edits():
            self._set_status(
                "Apply or save the pending local-area edits before starting "
                "a new project."
            )
            return
        if len(self.viewer.layers) and not self._confirm_clear(
            "Start a new Label Assistant project? Current layers will be "
            "removed from the viewer. "
            "Writable Zarr data already saved on disk will not be deleted."
        ):
            return
        self.viewer.layers.clear()
        self._disassociate_workspace()
        self._refresh_current_label()
        self._set_status("Started a new empty Label Assistant project.")

    def open_workspace(self) -> None:
        reference = self.workspace_path or self._last_workspace_path
        initial = str(reference.parent) if reference else ""
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Open Label Assistant Project",
            initial,
            "Label Assistant project (*.label-assistant.json *.sam3.json *.json)",
        )
        if chosen:
            self._load_path(Path(chosen))

    def open_selected_recent(self) -> None:
        item = self.recent_list.currentItem()
        if item is None:
            self._set_status("Select a recent workspace first.")
            return
        self._load_path(Path(item.data(Qt.UserRole)))

    def save(self) -> None:
        if self.viewer is None:
            self._set_status("No napari viewer is available.")
            return
        if self.workspace_path is None:
            self.save_as()
            return
        if self._project_layer_ids and not self._has_project_layer():
            self._disassociate_workspace()
            self._refresh_current_label()
            self._set_status(
                "The original project layers are no longer open. Choose a "
                "new project name to avoid overwriting the previous project."
            )
            self.save_as()
            return
        self._save_path(self.workspace_path)

    def save_as(self) -> None:
        if self.viewer is None:
            self._set_status("No napari viewer is available.")
            return
        initial = str(
            self.workspace_path
            or self._last_workspace_path
            or Path.cwd() / "project.label-assistant.json"
        )
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Save Label Assistant Project As",
            initial,
            "Label Assistant project (*.label-assistant.json)",
        )
        if not chosen:
            return
        path = Path(chosen)
        if not str(path).lower().endswith(".json"):
            path = path.with_suffix(".label-assistant.json")
        copy_labels = (
            self.workspace_path is None
            or path.resolve() != self.workspace_path.resolve()
        )
        self._save_path(path, copy_labels=copy_labels)

    def _save_path(self, path: Path, *, copy_labels: bool = False) -> None:
        if getattr(self.viewer, "_label_assistant_missing_workspace_layers", ()):
            message = "This workspace has unavailable layers. Restore access to their data and reopen it before saving. The original workspace is protected."
            self._set_status(message)
            QMessageBox.warning(self, "Incomplete workspace", message)
            return
        optimize = self.optimize_images_check.isChecked()
        dialog = QProgressDialog(self)
        dialog.setWindowTitle("Saving Workspace")
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setCancelButton(None)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setMinimumDuration(0)
        dialog.setRange(0, 0)
        dialog.setLabelText("Preparing workspace and mask data…")
        dialog.setMinimumWidth(420)
        dialog.show()
        QApplication.processEvents()

        def _progress(completed: int, total: int, message: str) -> None:
            if dialog is None:
                return
            maximum = max(1, int(total))
            dialog.setRange(0, maximum)
            dialog.setValue(min(int(completed), maximum))
            dialog.setLabelText(
                f"{message}\n{min(int(completed), maximum):,} of "
                f"{maximum:,} tiles"
            )
            self._set_status(message)
            QApplication.processEvents()

        def _success(result) -> str:
            optimized = int(result.get("optimized_images", 0))
            detail = (
                f" Prepared {optimized} image(s) for multiscale viewing; "
                "reopen the workspace to use them."
                if optimized
                else ""
            )
            saved_layers = result.get("saved_layers")
            saved_path = result.get("path")
            if result.get("complete_package"):
                detail += " Portable workspace saved. Keep the workspace file and data folder together."
            return f"Saved {saved_layers} layer(s) to {saved_path}." + detail

        try:
            self._run(
                lambda: save_workspace(
                    self.viewer,
                    path,
                    optimize_large_images=optimize,
                    progress=_progress,
                    copy_labels=copy_labels,
                    complete_package=bool(self.storage_mode_combo.currentData()),
                ),
                success=_success,
                completed_path=path,
            )
        finally:
            if dialog is not None:
                dialog.close()
                dialog.deleteLater()

    def _load_path(self, path: Path) -> None:
        if self.viewer is None:
            self._set_status("No napari viewer is available.")
            return
        if self._has_pending_local_edits():
            self._set_status(
                "Apply or save the pending local-area edits before opening "
                "another project."
            )
            return
        if len(self.viewer.layers) and not self._confirm_clear(
            "Open this Label Assistant project? Current layers will be "
            "removed from the viewer. "
            "Writable Zarr data already saved on disk will not be deleted."
        ):
            return
        self._load_with_progress(path)

    def _load_with_progress(self, path: Path) -> None:
        dialog = QProgressDialog(self)
        dialog.setWindowTitle("Loading Label Assistant Project")
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setCancelButton(None)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setMinimumDuration(0)
        dialog.setRange(0, 0)
        dialog.setLabelText("Preparing project…")
        dialog.setMinimumWidth(420)
        dialog.show()
        QApplication.processEvents()

        def _progress(completed: int, total: int, message: str) -> None:
            maximum = max(1, int(total))
            if dialog.maximum() != maximum:
                dialog.setRange(0, maximum)
            dialog.setLabelText(
                f"{message}\n{min(int(completed), maximum)} of {int(total)} "
                "layers completed"
            )
            dialog.setValue(min(int(completed), maximum))
            self._set_status(message)
            QApplication.processEvents()

        try:
            result = load_workspace(
                self.viewer,
                path,
                clear_existing=True,
                progress=_progress,
            )
        except Exception as exc:
            self._set_status(f"Project loading failed: {exc}")
            QMessageBox.warning(self, "Project loading failed", str(exc))
            return
        finally:
            dialog.close()
            dialog.deleteLater()

        self.viewer._label_assistant_missing_workspace_layers = list(result["skipped_layers"])
        if result.get("storage_state"):
            self.viewer._label_assistant_workspace_storage_state = result["storage_state"]
            self.storage_mode_combo.setCurrentIndex(1 if result["storage_state"] == "portable" else 0)
        self._associate_workspace(path)
        self._refresh_current_label()
        self._refresh_recent_list()
        prefix = (
            "Imported legacy project and loaded "
            if result.get("imported_legacy")
            else "Loaded "
        )
        self._set_status(
            prefix
            + f"{len(result['restored_layers'])} of "
            f"{len(result['restored_layers']) + len(result['skipped_layers'])} layers from "
            f"{result['path']}. "
            f"Unavailable: {len(result['skipped_layers'])} layer(s)."
        )
        if result["skipped_layers"]:
            details = "\n".join(f"{item['name']}: {item['reason']}" for item in result["skipped_layers"])
            QMessageBox.warning(
                self, "Workspace layers could not be loaded",
                "Some layers failed to load. The reasons below may indicate unavailable data "
                "or a Python dependency problem. For unavailable source paths, create a "
                "workspace with Include all data (portable) selected on the source computer "
                "and transfer the entire folder.\n\n" + details,
            )

    def _run(
        self,
        operation,
        *,
        success,
        completed_path: Path | None = None,
    ) -> None:
        try:
            result = operation()
        except Exception as exc:
            self._set_status(f"Workspace operation failed: {exc}")
            QMessageBox.warning(self, "Workspace operation failed", str(exc))
            return
        if completed_path is not None:
            if result.get("storage_state"):
                self.viewer._label_assistant_workspace_storage_state = result["storage_state"]
            self._associate_workspace(completed_path)
            self._refresh_current_label()
            self._refresh_recent_list()
        self._set_status(success(result))

    def _confirm_clear(self, message: str) -> bool:
        answer = QMessageBox.question(
            self,
            "Label Assistant Project",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def _has_pending_local_edits(self) -> bool:
        controllers = getattr(
            self.viewer, "_label_assistant_edit_controllers", ()
        )
        return any(
            bool(getattr(controller, "dirty", False))
            for controller in controllers or ()
            if not bool(getattr(controller, "closed", False))
        )

    def _associate_workspace(self, path: Path) -> None:
        self.workspace_path = path.expanduser()
        self._project_layer_ids = {
            id(layer) for layer in list(getattr(self.viewer, "layers", ()))
        }
        setattr(self.viewer, VIEWER_WORKSPACE_PATH_ATTR, str(self.workspace_path))
        setattr(
            self.viewer,
            VIEWER_WORKSPACE_LAYER_IDS_ATTR,
            set(self._project_layer_ids),
        )
        self._remember(self.workspace_path)

    def _disassociate_workspace(self) -> None:
        self.workspace_path = None
        self.viewer._label_assistant_missing_workspace_layers = []
        self.viewer._label_assistant_missing_workspace_records = []
        self.viewer._label_assistant_workspace_storage_state = None
        self._project_layer_ids = set()
        setattr(self.viewer, VIEWER_WORKSPACE_PATH_ATTR, "")
        setattr(self.viewer, VIEWER_WORKSPACE_LAYER_IDS_ATTR, set())

    def _has_project_layer(self) -> bool:
        return any(
            id(layer) in self._project_layer_ids
            for layer in list(getattr(self.viewer, "layers", ()))
        )

    def _on_layer_removed(self, _event=None) -> None:
        if (
            self.workspace_path is None
            or not self._project_layer_ids
            or bool(
                getattr(
                    self.viewer, "_label_assistant_workspace_loading", False
                )
            )
            or self._has_project_layer()
        ):
            return
        previous = self.workspace_path
        self._disassociate_workspace()
        self._refresh_current_label()
        self._set_status(
            f"Project layers from {previous} were removed. The current viewer "
            "is now an unsaved project; Save will ask for a new name."
        )

    def _recent_paths(self) -> list[str]:
        value = self.settings.value(RECENT_WORKSPACES_KEY, [])
        if isinstance(value, str):
            value = [value]
        return [str(item) for item in list(value or []) if str(item).strip()]

    def _remember(self, path: Path) -> None:
        normalized = str(path.expanduser().resolve())
        self._last_workspace_path = Path(normalized)
        recent = [item for item in self._recent_paths() if item != normalized]
        recent.insert(0, normalized)
        self.settings.setValue(RECENT_WORKSPACES_KEY, recent[:RECENT_LIMIT])
        self.settings.setValue(LAST_WORKSPACE_KEY, normalized)

    def _update_storage_mode(self, *_args) -> None:
        portable = bool(self.storage_mode_combo.currentData())
        self.settings.setValue(COMPLETE_PACKAGE_KEY, portable)
        self.package_note.setText(
            "Copy all images and masks. Keep the workspace file and data folder together."
            if portable else
            "Keep saves lightweight. Linked source files must remain accessible."
        )

    def _refresh_recent_list(self) -> None:
        self.recent_list.clear()
        self.recent_menu.clear()
        paths = self._recent_paths()
        self.open_recent_button.setEnabled(bool(paths))
        for value in paths:
            path = Path(value)
            available = path.is_file()
            suffix = "" if available else " — unavailable"
            item = QListWidgetItem(f"{path.name}{suffix}\n{path.parent}")
            item.setData(Qt.UserRole, value)
            item.setToolTip(value)
            self.recent_list.addItem(item)
            action = self.recent_menu.addAction(path.name + suffix)
            action.setToolTip(value)
            action.setStatusTip(value)
            action.setEnabled(available)
            action.triggered.connect(lambda _checked=False, p=path: self._load_path(p))

    def _refresh_current_label(self) -> None:
        text = (
            str(self.workspace_path) if self.workspace_path else "Unsaved workspace"
        )
        self.current_label.setText(f"Current workspace: {self.workspace_path.name if self.workspace_path else text}")
        self.current_label.setToolTip(text)
        missing = getattr(self.viewer, "_label_assistant_missing_workspace_layers", ())
        state = getattr(self.viewer, "_label_assistant_workspace_storage_state", None)
        if missing:
            self.storage_state_label.setText(f"Incomplete workspace · {len(missing)} unavailable layer(s) · saving protected")
        elif self.workspace_path is None:
            self.storage_state_label.setText("Not saved yet")
        elif state == "portable":
            self.storage_state_label.setText("Saved data: portable · keep the workspace file and data folder together")
        elif state == "linked":
            self.storage_state_label.setText("Saved data: linked · requires source files")
        else:
            self.storage_state_label.setText("Saved data: reopen the workspace to verify storage")

    def _set_status(self, message: str) -> None:
        self.status_label.setText(str(message))
        if self.viewer is not None:
            try:
                self.viewer.status = str(message)
            except Exception:
                pass
