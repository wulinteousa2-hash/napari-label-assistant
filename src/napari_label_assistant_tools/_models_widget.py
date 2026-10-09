"""Task-oriented model library and controlled benchmark configuration."""
from pathlib import Path
from dataclasses import replace
from copy import deepcopy
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QGroupBox, QComboBox,
    QLineEdit, QLabel, QPushButton, QHBoxLayout, QPlainTextEdit, QListWidget, QListWidgetItem, QFileDialog)
from qtpy.QtCore import Qt
from ._model_project import project, add_task, layer_id, remember_model, RUNTIME
from ._model_providers import ModelSpec, provider_available, spec_for

class ModelsWidget(QWidget):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
        self.viewer = trainer.viewer
        self.candidates = []
        layout = QVBoxLayout(self)
        task_group = QGroupBox("Biological task")
        form = QFormLayout(task_group)
        self.task_combo = QComboBox()
        self.task_name = QLineEdit()
        self.task_name.setPlaceholderText("e.g. Myelin segmentation")
        self.definition = QLineEdit()
        self.definition.setPlaceholderText("e.g. Myelin sheath pixels; exclude axon interiors")
        form.addRow("Task", self.task_combo)
        form.addRow("Name", self.task_name)
        form.addRow("Target definition", self.definition)
        row = QHBoxLayout()
        self.add_task_button = QPushButton("Create task")
        self.update_task_button = QPushButton("Update task")
        row.addWidget(self.add_task_button)
        row.addWidget(self.update_task_button)
        form.addRow(row)
        layout.addWidget(task_group)
        group = QGroupBox("New model configuration")
        form = QFormLayout(group)
        self.configuration_form = form
        self.name = QLineEdit()
        self.name.setPlaceholderText("Optional descriptive model name")
        self.provider = QComboBox()
        for title,key in (("Custom small U-Net", "custom"), ("Standard U-Net", "standard"), ("SMP", "smp")):
            self.provider.addItem(title,key)
        from ._model_providers import registered_provider_names
        for name in registered_provider_names():
            self.provider.addItem(name, name)
        self.architecture = QComboBox()
        self.encoder = QComboBox()
        self.encoder.setEditable(True)
        self.encoder.addItems(["resnet18", "resnet34", "resnet50", "efficientnet-b0", "mobilenet_v2"])
        self.encoder.setToolTip("Type another encoder supported by your installed SMP version.")
        self.initialization = QComboBox()
        self.width = QComboBox()
        form.addRow("Model name",self.name)
        form.addRow("Provider",self.provider)
        form.addRow("Architecture",self.architecture)
        form.addRow("Encoder",self.encoder)
        form.addRow("Initialization",self.initialization)
        form.addRow("Channel width",self.width)
        self.availability = QLabel()
        self.availability.setWordWrap(True)
        form.addRow(self.availability)
        self.to_train = QPushButton("Train this configuration →")
        self.to_train.clicked.connect(lambda: trainer.workflow_tabs.setCurrentWidget(trainer.train_page))
        form.addRow(self.to_train)
        layout.addWidget(group)
        group = QGroupBox("Saved and in-memory models")
        form = QFormLayout(group)
        self.models = QComboBox()
        self.run_combo = QComboBox()
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(140)
        form.addRow("Model",self.models)
        form.addRow("Checkpoint",self.run_combo)
        self.configuration_button = QPushButton("Use configuration for a new model")
        self.configuration_button.clicked.connect(lambda: trainer._safe(self.use_configuration))
        form.addRow(self.configuration_button)
        form.addRow(self.details)
        self.use_button = QPushButton("Select this model")
        self.use_button.setToolTip("Load the selected checkpoint when needed. Then continue training or use it for inference.")
        self.use_button.clicked.connect(lambda: trainer._safe(self.use_selected))
        form.addRow(self.use_button)
        row=QHBoxLayout()
        row.addWidget(trainer.load_button)
        trainer.save_button.setText("Export current model…")
        row.addWidget(trainer.save_button)
        form.addRow(row)
        self.current = QLabel("No current model.")
        self.current.setWordWrap(True)
        form.addRow(self.current)
        layout.addWidget(group)
        group = QGroupBox("Compare model configurations")
        form = QFormLayout(group)
        note=QLabel("Uses the prepared coverage dataset and identical spatial splits. Each candidate and fold starts fresh. Run settings are shared; the current model is retained.")
        note.setWordWrap(True)
        form.addRow(note)
        self.candidate_list=QListWidget()
        self.candidate_list.setMaximumHeight(110)
        form.addRow(self.candidate_list)
        row=QHBoxLayout()
        self.add_candidate=QPushButton("Add configuration")
        self.remove_candidate=QPushButton("Remove selected")
        row.addWidget(self.add_candidate)
        row.addWidget(self.remove_candidate)
        form.addRow(row)
        self.compare_button=QPushButton("Run comparison")
        form.addRow(self.compare_button)
        self.report_combo=QComboBox()
        form.addRow("Comparison history",self.report_combo)
        self.results=QPlainTextEdit()
        self.results.setReadOnly(True)
        self.results.setMinimumHeight(100)
        form.addRow(self.results)
        self.export_button=QPushButton("Export comparison CSV…")
        form.addRow(self.export_button)
        layout.addWidget(group)
        layout.addStretch(1)
        self.provider.currentIndexChanged.connect(self._provider_changed)
        self.task_combo.currentIndexChanged.connect(self._task_changed)
        self.models.currentIndexChanged.connect(self._model_changed)
        self.run_combo.currentIndexChanged.connect(self._details)
        self.report_combo.currentIndexChanged.connect(self._report_changed)
        self.add_task_button.clicked.connect(lambda: trainer._safe(self._add_task))
        self.update_task_button.clicked.connect(lambda: trainer._safe(self._update_task))
        self.add_candidate.clicked.connect(lambda: trainer._safe(self._add_candidate))
        self.remove_candidate.clicked.connect(self._remove_candidate)
        self.compare_button.clicked.connect(lambda: trainer._safe(trainer._start_comparison))
        self.export_button.clicked.connect(lambda: trainer._safe(self._export))
        self._provider_changed()
        self.refresh()

    def task(self):
        return next((t for t in project(self.viewer)["tasks"] if t["id"] == self.task_combo.currentData()), None)

    def ensure_task(self):
        task = self.task()
        if task is None:
            task = add_task(self.viewer, "Foreground segmentation", "Nonzero ground-truth pixels are foreground; reviewed zero pixels are background.")
            self.refresh()
        return task

    def bind_sources(self):
        task=self.ensure_task()
        sources={name: layer_id(combo.currentData()) for name,combo in (("image",self.trainer.image_combo),("ground_truth",self.trainer.truth_combo),("reviewed",self.trainer.reviewed_combo))}
        task["sources"]=sources
        return {"task_id":task["id"], "task_name":task["name"], "target_definition":task["target_definition"], "source_ids":sources, "image_id": sources["image"], "ground_truth_id": sources["ground_truth"]}

    def _add_task(self):
        add_task(self.viewer,self.task_name.text(),self.definition.text())
        self.refresh()

    def _update_task(self):
        task=self.task()
        if task is None:
            raise ValueError("Create a task first.")
        if not self.task_name.text().strip() or not self.definition.text().strip():
            raise ValueError("Give the task a name and target definition.")
        # Changing semantic meaning should create another task, not rewrite old runs.
        if any(r.get("task_id") == task["id"] for r in project(self.viewer)["models"]) and task["target_definition"] != self.definition.text().strip():
            raise ValueError("This task has trained models. Create another task to change its target definition.")
        task.update(name=self.task_name.text().strip(),target_definition=self.definition.text().strip())
        self.refresh()

    def _task_changed(self):
        task=self.task()
        project(self.viewer)["current_task"]=task["id"] if task else None
        self.task_name.setText(task["name"] if task else "")
        self.definition.setText(task["target_definition"] if task else "")
        if task:
            for name,combo in (("image",self.trainer.image_combo),("ground_truth",self.trainer.truth_combo),("reviewed",self.trainer.reviewed_combo)):
                identifier=task.get("sources",{}).get(name)
                if identifier:
                    for i in range(combo.count()):
                        layer=combo.itemData(i)
                        if layer is not None and layer.metadata.get("label_assistant_review",{}).get("id")==identifier:
                            combo.setCurrentIndex(i)
                            break
        if hasattr(self.trainer, "dataset_task_label"):
            self.trainer.dataset_task_label.setText(f"Task: {task['name']} — {task['target_definition']}" if task else "Choose a biological task in Models before training. Existing reviewed annotations remain available.")
        self._refresh_models()

    def _provider_changed(self):
        provider=self.provider.currentData()
        self.architecture.setEditable(provider not in ("custom", "standard", "smp"))
        self.architecture.clear()
        for label,key in (("U-Net","unet"),("U-Net++","unetplusplus"),("DeepLabV3+","deeplabv3plus")) if provider=="smp" else (("U-Net","unet"),):
            self.architecture.addItem(label,key)
        self.initialization.clear()
        self.initialization.addItem("Random weights","random")
        if provider=="smp":
            self.initialization.addItem("ImageNet pretrained encoder","imagenet")
        self.width.clear()
        for width in (4,8,12,16) if provider=="custom" else (16,32,64):
            self.width.addItem(str(width),width)
        self.width.setCurrentIndex(self.width.findData(8 if provider=="custom" else 32))
        self.encoder.setEnabled(provider=="smp")
        self.encoder.setVisible(provider=="smp")
        self.configuration_form.labelForField(self.encoder).setVisible(provider=="smp")
        self.width.setEnabled(provider!="smp")
        self.width.setVisible(provider!="smp")
        self.configuration_form.labelForField(self.width).setVisible(provider!="smp")
        self.availability.setText("Install in the napari environment: python -m pip install segmentation-models-pytorch" if not provider_available(provider) else "Grayscale image → foreground segmentation.")
        self.availability.setToolTip("Inputs use per-patch 1st–99th percentile normalization. SMP pretrained encoder weights are adapted to one input channel; saved checkpoints retain this preprocessing specification.")

    def spec(self):
        spec=ModelSpec(provider=self.provider.currentData(),architecture=self.architecture.currentData() or self.architecture.currentText().strip(),
            encoder=self.encoder.currentText().strip() if self.provider.currentData()=="smp" else "resnet18",
            initialization=self.initialization.currentData(),width=8 if self.provider.currentData()=="smp" else int(self.width.currentData()))
        spec.validate()
        return spec

    def refresh(self):
        state=project(self.viewer)
        self.task_combo.blockSignals(True)
        self.task_combo.clear()
        self.task_combo.addItem("Create or select a biological task…",None)
        for task in state["tasks"]:
            self.task_combo.addItem(task["name"],task["id"])
        index=self.task_combo.findData(state.get("current_task"))
        self.task_combo.setCurrentIndex(max(0,index))
        self.task_combo.blockSignals(False)
        self._task_changed()
        self.report_combo.blockSignals(True)
        self.report_combo.clear()
        for report in state["comparisons"]:
            self.report_combo.addItem(report.get("finished_at_utc",report["id"]),report["id"])
        self.report_combo.blockSignals(False)
        self._report_changed()
        self.current.setText("Current: "+spec_for(self.trainer.model.config).label() if self.trainer.model is not None and hasattr(self.trainer.model,"config") else "No current model. Select a checkpoint for continuation or inference.")

    def _refresh_models(self):
        state=project(self.viewer)
        previous=self.models.currentData()
        self.models.blockSignals(True)
        self.models.clear()
        for row in state["models"]:
            if self.task_combo.currentData() is None or row.get("task_id") == self.task_combo.currentData():
                self.models.addItem(row["name"],row["id"])
        index=self.models.findData(previous or state.get("current_model"))
        self.models.setCurrentIndex(max(0,index))
        self.models.blockSignals(False)
        self._model_changed()

    def _model_changed(self):
        row = self.selected()
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        self.run_combo.addItem("Latest checkpoint", None)
        if row:
            for number, run in enumerate(row.get("runs", []), 1):
                if run["revision"] != row["revision"]:
                    self.run_combo.addItem(f"Run {number} · {run.get('history', {}).get('started_at_utc', run['history'].get('mode', 'training'))}", run["revision"])
            selected = project(self.viewer).get("current_checkpoint")
            index = self.run_combo.findData(selected)
            self.run_combo.setCurrentIndex(max(0, index))
        self.run_combo.blockSignals(False)
        self._details()

    def checkpoint(self, row):
        return next((run for run in row.get("runs", []) if run["revision"] == self.run_combo.currentData()), row)

    def use_configuration(self):
        row = self.selected()
        if row is None:
            raise ValueError("Select a model first.")
        spec = row["spec"]
        index = self.provider.findData(spec["provider"])
        if index < 0:
            raise ValueError("Enable the custom model provider before using its configuration.")
        self.provider.setCurrentIndex(index)
        self.architecture.setCurrentIndex(self.architecture.findData(spec["architecture"]))
        if self.architecture.isEditable():
            self.architecture.setCurrentText(spec["architecture"])
        self.encoder.setCurrentText(spec["encoder"])
        self.initialization.setCurrentIndex(self.initialization.findData(spec["initialization"]))
        width_index=self.width.findData(spec["width"])
        if width_index >= 0:
            self.width.setCurrentIndex(width_index)
        self.name.setText(row["name"] + " — new model")
        self.trainer._update_model_summary()
        self.trainer._status("Configuration copied for fresh training. Current model weights retained.")

    def selected(self):
        return next((r for r in project(self.viewer)["models"] if r["id"]==self.models.currentData()),None)

    def _details(self):
        row=self.selected()
        if not row:
            self.details.setPlainText("Train a new model or load a checkpoint.")
            return
        entry=self.checkpoint(row)
        spec=ModelSpec(**row["spec"])
        runtime_key=row["id"] if entry is row else row["id"] + ":" + entry["revision"]
        status="In memory" if runtime_key in getattr(self.viewer,RUNTIME,{}) else "Saved checkpoint"
        if entry.get("checkpoint") and not Path(entry["checkpoint"]).is_file():
            status="Checkpoint unavailable — locate the original file and load it."
        if entry.get("saved_revision") != entry["revision"]:
            status+=" — updated weights not saved; Save workspace or export this checkpoint"
        if not provider_available(spec.provider):
            status+="; model provider unavailable"
        metrics=" · ".join(f"{k}: {v:.4f}" for k,v in (row.get("metrics",{}) if entry is row else entry.get("history",{}).get("retained_validation",{})).items())
        history=row.get("history",[])
        source=history[-1].get("source",{}) if history else {}
        text=f"{spec.label()}\n{status}\nParameters: {row.get('parameters',0):,}\nTraining rounds: {len(history)}\n{metrics}\nTarget: {source.get('target_definition','Not recorded')}\nCheckpoint: {entry.get('checkpoint','Not saved — workspace Save will save weights')}"
        for run in row.get("runs", []):
            history = run.get("history", {})
            text += "\n\nRun: " + history.get("started_at_utc", run["id"])
            text += f"\nMode: {history.get('mode','unknown')} · Epochs: {history.get('epochs','—')} · Steps: {history.get('steps','—')}"
            text += "\nValidation: " + str(history.get("validation", {}).get("method", "spatial sampled holdout"))
            text += "\nCheckpoint: " + run.get("checkpoint", "Pending workspace Save")
        self.details.setPlainText(text)

    def use_selected(self):
        row=self.selected()
        if row is None:
            raise ValueError("Select a trained model first.")
        entry=self.checkpoint(row)
        revision=entry["revision"]
        runtime_key=row["id"] if entry is row else row["id"] + ":" + revision
        runtime=getattr(self.viewer,RUNTIME,{})
        def selected(model):
            self.trainer._model_loaded(model,entry.get("checkpoint") or "in memory",register=False)
            project(self.viewer)["current_model"]=row["id"]
            project(self.viewer)["current_checkpoint"]=revision
            runtime[runtime_key] = model
            runtime[row["id"] + ":" + revision] = model
            self.refresh()
        if runtime_key in runtime:
            selected(runtime[runtime_key])
        elif entry.get("checkpoint"):
            self.trainer._start_worker(self.trainer._load_task,selected,entry["checkpoint"])
        else:
            raise ValueError("No checkpoint available.")

    def _add_candidate(self):
        spec=self.spec().to_dict()
        if spec in self.candidates:
            raise ValueError("This configuration is already selected.")
        self.candidates.append(spec)
        self.candidate_list.addItem(self.spec().label())

    def _remove_candidate(self):
        index=self.candidate_list.currentRow()
        if index>=0:
            self.candidates.pop(index)
            self.candidate_list.takeItem(index)

    def _report_changed(self):
        report=next((r for r in project(self.viewer)["comparisons"] if r["id"]==self.report_combo.currentData()),None)
        self.export_button.setEnabled(report is not None)
        if report is None:
            self.results.setPlainText("No comparison completed.")
            return
        lines=["Dataset snapshot: "+report["dataset"]["id"][:16]]
        for row in report["results"]:
            scores=" · ".join(f"{k}: {v['mean']:.4f}"+(f" ± {v['std']:.4f}" if v['std'] is not None else "") for k,v in row["metrics"].items())
            lines.append(ModelSpec(**row["spec"]).label()+"\n"+scores+f"\n{row['parameters']:,} parameters · {row['seconds']:.1f} seconds")
        lines.append(report["note"])
        if report.get("report_unavailable"):
            lines.append("Full report file is unavailable; saved summary metrics remain visible.")
        self.results.setPlainText("\n\n".join(lines))

    def _export(self):
        report=next(r for r in project(self.viewer)["comparisons"] if r["id"]==self.report_combo.currentData())
        path,_=QFileDialog.getSaveFileName(self,"Export model comparison","model-comparison.csv","CSV (*.csv)")
        if path:
            from ._model_comparison import export_comparison
            export_comparison(report,path)
            self.trainer._status("Comparison exported to "+path)

    def set_busy(self,busy):
        for widget in (self.task_combo,self.task_name,self.definition,self.add_task_button,self.update_task_button,
            self.name,self.provider,self.architecture,self.initialization,self.models,self.run_combo,self.use_button,self.configuration_button,
            self.add_candidate,self.remove_candidate,self.compare_button,self.to_train):
            widget.setEnabled(not busy)
        self.encoder.setEnabled(not busy and self.provider.currentData()=="smp")
        self.width.setEnabled(not busy and self.provider.currentData()!="smp")
