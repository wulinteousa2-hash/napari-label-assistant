"""Optional compute settings, kept on this computer rather than in shared projects."""
from qtpy.QtCore import QSettings
from qtpy.QtWidgets import QComboBox, QFormLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QWidget

from ._cluster_training import ClusterSettings, check_cluster


class TrainingComputeWidget(QGroupBox):
    def __init__(self, trainer):
        super().__init__('Training compute')
        self.trainer = trainer
        layout = QFormLayout(self)
        self.mode = QComboBox()
        self.mode.addItem('This computer (default)', 'local')
        self.mode.addItem('This computer + one SSH worker', 'two_nodes')
        layout.addRow('Run on', self.mode)
        self.options = QWidget()
        options = QFormLayout(self.options)
        options.setContentsMargins(0, 0, 0, 0)
        preferences = QSettings('napari-label-assistant', 'training-compute')
        self.host = QLineEdit(str(preferences.value('host', '')))
        self.host.setPlaceholderText('SSH alias, e.g. spark-worker')
        self.python = QLineEdit(str(preferences.value('python', '')))
        self.python.setPlaceholderText('/path/to/worker/environment/bin/python')
        self.address = QLineEdit(str(preferences.value('address', '')))
        self.address.setPlaceholderText('This computer’s IP on the direct link')
        self.interface = QLineEdit(str(preferences.value('interface', '')))
        self.interface.setPlaceholderText('Automatic from the IP address')
        for title, name in (('SSH worker', 'host'), ('Worker Python', 'python'),
                            ('This computer’s IP', 'address'), ('Network interface', 'interface')):
            control = getattr(self, name)
            options.addRow(title, control)
            control.editingFinished.connect(lambda name=name: preferences.setValue(name, getattr(self, name).text().strip()))
        note = QLabel('Optional: both GPUs train one Custom small U-Net using prepared coverage patches and holdout validation. '
                      'Batch size is the total across both nodes. Requires existing SSH access and matching CUDA/PyTorch environments. '
                      'Worker settings stay on this computer; every new session starts with local training.')
        note.setWordWrap(True)
        options.addRow(note)
        self.check_button = QPushButton('Check both nodes')
        self.check_button.clicked.connect(lambda: trainer._safe(self.check))
        options.addRow(self.check_button)
        self.status = QLabel('Not checked. No jobs are sent until you start training.')
        self.status.setWordWrap(True)
        options.addRow(self.status)
        layout.addRow(self.options)
        self.options.hide()
        self.mode.currentIndexChanged.connect(lambda: self.options.setVisible(self.two_nodes()))

    def two_nodes(self):
        return self.mode.currentData() == 'two_nodes'

    def settings(self):
        settings = ClusterSettings(**{name: getattr(self, name).text().strip()
                                     for name in ('host', 'python', 'address', 'interface')})
        settings.validate()
        return settings

    def check(self):
        settings = self.settings()
        self.status.setText('Checking SSH, both GPUs, and PyTorch compatibility…')
        def done(report):
            free_gib = report['free_bytes'] / 1024**3
            self.status.setText(f"Connected: this GPU + {report['host']} · PyTorch {report['torch']} · "
                                f"worker reports {free_gib:.1f} GiB free. "
                                'Training memory depends on model, patch size, and batch size. '
                                'The GPU communication test runs when training starts.')
            self.trainer._status(self.status.text())
        self.trainer._start_worker(check_cluster, done, settings)

    def set_busy(self, busy):
        self.setEnabled(not busy)
