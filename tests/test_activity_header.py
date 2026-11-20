"""The always-visible summary must remain trustworthy during competing updates."""
from napari_label_assistant_tools._activity import ActivityHub, ActivityIndicator
from qtpy.QtWidgets import QPlainTextEdit
import numpy as np
import pytest


def test_running_task_keeps_progress_and_summary_during_view_updates(qtbot):
    hub = ActivityHub()
    header = ActivityIndicator(hub=hub)
    qtbot.addWidget(header)
    header.resize(360, header.height())
    header.show()
    hub.set_summary('Confirmed 160 / 420', 'Batch 2 · 53 / 53 regions reviewed')
    initial_height = header.height()
    hub.report('tiles', 'busy', 'Approving tiles · 12 / 53', completed=12, total=53,
               cancel=lambda: None, priority=65)
    for _ in range(10):
        header.set_state('busy', 'Preparing view…')
        header.set_state('ready', 'View ready · Curated')
        hub.report('camera', 'busy', 'Loading edit area…', priority=10, log=False)
        hub.report('journal', 'ready', 'Tile progress saved', priority=20)
    assert header._full_message == 'Approving tiles · 12 / 53'
    assert header.progress.maximum() == 53 and header.progress.value() == 12
    assert header.confirmed_label.text() == 'Confirmed 160 / 420'
    assert 'Batch 2' in header.batch_label.text()
    assert header.height() == initial_height
    assert header.cancel_button.isVisible()
    assert len(hub.history) < 10


def test_failure_stays_visible_until_dismissed_and_details_are_available(qtbot):
    hub = ActivityHub()
    header = ActivityIndicator(hub=hub)
    qtbot.addWidget(header)
    header.show()
    hub.report('journal', 'error', 'Tile progress not saved', details='Disk is full', priority=20)
    header.set_state('ready', 'View ready · Curated')
    hub.report('training', 'ready', 'Dataset ready', priority=80)
    hub.finish('journal', 'Saved')
    assert header.state == 'error'
    assert 'Disk is full' in header.message.toolTip()
    header.details_button.click()
    assert 'Disk is full' in header._details_dialog.findChild(QPlainTextEdit).toPlainText()
    header._details_dialog.close()
    header.dismiss_button.click()
    assert header.state == 'ready'
    assert header._full_message == 'Dataset ready'


def test_stop_uses_current_task_and_progress_never_claims_unfinished_work(qtbot):
    hub = ActivityHub()
    header = ActivityIndicator(hub=hub)
    qtbot.addWidget(header)
    stopped = []
    hub.report('training', 'busy', 'Training', completed=3, total=10,
               cancel=lambda: stopped.append(True), priority=80)
    header.cancel_button.click()
    assert stopped == [True]
    hub.finish('training', 'Early stop · model retained')
    assert header.progress.value() == 3 and header.progress.maximum() == 10
    assert header.cancel_button.isHidden()


def test_completion_expires_without_losing_persistent_counts(qtbot, monkeypatch):
    from napari_label_assistant_tools import _activity
    hub = ActivityHub()
    header = ActivityIndicator(hub=hub)
    qtbot.addWidget(header)
    hub.set_summary('Confirmed 160 / 420', 'Batch 2 · 53 / 53 reviewed')
    header.set_state('ready', 'View ready · Curated')
    hub.report('tiles', 'ready', '53 tiles approved', completed=53, total=53)
    deadline = hub.records['tiles']['expires']
    monkeypatch.setattr(_activity, 'monotonic', lambda: deadline + 1)
    hub.expire()
    assert header._full_message == 'View ready · Curated'
    assert header._summary_text == ('Confirmed 160 / 420', 'Batch 2 · 53 / 53 reviewed')
    assert header.progress.isHidden()


def test_long_batch_name_cannot_hide_review_counts_in_narrow_header(qtbot):
    hub = ActivityHub()
    header = ActivityIndicator(hub=hub)
    qtbot.addWidget(header)
    header.resize(360, header.height())
    header.show()
    hub.set_summary('Confirmed 160 / 420', 'A very long annotation batch name ' * 5 + ' · 53 / 53 reviewed')
    qtbot.wait(20)
    assert '53 / 53 reviewed' in header.batch_label.text()
    assert header.batch_label.fontMetrics().horizontalAdvance(header.batch_label.text()) <= header.batch_label.width()


@pytest.mark.parametrize('fail', [False, True])
def test_workspace_uses_header_and_preserves_input_protection(make_napari_viewer, qtbot, tmp_path, monkeypatch, fail):
    from napari_label_assistant_tools._label_assistant import label_assistant_widget
    from napari_label_assistant_tools import _workspace_widget as module
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((32, 32), dtype=np.uint8), name='Image')
    viewer.add_labels(np.zeros((32, 32), dtype=np.uint8), name='Curated')
    shell = label_assistant_widget(viewer)
    qtbot.addWidget(shell)
    shell.show()
    header = shell._activity_header
    initial_height = header.height()
    monkeypatch.setattr(module, 'QProgressDialog', lambda *_: pytest.fail('Progress belongs in the fixed header'))
    monkeypatch.setattr(module.QMessageBox, 'warning', lambda *_: None)

    def save(viewer, path, *, progress, **kwargs):
        assert not shell.stack.isEnabled()
        assert not shell.training.isEnabled()
        assert header.isEnabled()
        progress(2, 10, 'Writing mask tiles')
        header.set_state('ready', 'View ready · Curated')
        assert 'Saving workspace' in header._full_message
        assert header.progress.value() == 2 and header.progress.maximum() == 10
        assert header.height() == initial_height
        if fail:
            raise OSError('Disk is full')
        return dict(path=str(path), saved_layers=2)

    monkeypatch.setattr(module, 'save_workspace', save)
    shell.workspace._save_path(tmp_path / 'project.json')
    assert shell.stack.isEnabled() and shell.training.isEnabled()
    assert header.state == ('error' if fail else 'ready')
    if fail:
        assert 'Disk is full' in header.message.toolTip()
    else:
        assert header._full_message == 'Workspace saved'
