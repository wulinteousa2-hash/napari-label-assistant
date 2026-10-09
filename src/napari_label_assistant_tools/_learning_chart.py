"""Live Qt view of exportable validation learning curves."""
from qtpy.QtCore import QByteArray, QRectF
from qtpy.QtGui import QColor, QPainter
from qtpy.QtSvg import QSvgRenderer
from qtpy.QtWidgets import QWidget
from ._training_report import learning_svg


class LearningChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []
        self.loss_mode = False
        self.setMinimumHeight(210)
        self.setToolTip("Dice, IoU, precision and recall on validation pixels. Fold curves represent independent models; checkpoint scores are not independent test-set results.")

    def set_loss_mode(self, enabled):
        self.loss_mode = bool(enabled)
        self.update()

    def set_points(self, points):
        self.points = list(points)
        self.update()

    def add_point(self, epoch, metrics, fold=None, step=None, training_loss_last_batch=None, axis_unit="epoch", series=None):
        self.points.append({'epoch': epoch, 'axis_unit': axis_unit, 'metrics': dict(metrics), 'series': series or (f'Fold {fold}' if fold else 'Training'), 'step': step if step is not None else '', 'training_loss_last_batch': training_loss_last_batch if training_loss_last_batch is not None else ''})
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        from napari.settings import get_settings
        from napari.utils.theme import get_theme
        theme = get_theme(get_settings().appearance.theme)
        background, foreground, grid = (getattr(theme, key).as_hex() for key in ('background', 'text', 'secondary'))
        dark = QColor(background).lightness() < 128
        painter.fillRect(self.rect(), QColor(background))
        svg = learning_svg(self.points, title="Validation loss" if self.loss_mode else "Validation learning progress", loss=self.loss_mode, background=background, foreground=foreground, grid=grid, dark=dark)
        renderer = QSvgRenderer(QByteArray(svg.encode()))
        renderer.render(painter, QRectF(self.rect()))
        painter.end()
