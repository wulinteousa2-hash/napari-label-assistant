import xml.etree.ElementTree as ET

import pytest
from qtpy.QtGui import QColor

from napari_label_assistant_tools._training_report import learning_svg
from napari_label_assistant_tools._learning_chart import LearningChart


def test_svg_background_has_explicit_dimensions():
    svg = ET.fromstring(learning_svg([]))
    background = svg.find('{http://www.w3.org/2000/svg}rect')
    assert background.attrib['width'] == svg.attrib['width']
    assert background.attrib['height'] == svg.attrib['height']


@pytest.mark.parametrize('theme_name', ['dark', 'light'])
@pytest.mark.parametrize('has_metrics', [False, True])
def test_chart_background_covers_widget_in_both_themes(qtbot, theme_name, has_metrics):
    from napari.settings import get_settings
    from napari.utils.theme import get_theme
    settings = get_settings()
    previous = settings.appearance.theme
    try:
        settings.appearance.theme = theme_name
        chart = LearningChart()
        qtbot.addWidget(chart)
        chart.resize(480, 260)
        if has_metrics:
            chart.add_point(1, {'dice': .5, 'iou': .4, 'precision': .8, 'recall': .6})
            chart.add_point(2, {'dice': .7, 'iou': .6, 'precision': .9, 'recall': .75})
        chart.show()
        rendered = chart.grab().toImage()
        expected = QColor(get_theme(theme_name).background.as_hex())
        for x, y in ((2,2), (470,2), (2,250), (470,250), (420,50)):
            assert rendered.pixelColor(x,y).rgb() == expected.rgb()
    finally:
        settings.appearance.theme = previous


def test_loss_chart_uses_loss_units_and_csv_exports_raw_values(tmp_path):
    import csv
    from napari_label_assistant_tools._training_report import write_metrics_csv
    points = [{'epoch': 1, 'series': 'Fold 2', 'step': 20, 'metrics': {'loss': 2.4, 'dice': .8, 'iou': .7}},
              {'epoch': 2, 'series': 'Fold 2', 'step': 40, 'training_loss_last_batch': .3,
               'metrics': {'loss': 1.2, 'dice': .9, 'iou': .8}}]
    svg = learning_svg(points, loss=True)
    ET.fromstring(svg)
    assert '%' not in svg
    assert 'Validation loss (lower is better)' in svg
    assert 'polyline' in svg
    path = tmp_path / 'metrics.csv'
    write_metrics_csv(points, path)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['fold'] == '2'
    assert float(rows[0]['validation_loss']) == 2.4
    assert rows[0]['step'] == '20'
    assert rows[0]['training_loss_last_batch'] == ''
    assert float(rows[1]['training_loss_last_batch']) == .3
