from napari_label_assistant_tools._widget import _ComponentTableWidget


def test_component_table_dark_theme_has_readable_alternating_rows(qtbot):
    table = _ComponentTableWidget(theme="dark")
    qtbot.addWidget(table)

    style = table.styleSheet()
    assert "background-color: #25262b" in style
    assert "alternate-background-color: #34363d" in style
    assert "color: #f8f9fa" in style


def test_component_table_light_theme_uses_dark_text(qtbot):
    table = _ComponentTableWidget(theme="light")
    qtbot.addWidget(table)

    style = table.styleSheet()
    assert "background-color: #ffffff" in style
    assert "alternate-background-color: #e9ecef" in style
    assert "color: #111111" in style
