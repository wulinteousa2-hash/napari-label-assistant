"""Generate the user-facing purpose flowchart (SVG and optional PNG)."""
from html import escape
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1560, 950


def build_svg():
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
             '<title id="title">napari-label-assistant: purpose and workflow</title>',
             '<desc id="desc">Open microscopy images and candidate masks, edit and curate, review and confirm, then use curated labels for measurement or optional dataset preparation, training, and prediction. Predictions return to human editing. Workspace saving supports every stage.</desc>',
             '<defs>']
    for key, color in [('main', '#324b5b'), ('learn', '#6a55a1'), ('feedback', '#bd7024')]:
        parts.append(f'<marker id="{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="{color}"/></marker>')
    parts.extend(['</defs>', '<rect width="1560" height="950" fill="#f8fafc"/>',
                  '<g font-family="DejaVu Sans, Arial, sans-serif">'])

    def text(x, y, value, size=17, color='#425466', weight='400', anchor='start'):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>')

    def box(x, y, w, h, fill, border, title, lines):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{border}" stroke-width="1.7"/>')
        text(x+22, y+37, title, 23, '#182f40', '700')
        for i, line in enumerate(lines):
            text(x+22, y+70+i*25, line, 17)

    def arrow(path, kind='main', dashed=False):
        color = {'main': '#324b5b', 'learn': '#6a55a1', 'feedback': '#bd7024'}[kind]
        dash = ' stroke-dasharray="9 7"' if dashed else ''
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round"{dash} marker-end="url(#{kind})"/>')

    text(60, 59, 'napari-label-assistant', 24, '#206b65', '700')
    text(60, 111, 'Turn candidate masks into curated annotations', 36, '#182f40', '700')
    text(60, 148, 'Human-guided editing, object cleanup and spatial review — with optional model training.', 21)

    box(60, 210, 300, 150, '#edf3f9', '#a4bad1', 'Images + masks',
        ['Microscopy images', 'Manual / SAM3 / other tools', 'Start or resume a workspace'])
    box(430, 210, 300, 150, '#eaf5ef', '#8ab89a', 'Edit & curate',
        ['Paint, erase and inspect', 'Clean up connected objects', 'Combine and correct labels'])
    box(800, 210, 300, 150, '#eaf5ef', '#8ab89a', 'Review & confirm',
        ['Check targets + background', 'Confirm complete regions', 'Locate findings by grid cell'])
    box(1170, 210, 300, 150, '#def0ed', '#5e9e95', 'Curated Labels',
        ['Reusable annotations', 'For measurement, export', 'and optional model training'])
    for start, end in [(360, 430), (730, 800), (1100, 1170)]:
        arrow(f'M{start} 285 H{end-5}')

    box(1150, 405, 320, 80, '#edf3f9', '#a4bad1', 'Measure / quantify', ['Inspect components or export'])
    arrow('M1470 285 H1505 V445 H1475')
    text(280, 425, 'Optional: learn from confirmed annotations', 18, '#6a55a1')

    parts.append('<rect x="45" y="495" width="1440" height="190" rx="20" fill="#f2eff8" stroke="#c9bedf" stroke-dasharray="6 6"/>')
    box(65, 515, 385, 150, '#ffffff', '#b1a1ce', 'Prepare dataset',
        ['Use confirmed annotation pixels', 'Exclude unknown / unreviewed pixels', 'Reserve spatial or specimen holdouts'])
    box(565, 515, 385, 150, '#ffffff', '#b1a1ce', 'Train & validate',
        ['Train a model for the biological task', 'Evaluate on held-out annotations', 'Save checkpoints and run history'])
    box(1065, 515, 385, 150, '#ffffff', '#b1a1ce', 'Predict new masks',
        ['Apply a selected model', 'Predict an area or an entire image', 'Inspect and correct the results'])
    arrow('M1320 360 V395 H257 V510', 'learn')
    arrow('M450 590 H560', 'learn')
    arrow('M950 590 H1060', 'learn')
    arrow('M1450 590 H1525 V735 H30 V170 H580 V205', 'feedback', True)
    text(780, 767, 'Predictions return to human editing and review before reuse.', 20, '#a45d19', '500', 'middle')

    text(780, 813, 'Training validity: reviewed regions = foreground + background · selected objects = foreground only · unknown = excluded', 16, '#425466', anchor='middle')
    parts.append('<rect x="60" y="850" width="1410" height="64" rx="12" fill="#e7edf3"/>')
    text(80, 889, 'Across all stages: save & reopen the workspace', 21, '#182f40', '700')
    text(700, 889, 'Images · annotations · review status · model records', 19)
    text(60, 939, 'Full-resolution editing and analysis; bounded views and chunked processing for large images.', 16)
    parts.extend(['</g>', '</svg>'])
    return '\n'.join(parts) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--png', action='store_true')
    args = parser.parse_args()
    svg = ROOT / 'purpose-flowchart.svg'
    svg.write_text(build_svg())
    print(svg)
    if args.png:
        import os
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtGui import QGuiApplication, QImage, QPainter
        from PySide6.QtSvg import QSvgRenderer
        app = QGuiApplication.instance() or QGuiApplication([])
        renderer = QSvgRenderer(str(svg))
        if not renderer.isValid():
            raise RuntimeError('Invalid flowchart SVG')
        image = QImage(WIDTH*2, HEIGHT*2, QImage.Format.Format_ARGB32)
        image.fill(0)
        painter = QPainter(image)
        renderer.render(painter)
        painter.end()
        png = ROOT / 'purpose-flowchart.png'
        if not image.save(str(png)):
            raise RuntimeError(f'Could not save {png}')
        print(png)
        del app


if __name__ == '__main__':
    main()
