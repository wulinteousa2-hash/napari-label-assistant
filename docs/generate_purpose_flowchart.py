"""Generate the user-facing purpose flowchart (SVG and optional PNG)."""
from html import escape
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1560, 1090


def build_svg():
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
             '<title id="title">napari-label-assistant: purpose and workflow</title>',
             '<desc id="desc">An end-to-end visual segmentation workflow in napari: open an image, create or import masks, curate and confirm annotations, prepare reviewed training data, train and validate a model locally or with distributed compute, and predict new masks. Predictions return to human editing and review. Bounded memory use supports large 2D images approaching 100,000 by 100,000 pixels. Curated annotations can also be measured or exported, and the workspace can be saved and resumed throughout.</desc>',
             '<defs>']
    for key, color in [('main', '#324b5b'), ('learn', '#6a55a1'), ('feedback', '#bd7024')]:
        parts.append(f'<marker id="{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="{color}"/></marker>')
    parts.extend(['</defs>', f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#f8fafc"/>',
                  '<g font-family="DejaVu Sans, Arial, sans-serif">'])

    def text(x, y, value, size=17, color='#425466', weight='400', anchor='start'):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>')

    def box(x, y, w, h, fill, border, title, lines):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{border}" stroke-width="1.7"/>')
        text(x+22, y+37, title, 23, '#182f40', '700')
        for i, line in enumerate(lines):
            text(x+22, y+79+i*25, line, 17)

    def arrow(path, kind='main', dashed=False):
        color = {'main': '#324b5b', 'learn': '#6a55a1', 'feedback': '#bd7024'}[kind]
        dash = ' stroke-dasharray="9 7"' if dashed else ''
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round"{dash} marker-end="url(#{kind})"/>')

    text(60, 59, 'napari-label-assistant', 24, '#206b65', '700')
    text(60, 111, 'From large images to masks, models, and predictions', 34, '#182f40', '700')
    text(60, 146, 'Create training annotations, build models, and refine predictions in one visual workspace.', 20)

    parts.append('<rect x="60" y="165" width="1440" height="82" rx="12" fill="#def0ed"/>')
    text(80, 197, 'Built for large 2D images approaching 100k × 100k pixels', 22, '#206b65', '700')
    text(80, 226, 'Bounded memory use · full-resolution local editing · streamed component analysis', 18)

    box(60, 290, 440, 185, '#edf3f9', '#a4bad1', '1  Open image',
        ['Start from a microscopy image', 'Create a new mask or import candidates', 'Start or resume a workspace'])
    box(560, 290, 440, 185, '#eaf5ef', '#8ab89a', '2  Create & curate masks',
        ['Paint, erase and correct labels', 'Inspect connected components', 'Clean up and combine objects'])
    box(1060, 290, 440, 185, '#eaf5ef', '#8ab89a', '3  Review & confirm',
        ['Check objects AND background', 'Track regions in annotation batches', 'Locate findings by reproducible grid cell'])
    for start, end in [(500, 560), (1000, 1060)]:
        arrow(f'M{start} 382 H{end-5}')

    arrow('M1280 475 V520 H280 V570', 'learn')
    text(780, 550, 'Confirmed annotations become training supervision', 18, '#6a55a1', anchor='middle')

    box(60, 575, 440, 185, '#f2eff8', '#b1a1ce', '4  Prepare dataset',
        ['Use confirmed annotation pixels', 'Exclude unknown or unreviewed pixels', 'Reserve spatial or specimen holdouts'])
    box(560, 575, 440, 185, '#f2eff8', '#b1a1ce', '5  Build & validate model',
        ['Choose a task and model architecture', 'Train locally or with distributed compute', 'Validate on reserved regions or specimens', 'Save checkpoints and learning history'])
    box(1060, 575, 440, 185, '#f2eff8', '#b1a1ce', '6  Predict & inspect',
        ['Apply a selected model to images', 'Predict an area or an entire image', 'Inspect results in the same viewer', 'Return to editing, review, and refinement'])
    for start, end in [(500, 560), (1000, 1060)]:
        arrow(f'M{start} 667 H{end-5}', 'learn')

    arrow('M1280 760 V805 H30 V265 H780 V285', 'feedback', True)
    text(60, 845, 'Review and correct predictions before reusing them for training.', 20, '#a45d19', '500')
    text(60, 879, 'Repeat the loop with new annotation batches and model revisions.', 17)

    parts.append('<rect x="1060" y="815" width="440" height="90" rx="14" fill="#edf3f9" stroke="#a4bad1" stroke-width="1.7"/>')
    text(1082, 851, 'Measure or export', 23, '#182f40', '700')
    text(1082, 882, 'Use curated masks for downstream analysis', 17)
    arrow('M1500 382 H1530 V860 H1505')

    text(780, 935, 'Training validity: reviewed regions = foreground + background · selected objects = foreground only · unknown = excluded', 15, '#425466', anchor='middle')
    parts.append('<rect x="60" y="960" width="1440" height="92" rx="12" fill="#e7edf3"/>')
    text(80, 995, 'Save and resume at every stage', 22, '#182f40', '700')
    text(80, 1026, 'Images · masks · review and batch state · dataset settings · model records and checkpoints', 18)
    text(60, 1078, 'Training and prediction are optional; curated masks can also be used directly for measurement or export.', 16)
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
