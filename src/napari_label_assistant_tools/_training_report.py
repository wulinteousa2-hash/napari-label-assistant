"""Dependency-free SVG learning curves for model and validation reports."""
from html import escape
import csv
from pathlib import Path


METRICS = {'dice': '#16a085', 'iou': '#2980b9', 'precision': '#d35400', 'recall': '#8e44ad'}


def history_points(history):
    points = []
    for round_index, record in enumerate(history, 1):
        for checkpoint in record.get('checkpoints', []):
            if checkpoint.get('validation'):
                points.append({'epoch': checkpoint.get('epoch', checkpoint.get('step', 0)),
                               'axis_unit': 'epoch' if 'epoch' in checkpoint else 'step',
                               'series': f"Round {round_index}", 'step': checkpoint.get('step', ''),
                               'training_loss_last_batch': checkpoint.get('training_loss_last_batch', checkpoint.get('loss', '')), 'metrics': checkpoint['validation']})
    if not points and history:
        report = history[-1].get('cross_validation_report', {})
        for fold in report.get('fold_results', []):
            for record in fold['history']:
                for checkpoint in record.get('checkpoints', []):
                    if checkpoint.get('validation'):
                        points.append({'epoch': checkpoint.get('epoch', 0), 'series': f"Fold {fold['fold']}", 'step': checkpoint.get('step', ''),
                                       'training_loss_last_batch': checkpoint.get('training_loss_last_batch', checkpoint.get('loss', '')), 'metrics': checkpoint['validation']})
    return points


def learning_svg(points, title='Validation learning progress', *, background='#fafafa', foreground='#333333', grid='#dddddd', dark=False, loss=False):
    width, height, left, top, plot_w, plot_h = 760, 360, 60, 55, 650, 220
    max_epoch = max([p['epoch'] for p in points] + [1])
    loss_values = [p['metrics']['loss'] for p in points if 'loss' in p['metrics']]
    maximum = max(loss_values)*1.1 if loss and loss_values else 1.0
    maximum = max(maximum, 1e-6)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           f'<rect x="0" y="0" width="{width}" height="{height}" fill="{background}"/>',
           f'<text x="60" y="25" font-size="16" fill="{foreground}">{escape(title)}</text>']
    for index in range(6):
        y = top + plot_h*(1-index/5)
        tick = f"{maximum*index/5:.3g}" if loss else f"{index*20}%"
        svg += [f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" stroke="{grid}"/>',
                f'<text x="18" y="{y+4}" font-size="12" fill="{foreground}">{tick}</text>']
    svg += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="{foreground}"/>',
            f'<text x="{left}" y="{top+plot_h+20}" font-size="12" fill="{foreground}">0</text>',
            f'<text x="{left+plot_w-20}" y="{top+plot_h+20}" font-size="12" fill="{foreground}">{max_epoch}</text>',
            f'<text x="350" y="310" font-size="12" fill="{foreground}">Epoch / checkpoint step</text>']
    colors = {'dice': '#39cfac', 'iou': '#65b7ff', 'precision': '#ffb064', 'recall': '#da9bff'} if dark else METRICS
    if loss:
        colors = {'loss': '#ffb064' if dark else '#d35400'}
    series = list(dict.fromkeys(p.get('series', 'Training') for p in points))
    for group_index, group in enumerate(series):
        rows = [p for p in points if p.get('series', 'Training') == group]
        for metric, color in colors.items():
            coords = [(left+plot_w*p['epoch']/max_epoch, top+plot_h*(1-p['metrics'][metric]/maximum)) for p in rows if metric in p['metrics']]
            if coords:
                values = ' '.join(f'{x:.2f},{y:.2f}' for x,y in coords)
                dash = ' stroke-dasharray="5,3"' if group_index % 2 else ''
                svg.append(f'<polyline points="{values}" fill="none" stroke="{color}" stroke-width="2" opacity="0.8"{dash}/>')
                for x,y in coords:
                    svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" fill="{color}"/>')
    if not points or (loss and not loss_values):
        svg.append(f'<text x="100" y="150" font-size="14" fill="{foreground}">Validation metrics appear after each evaluated epoch.</text>')
    for index, (metric,color) in enumerate(colors.items()):
        x = 60+index*165
        label = "Validation loss (lower is better)" if metric == "loss" else metric.title()
        svg.append(f'<text x="{x}" y="340" font-size="13" fill="{color}">{label}</text>')
    svg.append('</svg>')
    return '\n'.join(svg)


def write_metrics_csv(points, path):
    """Export raw validation values; absent legacy metrics remain empty."""
    fields = ['series', 'fold', 'epoch', 'step', 'validation_loss', 'dice', 'iou', 'precision', 'recall', 'training_loss_last_batch']
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for point in points:
            metrics = point['metrics']
            series = point.get('series', 'Training')
            writer.writerow({'series': series, 'fold': series[5:] if series.startswith('Fold ') else '',
                'epoch': point['epoch'] if point.get('axis_unit', 'epoch') == 'epoch' else '', 'step': point.get('step', ''),
                'validation_loss': metrics.get('loss', ''),
                **{key: metrics.get(key, '') for key in ('dice','iou','precision','recall')},
                'training_loss_last_batch': point.get('training_loss_last_batch', '')})
