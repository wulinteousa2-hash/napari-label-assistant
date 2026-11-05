"""Measure bounded opening of three sparse 84k masks; no production project IO."""
import json
from pathlib import Path
import resource
import tempfile
from time import perf_counter

import numpy as np
import psutil
import zarr
import napari
from qtpy.QtWidgets import QApplication

from napari_label_assistant_tools.workspace import WORKSPACE_FORMAT, load_workspace
from napari_label_assistant_tools._mask_tiles import manager_for

app = QApplication.instance() or QApplication([])
viewer = napari.Viewer(show=False)
process = psutil.Process()
with tempfile.TemporaryDirectory(prefix='label-assistant-grid-measure-') as folder:
    root = Path(folder)
    records = []
    for index,dtype in enumerate(('uint32','uint8','uint8')):
        source=zarr.open_array(str(root/f'{index}.zarr'),mode='w',shape=(84175,79966),chunks=(1024,1024),dtype=dtype)
        source[10:20,10:20]=1
        records.append(dict(layer_type='Labels',name=f'mask-{index}',
            storage=dict(kind='zarr',path=f'{index}.zarr',array_path='')))
    path=root/'project.label-assistant.json'
    path.write_text(json.dumps(dict(format=WORKSPACE_FORMAT,version=2,layers=records)))
    before=process.memory_info().rss
    started=perf_counter()
    result=load_workspace(viewer,path)
    app.processEvents()
    elapsed=perf_counter()-started
    after=process.memory_info().rss
    manager=manager_for(viewer)
    assert not result['skipped_layers'] and len(manager.entries)==3
    assert all(isinstance(layer.data,np.ndarray) and layer.data.shape==(1024,1024) for layer in viewer.layers)
    observation=dict(source_shape=[84175,79966],source_dtypes=['uint32','uint8','uint8'],
        logical_source_mask_bytes=84175*79966*6,
        visible_layer_shapes=[list(layer.data.shape) for layer in viewer.layers],
        visible_pixel_buffer_bytes=sum(layer.data.nbytes for layer in viewer.layers),
        original_edit_snapshot_bytes=sum(entry.original.nbytes for entry in manager.entries),
        rss_before_bytes=before,rss_after_bytes=after,rss_difference_bytes=after-before,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        opening_seconds=elapsed,mode='automatic -> grid',
        scope='Sparse local Zarr, three masks only, Qt offscreen; no real-desktop latency or low-end hardware guarantee.')
    viewer.close()
out=Path(__file__).resolve().parents[1]/'evidence/grid_tiles'
out.mkdir(parents=True,exist_ok=True)
(out/'opening_memory.json').write_text(json.dumps(observation,indent=2))
print(json.dumps(observation,indent=2))
