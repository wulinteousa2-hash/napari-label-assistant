"""Safe subprocess measurements. Never materializes the logical 100k image."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np

OUT=Path(__file__).resolve().parents[1]/'evidence'


def consume(task):
    progress=0
    while True:
        try:
            next(task); progress+=1
        except StopIteration as done:
            return done.value,progress


def run(kind):
    import psutil
    start=perf_counter()
    rss_before=psutil.Process().memory_info().rss
    if kind.startswith('components'):
        from napari_label_assistant_tools._large_components import build_large_component_index
        from napari_label_assistant_tools._widget import _build_fast_component_index
        factor=int(kind.split('-')[1]); shape=(512*factor,768*factor)
        mask=np.zeros(shape,np.uint16)
        for y in range(11,shape[0]-8,71):
            for x in range(13,shape[1]-8,67):
                mask[y:y+7,x:x+9]=300
        operation_start=perf_counter()
        index,progress=consume(build_large_component_index(mask,tile_size=256))
        operation_elapsed=perf_counter()-operation_start
        expected=int(np.count_nonzero(mask)//63)
        assert len(index.records)==expected
        assert sum(r.area for r in index.records.values())==np.count_nonzero(mask)
        disk_bytes=sum(p.stat().st_size for p in Path(index._cache.name).rglob('*') if p.is_file())
        result=dict(shape=shape,dtype=str(mask.dtype),components=expected,tile_size=256,
                    source_bytes=mask.nbytes,cache_bytes=disk_bytes,progress_events=progress,operation_seconds=operation_elapsed)
        index.close()
    elif kind=='logical100k':
        import zarr
        from napari_label_assistant_tools._dynamic_edit import DynamicLabelsEditController,TileBounds
        with tempfile.TemporaryDirectory(prefix='nla-logical-100k-') as temp:
            path=Path(temp)/'sparse.zarr'
            source=zarr.open_array(str(path),mode='w',shape=(100000,100000),chunks=(512,512),dtype='uint32')
            source[99900:99910,99900:99920]=2**24+1
            operation_start=perf_counter()
            tile=DynamicLabelsEditController._read_tile(source,TileBounds(99488,99488,100000,100000))
            operation_elapsed=perf_counter()-operation_start
            assert tile.shape==(512,512) and np.count_nonzero(tile)==200
            assert tile[412,412]==2**24+1
            result=dict(logical_shape=list(source.shape),logical_bytes=100000*100000*4,
                        tile_shape=list(tile.shape),tile_bytes=tile.nbytes,
                        stored_bytes=sum(p.stat().st_size for p in path.rglob('*') if p.is_file()),operation_seconds=operation_elapsed,
                        scope='one bounded edge read; no full analysis/render/inference performed')
    elif kind=='prediction':
        import torch
        from napari_label_assistant_tools._quick_train import QuickTrainConfig, QuickTrainModel,predict_region
        class Constant(torch.nn.Module):
            def forward(self,x): return torch.zeros_like(x)
        shape=(1024,1536)
        image=np.zeros(shape,np.uint16)
        config=QuickTrainConfig(crop_pixels=256,input_pixels=32,device='cpu')
        model=QuickTrainModel(Constant(),config,256,{},'cpu')
        operation_start=perf_counter()
        prediction,progress=consume(predict_region(model,image,device='cpu'))
        operation_elapsed=perf_counter()-operation_start
        assert prediction.probability.shape==shape
        np.testing.assert_allclose(prediction.probability,.5)
        result=dict(shape=shape,tiles=progress,probability_bytes=prediction.probability.nbytes,operation_seconds=operation_elapsed,
                    scope='constant network; tests stitching and allocation, not U-Net throughput')
    else:
        raise ValueError(kind)
    result.update(kind=kind,elapsed_seconds=perf_counter()-start,rss_before_bytes=rss_before,
                  rss_after_bytes=psutil.Process().memory_info().rss,
                  peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  measurement='Linux process high-water RSS includes imports; cold independent subprocess')
    return result


if __name__=='__main__':
    if len(sys.argv)>1:
        print(json.dumps(run(sys.argv[1])))
    else:
        results=[]
        for kind in ('components-1','components-2','components-4','logical100k','prediction'):
            done=subprocess.run([sys.executable,__file__,kind],capture_output=True,text=True)
            if done.returncode:
                results.append(dict(kind=kind,error=done.stderr,returncode=done.returncode))
            else:
                results.append(json.loads(done.stdout.strip().splitlines()[-1]))
        (OUT/'performance.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(results,indent=2))
