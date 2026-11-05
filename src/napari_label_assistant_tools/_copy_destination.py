"""Prepare manual copy destinations without resampling existing annotations."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from qtpy.QtWidgets import QMessageBox

from ._coordinates import crop_transform_kwargs, mask_world_affine
from ._mask_tiles import full_source, full_data, manager_for, MaskTileManager, MaskEntry, Labels

MEMORY_LIMIT = 64 * 1024**2


def array_data(layer):
    data=layer.data
    while isinstance(data,(list,tuple)):
        data=data[0]
    if len(data.shape)==2:
        return data
    if isinstance(data,np.ndarray) and np.squeeze(data).ndim==2:
        return np.squeeze(data)
    raise ValueError('Choose a 2D source mask for automatic destination matching.')


def offset_between(source, target):
    """Exact integer-pixel placement only; never interpolate categorical IDs."""
    try:
        a, b = mask_world_affine(source), mask_world_affine(target)
        linear = np.linalg.solve(b[:2,:2], a[:2,:2])
        offset = np.linalg.solve(b[:2,:2], a[:2,2]-b[:2,2])
    except (ValueError,np.linalg.LinAlgError):
        return None
    if not np.allclose(linear,np.eye(2),rtol=0,atol=1e-9) or not np.allclose(offset,np.rint(offset),rtol=0,atol=1e-6):
        return None
    return tuple(int(value) for value in np.rint(offset))


def fitted_offset(source, target, records):
    """A crop can fit a full destination even when visible sizes differ."""
    data=array_data(source)
    if data.dtype.kind not in 'uib' or len(full_data(target).shape) != 2:
        return None
    backend=full_data(target)
    if getattr(backend,'read_only',False) or (isinstance(backend,np.ndarray) and not backend.flags.writeable):
        return None
    offset = offset_between(source,full_source(target))
    if offset is None:
        return None
    y,x=offset;h,w=data.shape;dh,dw=full_data(target).shape
    if y<0 or x<0 or y+h>dh or x+w>dw:
        return None
    limits=np.iinfo(full_data(target).dtype)
    if any(record.label_value<limits.min or record.label_value>limits.max for record in records):
        return None
    return offset


def confirmation_key(target):
    try:geometry=tuple(mask_world_affine(full_source(target)).ravel())
    except ValueError:geometry=None
    return (id(target),target.name,tuple(full_data(target).shape),geometry,
            target.metadata.get('label_assistant_review',{}).get('role'))


def confirm_manual_destination(parent, source, target, matching):
    def dimensions(layer):
        return ' × '.join(f'{size:,}' for size in reversed(full_data(layer).shape))
    message=(f'Copy selected components from “{source.name}” to “{target.name}”?\n\n'
             f'Source: {dimensions(source)} pixels\nDestination: {dimensions(target)} pixels\n\n')
    if matching:
        message+='A matching destination will be prepared automatically. The original destination will be kept as a separate layer so you can restore it.\n'
    else:
        message+='Components will be placed at their image coordinates. Existing destination pixels under the selection can be replaced.\n'
    role=target.metadata.get('label_assistant_review',{}).get('role')
    if role in ('reviewed_regions','reviewed_annotations'):
        message+='\nThis destination is a review/coverage mask. Check that you intended to choose it rather than Curated.\n'
    message+='\nCancel to choose another destination. Confirmation is remembered for repeated copies to this destination.'
    box=QMessageBox(QMessageBox.Question,'Confirm copy destination',message,
                    QMessageBox.Yes | QMessageBox.Cancel,parent)
    box.button(QMessageBox.Yes).setText('Copy')
    box.setDefaultButton(QMessageBox.Cancel)
    return box.exec()==QMessageBox.Yes


def unique_name(viewer, name):
    candidate=name;number=2
    while any(layer.name==candidate for layer in viewer.layers):
        candidate=f'{name} {number}';number+=1
    return candidate


def create_matching(viewer, source, target, name):
    """Keep the chosen name usable; preserve the previous layer as recovery."""
    from .workspace.service import _common_record
    template=full_source(source)
    data=array_data(template)
    shape=tuple(data.shape)
    if len(shape)!=2:
        raise ValueError('Choose a 2D source Labels layer for automatic destination matching.')
    dtype=np.dtype(data.dtype)
    if dtype.kind=='b':
        dtype=np.dtype('uint8')
    if dtype.kind not in 'ui':
        raise ValueError('Copying needs an integer or binary mask, not grayscale intensities.')
    if target is not None:
        promoted=np.result_type(dtype,full_data(target).dtype)
        if promoted.kind in 'ui':
            dtype=promoted
    old_name=target.name if target is not None else None
    old_visible=target.visible if target is not None else None
    backup_name=unique_name(viewer,f'{name} — original') if target is not None else None
    if target is not None:
        target.name=backup_name
    created=None
    try:
        if int(np.prod(shape))*dtype.itemsize<=MEMORY_LIMIT and not hasattr(source,'_label_assistant_mask_entry'):
            created=viewer.add_labels(np.zeros(shape,dtype=dtype),name=name,**crop_transform_kwargs(template))
        else:
            import zarr
            temporary=TemporaryDirectory(prefix='label-assistant-copy-destination-')
            path=Path(temporary.name)/'mask.zarr'
            array=zarr.open_array(str(path),mode='w',shape=shape,
                                  chunks=tuple(min(n,1024) for n in shape),dtype=dtype)
            manager=manager_for(viewer) or MaskTileManager(viewer)
            manager._temporary_stores.append(temporary)
            record=_common_record(template,'Labels')
            for key in ('quick_train','label_assistant_review','label_assistant_component_library'):
                record.pop(key,None)
            record['name']=name
            record['visible']=True
            record['opacity']=.7
            created=manager.add(record,array,{'path':path,'array_path':''})
            created._label_assistant_mask_entry.source._label_assistant_mask_temporary=True
        if target is not None:
            # Preserve the overlap only if it has an exact integer-pixel mapping.
            # Larger transfers are queued for bounded, cancellable processing.
            offset=offset_between(full_source(target),full_source(created))
            if offset is not None:
                y,x=offset;old=full_data(target);new=full_data(created)
                y0,x0=max(0,y),max(0,x)
                y1,x1=min(new.shape[0],y+old.shape[0]),min(new.shape[1],x+old.shape[1])
                if y1>y0 and x1>x0:
                    if (y1-y0)*(x1-x0)*old.dtype.itemsize>MEMORY_LIMIT:
                        created._label_assistant_copy_prefill=(old,(y0-y,x0-x,y1-y,x1-x),offset)
                    else:
                        block=np.asarray(old[y0-y:y1-y,x0-x:x1-x])
                        limits=np.iinfo(dtype)
                        if block.min()>=limits.min and block.max()<=limits.max:
                            new[y0:y1,x0:x1]=block
                            entry=getattr(created,'_label_assistant_mask_entry',None)
                            if entry is not None:
                                entry.manager.loading=True
                                try:entry.show()
                                finally:entry.manager.loading=False
        def restore():
            if target is None or not any(layer is target for layer in viewer.layers):
                raise ValueError('The original destination is no longer available.')
            if any(layer is created for layer in viewer.layers):
                created.name=unique_name(viewer,f'{name} — copied result')
                created.visible=False
            target.name=old_name
            target.visible=old_visible
            return target
        if target is not None:
            target.visible=False
        note=(f'Matched {name} automatically. Original retained as {backup_name} (hidden). ') if target is not None else f'Created matching destination {name}. '
        return created,note,restore if target is not None else None
    except Exception:
        if created is not None and any(layer is created for layer in viewer.layers):
            viewer.layers.remove(created)
        if target is not None:
            target.name=old_name
        raise


def selected_regions(source, index, ids):
    """Bounded selected source blocks for ordinary and tiled component indices."""
    data=array_data(source)
    if hasattr(index,'selected_tiles'):
        for tile,mask in index.selected_tiles(ids):
            slices=(slice(tile.y0,tile.y1),slice(tile.x0,tile.x1))
            yield slices,mask,np.asarray(data[slices]).copy()
        return
    records=[index.records[value] for value in ids]
    y0,x0=min(r.bbox_y0 for r in records),min(r.bbox_x0 for r in records)
    y1,x1=max(r.bbox_y1 for r in records),max(r.bbox_x1 for r in records)
    for y in range(y0,y1,1024):
        for x in range(x0,x1,1024):
            slices=(slice(y,min(y+1024,y1)),slice(x,min(x+1024,x1)))
            mask=np.isin(index.component_id_map[slices],ids)
            if mask.any():
                yield slices,mask,np.asarray(data[slices]).copy()


def copy_work(source,index,ids,target,offset):
    pending=getattr(target,'_label_assistant_copy_prefill',None)
    if pending is not None:
        data,(y0,x0,y1,x1),old_offset=pending
        for y in range(y0,y1,1024):
            for x in range(x0,x1,1024):
                slices=(slice(y,min(y+1024,y1)),slice(x,min(x+1024,x1)))
                values=np.asarray(data[slices]).copy()
                mask=values!=0
                if mask.any():
                    limits=np.iinfo(full_data(target).dtype)
                    if values[mask].min()<limits.min or values[mask].max()>limits.max:
                        raise ValueError('Existing labels exceed the matching mask dtype. Original is retained; restore it or choose a wider label type.')
                yield slices,mask,values,old_offset,True
        del target._label_assistant_copy_prefill
    for slices,mask,values in selected_regions(source,index,ids):
        yield slices,mask,values,offset,False


def write_region(viewer, target, slices, mask, values, offset):
    """Write an ROI into the full destination without resizing its display crop."""
    region=tuple(slice(s.start+d,s.stop+d) for s,d in zip(slices,offset))
    entry=getattr(target,'_label_assistant_mask_entry',None)
    if entry is not None:
        entry.flush()
        original=np.asarray(entry.source.data[region]).copy()
        pixels=original.copy();pixels[mask]=values[mask]
        # Reuse the existing bounded recovery/validation contract for this ROI.
        patch=object.__new__(MaskEntry)
        patch.manager=entry.manager;patch.source=entry.source;patch.bounds=region
        patch.original=original;patch.pending_pixels=None;patch.pending_invalidations=[]
        patch.layer=Labels(data=pixels,name=target.name,visible=entry.source.visible,
            opacity=target.opacity,blending=target.blending,metadata=deepcopy(entry.source.metadata),
            **crop_transform_kwargs(entry.source,(region[0].start,region[1].start)))
        try:
            patch.flush()
        except Exception:
            entry.pending_roi_writes=True
            entry.manager.error=f'Copy interrupted. Recovery records retained in {entry._recovery_dir()}.'
            raise
    else:
        block=np.asarray(target.data[region]).copy();block[mask]=values[mask]
        target.data[region]=block
        from ._component_library import annotation_changed
        annotation_changed(viewer,target,(region[0].start,region[1].start,region[0].stop,region[1].stop))
    return int(np.count_nonzero(mask))


def refresh_destination(target):
    entry=getattr(target,'_label_assistant_mask_entry',None)
    if entry is not None:
        entry.manager.loading=True
        try:entry.show()
        finally:entry.manager.loading=False
    else:
        target._label_assistant_library_precise_edit=True
        try:target.events.data()
        finally:target._label_assistant_library_precise_edit=False
        target.refresh()


def reveal_written_tile(source,target,point):
    entry=getattr(target,'_label_assistant_mask_entry',None)
    if entry is None or hasattr(source,'_label_assistant_mask_entry') or point is None:
        return
    bounds=entry.bounds
    if bounds is not None and all(s.start<=p<s.stop for s,p in zip(bounds,point)):
        return
    viewer=entry.manager.viewer
    camera=getattr(getattr(viewer,'scene',None),'camera',None)
    if camera is None:camera=viewer.camera
    center,zoom=tuple(camera.center),camera.zoom
    entry.manager.go(point[0]//entry.manager.size,point[1]//entry.manager.size)
    camera.center,camera.zoom=center,zoom
