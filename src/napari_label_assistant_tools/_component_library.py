"""Persistent component identities and foreground-only training selections.

SQLite holds paged records and compressed spatial ID tiles. Pixel data is never
materialized for the whole source. Updates publish records and tiles in one
transaction; cancelled scans leave the previous library intact.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import closing
from hashlib import blake2b
from pathlib import Path
import tempfile
import zlib

import numpy as np

from ._large_components import LargeAnalysisCancelled, build_large_component_index
from ._quick_train import array_2d, QuickTrainCancelled

KEY = "label_assistant_component_library"
SCHEMA = 1
TILE_SIZE = 512


def _digest(data):
    data = np.ascontiguousarray(data)
    h = blake2b(digest_size=20)
    h.update(str(data.dtype).encode())
    h.update(data.tobytes())
    return h.hexdigest()


def _cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise QuickTrainCancelled()


def _connect(path, *, create=False):
    try:
        import sqlite3
    except ImportError as error:
        raise ValueError(
            "The Python environment cannot load SQLite. Repair its Python/SQLite "
            "installation before building a component library."
        ) from error
    mode = "rwc" if create else "rw"
    conn = sqlite3.connect(Path(path).resolve().as_uri()+f"?mode={mode}", uri=True, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


class ComponentLibrary:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise ValueError(f"Component library is unavailable: {self.path}")
        with closing(_connect(self.path)) as conn:
            schema = conn.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
            if schema is None or int(schema[0]) != SCHEMA:
                raise ValueError("Unsupported component library schema.")

    @classmethod
    def create(cls, path, shape, source_id, *, tile_size=TILE_SIZE):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        with closing(_connect(path, create=True)) as conn, conn:
            conn.executescript("""
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE components (
                    id INTEGER PRIMARY KEY, label_value INTEGER NOT NULL,
                    pixels INTEGER NOT NULL, y0 INTEGER NOT NULL, x0 INTEGER NOT NULL,
                    y1 INTEGER NOT NULL, x1 INTEGER NOT NULL,
                    cy REAL NOT NULL, cx REAL NOT NULL, fingerprint TEXT NOT NULL,
                    included INTEGER NOT NULL DEFAULT 0,
                    active INTEGER NOT NULL DEFAULT 1,
                    needs_review INTEGER NOT NULL DEFAULT 0);
                CREATE INDEX active_components ON components(active, id);
                CREATE INDEX selected_components ON components(active, included, needs_review, id);
                CREATE TABLE tiles (
                    row INTEGER NOT NULL, col INTEGER NOT NULL, height INTEGER NOT NULL,
                    width INTEGER NOT NULL, ids BLOB NOT NULL, digest TEXT NOT NULL,
                    PRIMARY KEY(row, col));
                CREATE TABLE dirty (y0 INTEGER, x0 INTEGER, y1 INTEGER, x1 INTEGER);
            """)
            conn.executemany("INSERT INTO meta VALUES (?,?)", [
                ("schema", str(SCHEMA)), ("source_id", source_id),
                ("height", str(shape[0])), ("width", str(shape[1])),
                ("tile_size", str(tile_size)), ("next_id", "1"),
                ("revision", "0"), ("edit_revision", "0"),
                ("total_count", "0"), ("selected_count", "0"), ("review_count", "0"),
            ])
        return cls(path)

    def state(self):
        with closing(_connect(self.path)) as conn:
            meta = dict(conn.execute("SELECT key,value FROM meta").fetchall())
            return {**meta, "revision": int(meta["revision"]),
                    "edit_revision": int(meta["edit_revision"]),
                    "pending": bool(conn.execute("SELECT 1 FROM dirty LIMIT 1").fetchone())}

    @property
    def shape(self):
        state = self.state()
        return int(state["height"]), int(state["width"])

    def records(self, *, offset=0, limit=500):
        with closing(_connect(self.path)) as conn:
            return [dict(row) for row in conn.execute(
                "SELECT * FROM components WHERE active=1 ORDER BY id LIMIT ? OFFSET ?",
                (limit, offset))]

    def counts(self):
        with closing(_connect(self.path)) as conn:
            values = dict(conn.execute(
                "SELECT key,value FROM meta WHERE key IN ('total_count','selected_count','review_count')"))
            return tuple(int(values[key]) for key in ('total_count','selected_count','review_count'))

    def selected_ids(self):
        with closing(_connect(self.path)) as conn:
            return [int(row[0]) for row in conn.execute(
                "SELECT id FROM components WHERE active=1 AND included=1 AND needs_review=0")]

    def select(self, component_id, included):
        with closing(_connect(self.path)) as conn, conn:
            if conn.execute("SELECT 1 FROM dirty LIMIT 1").fetchone():
                raise ValueError("Update the library after edits before selecting objects.")
            row = conn.execute("SELECT active,included,needs_review FROM components WHERE id=?", (int(component_id),)).fetchone()
            if row is None or not row[0]:
                raise ValueError("This component is no longer active.")
            delta = int(bool(included))-int(row["included"])
            conn.execute("UPDATE meta SET value=CAST(value AS INTEGER)+? WHERE key='selected_count'",(delta,))
            conn.execute("UPDATE meta SET value=CAST(value AS INTEGER)-? WHERE key='review_count'",(row["needs_review"],))
            conn.execute("UPDATE components SET included=?,needs_review=0 WHERE id=?",
                         (int(bool(included)), int(component_id)))
            conn.execute("UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='revision'")

    def mark_dirty(self, bounds=None):
        h, w = self.shape
        y0, x0, y1, x1 = map(int, bounds) if bounds is not None else (0, 0, h, w)
        y0, x0, y1, x1 = max(0, y0), max(0, x0), min(h, y1), min(w, x1)
        if y0 >= y1 or x0 >= x1:
            return
        with closing(_connect(self.path)) as conn, conn:
            conn.execute("INSERT INTO dirty VALUES (?,?,?,?)", (y0, x0, y1, x1))
            conn.execute("UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='edit_revision'")

    def read_ids(self, region):
        ys, xs = region
        y0, x0, y1, x1 = ys.start, xs.start, ys.stop, xs.stop
        result = np.zeros((y1-y0, x1-x0), np.int64)
        size = int(self.state()["tile_size"])
        with closing(_connect(self.path)) as conn:
            for row in conn.execute(
                "SELECT * FROM tiles WHERE row BETWEEN ? AND ? AND col BETWEEN ? AND ?",
                (y0//size, (y1-1)//size, x0//size, (x1-1)//size)):
                if not row["ids"]:
                    continue
                tile = np.frombuffer(zlib.decompress(row["ids"]), dtype=np.int64).reshape(
                    row["height"], row["width"])
                ty, tx = row["row"]*size, row["col"]*size
                a,b,c,d = max(y0,ty), max(x0,tx), min(y1,ty+tile.shape[0]), min(x1,tx+tile.shape[1])
                result[a-y0:c-y0,b-x0:d-x0] = tile[a-ty:c-ty,b-tx:d-tx]
        return result

    def verify_region(self, source, region, verified):
        """Detect external edits, even when foreground counts stay unchanged."""
        size = int(self.state()["tile_size"])
        ys, xs = region
        with closing(_connect(self.path)) as conn:
            for row in conn.execute(
                "SELECT row,col,height,width,digest FROM tiles "
                "WHERE row BETWEEN ? AND ? AND col BETWEEN ? AND ?",
                (ys.start//size,(ys.stop-1)//size,xs.start//size,(xs.stop-1)//size)):
                key = row["row"], row["col"]
                if key in verified:
                    continue
                y,x = key[0]*size,key[1]*size
                data = np.asarray(source[y:y+row["height"],x:x+row["width"]])
                if _digest(data) != row["digest"]:
                    raise ValueError("Curated pixels differ from the saved library. Update the library first.")
                verified.add(key)

    def update(self, source, *, cancel=None, verify_all=False):
        """Reindex affected components; preserve unambiguous IDs by pixel overlap.

        Known edits expand to complete old/new connected components, including
        components crossing the edited area. Explicit refresh also verifies all
        source tiles to detect external changes.
        """
        source = array_2d(source)
        if tuple(source.shape) != self.shape:
            raise ValueError("Library source dimensions changed; use a new Labels layer.")
        state = self.state()
        size = int(state["tile_size"])
        h,w = self.shape
        with closing(_connect(self.path)) as conn:
            dirty = [tuple(row) for row in conn.execute("SELECT * FROM dirty")]
            old_rows = {row["id"]: dict(row) for row in conn.execute(
                "SELECT * FROM components WHERE active=1")}
            old_hashes = {(r["row"],r["col"]):r["digest"] for r in conn.execute(
                "SELECT row,col,digest FROM tiles")}
        if not old_hashes:
            dirty = [(0,0,h,w)]
        elif verify_all:
            total = ((h+size-1)//size)*((w+size-1)//size)
            done = 0
            for y in range(0,h,size):
                for x in range(0,w,size):
                    _cancel(cancel)
                    block = np.asarray(source[y:min(y+size,h),x:min(x+size,w)])
                    if _digest(block) != old_hashes.get((y//size,x//size)):
                        dirty.append((y,x,min(y+size,h),min(x+size,w)))
                    done += 1
                    if done % 16 == 0 or done == total:
                        yield done,total,"Checking saved library against Curated"
        if not dirty:
            return self.state()
        y0 = min(r[0] for r in dirty)//size*size
        x0 = min(r[1] for r in dirty)//size*size
        y1 = min(h, ((max(r[2] for r in dirty)+size-1)//size)*size)
        x1 = min(w, ((max(r[3] for r in dirty)+size-1)//size)*size)
        # Closure includes complete old footprints and any connected foreground
        # continuing across the new boundary. Reads stay bounded to edge chunks.
        while True:
            _cancel(cancel)
            before = y0,x0,y1,x1
            for record in old_rows.values():
                if record["y0"] < y1 and record["y1"] > y0 and record["x0"] < x1 and record["x1"] > x0:
                    y0,x0 = min(y0,record["y0"]//size*size),min(x0,record["x0"]//size*size)
                    y1 = min(h,max(y1,((record["y1"]+size-1)//size)*size))
                    x1 = min(w,max(x1,((record["x1"]+size-1)//size)*size))
            extend = [False]*4
            for x in range(x0,x1,size):
                stop = min(x+size,x1)
                if y0:
                    a,b = np.asarray(source[y0-1:y0+1,x:stop])
                    extend[0] |= bool(np.any((a!=0)&(a==b)))
                if y1<h:
                    a,b = np.asarray(source[y1-1:y1+1,x:stop])
                    extend[2] |= bool(np.any((a!=0)&(a==b)))
            for y in range(y0,y1,size):
                stop = min(y+size,y1)
                if x0:
                    a = np.asarray(source[y:stop,x0-1:x0+1])
                    extend[1] |= bool(np.any((a[:,0]!=0)&(a[:,0]==a[:,1])))
                if x1<w:
                    a = np.asarray(source[y:stop,x1-1:x1+1])
                    extend[3] |= bool(np.any((a[:,0]!=0)&(a[:,0]==a[:,1])))
            y0,x0 = max(0,y0-size*extend[0]),max(0,x0-size*extend[1])
            y1,x1 = min(h,y1+size*extend[2]),min(w,x1+size*extend[3])
            if before == (y0,x0,y1,x1):
                break
            yield 0,0,"Expanding update to complete boundary-crossing objects"
        view = _RegionSource(source,(y0,x0,y1,x1))
        index = None
        try:
            try:
                index = yield from build_large_component_index(
                    view,tile_size=size,cancel_event=cancel,compute_euler=False)
            except LargeAnalysisCancelled as error:
                raise QuickTrainCancelled() from error
            new_old,old_new = defaultdict(set),defaultdict(set)
            hashes = {key:blake2b(digest_size=20) for key in index.records}
            affected = set()
            for tile in index.tiles:
                _cancel(cancel)
                ids = index._final_tile(tile)
                region = (slice(y0+tile.y0,y0+tile.y1),slice(x0+tile.x0,x0+tile.x1))
                previous = self.read_ids(region)
                affected.update(map(int,np.unique(previous[previous!=0])))
                from scipy.ndimage import find_objects
                local = index._load_local(tile)
                for local_id, bounds in enumerate(find_objects(local, max_label=tile.count), 1):
                    if bounds is None:
                        continue
                    component = int(index.provisional_to_final[tile.base+local_id])
                    fragment_mask = local[bounds]==local_id
                    for old in np.unique(previous[bounds][fragment_mask]):
                        if old:
                            new_old[component].add(int(old))
                            old_new[int(old)].add(component)
                    positions = np.argwhere(fragment_mask).astype(np.int64)
                    positions += (y0+tile.y0+bounds[0].start,x0+tile.x0+bounds[1].start)
                    hashes[component].update(positions.tobytes())
                yield tile.y0//size*((view.shape[1]+size-1)//size)+tile.x0//size+1, len(index.tiles), "Matching saved component identities"
            mapping = {}
            next_id = int(state["next_id"])
            for new,record in index.records.items():
                candidates = new_old[new]
                old = next(iter(candidates)) if len(candidates)==1 else None
                if (old is not None and len(old_new[old])==1
                        and old_rows[old]["label_value"]==record.label_value):
                    mapping[new] = old
                else:
                    mapping[new] = next_id;next_id += 1
            _cancel(cancel)
            with closing(_connect(self.path)) as conn, conn:
                current_edit = int(conn.execute(
                    "SELECT value FROM meta WHERE key='edit_revision'").fetchone()[0])
                current_revision = int(conn.execute("SELECT value FROM meta WHERE key='revision'").fetchone()[0])
                if current_edit != state["edit_revision"] or current_revision != state["revision"]:
                    raise ValueError("Annotations changed during indexing; update the library again.")
                for old in affected:
                    conn.execute("UPDATE components SET active=0,included=0 WHERE id=?",(old,))
                for new,record in index.records.items():
                    permanent = mapping[new]
                    old = old_rows.get(permanent)
                    fingerprint = hashes[new].hexdigest()
                    unchanged = old is not None and fingerprint==old["fingerprint"]
                    included = old["included"] if unchanged else 0
                    needs_review = old["needs_review"] if unchanged else int(old is not None)
                    conn.execute(
                        "INSERT OR REPLACE INTO components VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (permanent,record.label_value,record.area,
                         record.bbox_y0+y0,record.bbox_x0+x0,
                         record.bbox_y1+y0,record.bbox_x1+x0,
                         record.centroid_y+y0,record.centroid_x+x0,
                         fingerprint,included,1,needs_review))
                lookup = np.zeros(len(index.records)+1,np.int64)
                for new,permanent in mapping.items():
                    lookup[new]=permanent
                for done, tile in enumerate(index.tiles, 1):
                    _cancel(cancel)
                    region = (slice(y0+tile.y0,y0+tile.y1),slice(x0+tile.x0,x0+tile.x1))
                    ids = lookup[index._final_tile(tile)]
                    digest = _digest(np.asarray(source[region]))
                    if digest != view.digests[(tile.y0,tile.x0,tile.y1,tile.x1)]:
                        raise ValueError("Source pixels changed during indexing; update the library again.")
                    payload = zlib.compress(ids.tobytes()) if ids.any() else b""
                    conn.execute("INSERT OR REPLACE INTO tiles VALUES (?,?,?,?,?,?)",
                                 ((y0+tile.y0)//size,(x0+tile.x0)//size,
                                  ids.shape[0],ids.shape[1],payload,digest))
                    if done % 16 == 0 or done == len(index.tiles):
                        yield done,len(index.tiles),"Saving the component library"
                _cancel(cancel)
                conn.execute("UPDATE meta SET value=? WHERE key='next_id'",(str(next_id),))
                conn.execute("UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='revision'")
                conn.execute("DELETE FROM dirty")
                total,selected,review = conn.execute(
                    "SELECT count(*),coalesce(sum(included),0),coalesce(sum(needs_review),0) "
                    "FROM components WHERE active=1").fetchone()
                conn.executemany("UPDATE meta SET value=? WHERE key=?",[
                    (str(total),"total_count"),(str(selected),"selected_count"),(str(review),"review_count")])

            return self.state()
        finally:
            if index is not None:
                index.close()


class _RegionSource:
    def __init__(self, source, bounds):
        self.source = source
        self.bounds = bounds
        self.shape = bounds[2]-bounds[0],bounds[3]-bounds[1]
        self.dtype = source.dtype
        self.digests = {}

    def __getitem__(self, region):
        y,x = region
        data = np.asarray(self.source[slice(self.bounds[0]+(y.start or 0),self.bounds[0]+y.stop),
                                      slice(self.bounds[1]+(x.start or 0),self.bounds[1]+x.stop)])
        self.digests[(y.start or 0,x.start or 0,y.stop,x.stop)] = _digest(data)
        return data


def library_for(layer):
    state = getattr(layer, "metadata", {}).get(KEY)
    return ComponentLibrary(state["path"]) if state else None


def ensure_library(layer, path=None):
    library = library_for(layer)
    if library is not None:
        return library
    from ._reviewed_regions import identity
    source_id = identity(layer)
    if path is None:
        path = Path(tempfile.mkdtemp(prefix="label-assistant-library-"))/"components.sqlite"
    data = array_2d(layer.data)
    chunks = getattr(data, "chunks", ())
    tile_size = TILE_SIZE
    if len(chunks) == 2 and isinstance(chunks[0], int) and chunks[0] == chunks[1] and 128 <= chunks[0] <= 2048:
        tile_size = chunks[0]
    library = ComponentLibrary.create(path,data.shape,source_id,tile_size=tile_size)
    layer.metadata = {**layer.metadata,KEY:{"path":str(library.path),"source_id":source_id}}
    return library


def ensure_library_event(layer):
    if getattr(layer.events, "component_library", None) is None:
        layer.events.add(component_library=None)
    return layer.events.component_library


def notify_library(layer):
    """Invalidate training without triggering a full-image data refresh."""
    ensure_library_event(layer)(library_selection=True)


def annotation_changed(viewer, layer, bounds=None):
    for widget in getattr(viewer, "_label_assistant_library_widgets", ()):
        if widget.layer is layer and widget.cancel_event is not None:
            widget.cancel_event.set()
    library = library_for(layer)
    if library is not None:
        library.mark_dirty(bounds)
    invalidate_reviewed_regions(viewer,layer,bounds)


def subtract_rectangle(area, cut):
    a,b,c,d = area
    y0,x0,y1,x1 = max(a,cut[0]),max(b,cut[1]),min(c,cut[2]),min(d,cut[3])
    if y0>=y1 or x0>=x1:
        return [list(area)]
    return [list(r) for r in ((a,b,y0,d),(y1,b,c,d),(y0,b,y1,x0),(y0,x1,y1,d))
            if r[0]<r[2] and r[1]<r[3]]


def invalidate_reviewed_regions(viewer, truth, bounds=None):
    from ._reviewed_regions import KEY as REVIEW_KEY
    source_id = getattr(truth, "metadata", {}).get(REVIEW_KEY,{}).get("id")
    if not source_id:
        return
    h,w = array_2d(truth.data).shape
    cut = list(map(int,bounds)) if bounds is not None else [0,0,h,w]
    for layer in viewer.layers:
        state = layer.metadata.get(REVIEW_KEY,{})
        if state.get("role")!="reviewed_regions" or state.get("truth_id")!=source_id:
            continue
        cells = state.get("review_cells", state.get("areas", []))
        affected = [cell for cell in cells if cell[0]<cut[2] and cell[2]>cut[0]
                    and cell[1]<cut[3] and cell[3]>cut[1]]
        if not affected:
            continue
        areas = list(state.get("areas", []))
        invalid = list(state.get("invalid_regions", []))
        for cell in affected:
            areas = [part for area in areas for part in subtract_rectangle(area,cell)]
            if list(cell) not in invalid:
                invalid.append(list(cell))
        layer.metadata = {**layer.metadata,REVIEW_KEY:{**state,"areas":areas,
                          "invalid_regions":invalid,"review_cells":cells}}
        notify_library(layer)
    for widget in getattr(viewer, "_label_assistant_review_widgets", ()):
        if not widget.closed:
            widget.update_status()
            if widget.grid_config and widget.grid_config[0]() is truth:
                widget.update_review_overlay(truth)


class TrainingSupervision:
    """Union confirmed foreground with reviewed coverage; never infer background."""
    def __init__(self, truth, reviewed=None, *, library=None, invalid_regions=(), reviewed_areas=None):
        self.truth,self.reviewed,self.library = truth,reviewed,library
        self.shape = tuple(truth.shape)
        self.dtype = np.dtype("uint8")
        self.invalid_regions = tuple(tuple(r) for r in invalid_regions)
        self.reviewed_areas = reviewed_areas
        self.selected = library.selected_ids() if library else []
        self.require_background = bool(self.selected)
        self.snapshot = library.state() if library else None
        self._verified = set()

    def candidate_tiles(self, size):
        if self.reviewed is not None and self.reviewed_areas is None:
            return None  # A manually supplied coverage mask must be scanned.
        areas = list(self.reviewed_areas or [])
        if self.library is not None:
            with closing(_connect(self.library.path)) as conn:
                areas.extend(tuple(row) for row in conn.execute(
                    "SELECT y0,x0,y1,x1 FROM components WHERE active=1 AND included=1 AND needs_review=0"))
        coordinates = set()
        for y0,x0,y1,x1 in areas:
            for row in range(y0//size,(y1-1)//size+1):
                for col in range(x0//size,(x1-1)//size+1):
                    coordinates.add((row,col))
        return sorted(coordinates)

    def component_ids(self, region, truth, valid):
        if self.library is None:
            return []
        ids = self.library.read_ids(region)
        return list(map(int,np.unique(ids[(truth & valid) & (ids != 0)])))

    def __getitem__(self, region):
        ys,xs = region
        valid = (np.asarray(self.reviewed[region])!=0) if self.reviewed is not None else np.zeros(
            (ys.stop-ys.start,xs.stop-xs.start),bool)
        for a,b,c,d in self.invalid_regions:
            y0,x0,y1,x1 = max(a,ys.start),max(b,xs.start),min(c,ys.stop),min(d,xs.stop)
            if y0<y1 and x0<x1:
                valid[y0-ys.start:y1-ys.start,x0-xs.start:x1-xs.start]=False
        if self.library is not None:
            state = self.library.state()
            if state["pending"] or state["revision"]!=self.snapshot["revision"] or state["edit_revision"]!=self.snapshot["edit_revision"]:
                raise ValueError("Library selections or annotations changed; update and prepare patches again.")
            if self.selected or valid.any():
                self.library.verify_region(self.truth,region,self._verified)
            if self.selected:
                valid |= np.isin(self.library.read_ids(region),self.selected)
        return valid.astype(np.uint8)
