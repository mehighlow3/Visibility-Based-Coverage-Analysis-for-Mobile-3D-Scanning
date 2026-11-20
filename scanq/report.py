"""Outputs of an assessment: JSON for a program, a phone page for the operator."""
import base64
import io as _io
import json
import zlib

import numpy as np

KIND_RGB = {"closer": (232, 120, 20), "enter": (200, 40, 60), "explore": (130, 60, 190),
            "no_floor": (110, 150, 170), "blocked": (120, 120, 120), "beyond": (60, 60, 60)}
KIND_TEXT = {"closer": "Stand here", "enter": "Get into this space", "explore": "Not seen yet: go and look"}
KIND_NAMES = ("closer", "enter", "explore", "no_floor", "blocked", "beyond")
TYPE_TEXT = {"wall": "walls", "floor": "floor", "ceiling": "ceiling", "other_flat": "tilted flat surfaces",
             "edge": "edges and corners", "top": "table tops and shelves", "clutter": "clutter"}
NOFLOOR_RGB = (214, 232, 236)
THING_RGB = (222, 206, 178)


def _png(rgb):
    from PIL import Image
    b = _io.BytesIO()
    Image.fromarray(rgb).save(b, "PNG", optimize=True)
    return base64.b64encode(b.getvalue()).decode()


def _up(a, k):
    return np.repeat(np.repeat(a, k, 0), k, 1)


def _band(z, F, lo=0.3, hi=2.3):
    """Patches drawn on the map: the wall band above the floor (floor and ceiling
    patches are counted in the numbers but would hide the plan)."""
    return (z > F["floor_z"] + lo) & (z < F["floor_z"] + hi)


def floor_image(A, scale=1):
    """RGB top view: floor classes, deficient patches by action kind, in image orientation."""
    F = A["F"]
    img = np.full(F["reach"].shape + (3,), 255, np.uint8)
    img[F["unknown"]] = (226, 226, 232)
    img[F["open"]] = (232, 240, 250)
    img[F["nofloor"]] = NOFLOOR_RGB
    img[F["cut"]] = (255, 214, 190)
    img[F["reach"]] = (168, 204, 236)
    img[F["blocked"]] = (50, 50, 55)
    if not A["summary"]["things_counted"]:
        img[A["thing_cols"]] = THING_RGB          # furniture: not asked for, the floor is taken to go on under it
    Q, idx = A["Q"], A["idx"]
    ij = np.floor(Q["centre"][idx, :2] / F["res"]).astype(int) - F["gl"]
    ok = ((ij >= 0) & (ij < np.array(F["reach"].shape))).all(1) & _band(Q["centre"][idx, 2], F)
    for k in (5, 4, 3, 2, 1, 0):
        m = ok & (A["kind"] == k)
        img[ij[m, 0], ij[m, 1]] = KIND_RGB[KIND_NAMES[k]]
    img[F["opening"]] = KIND_RGB["explore"]
    return _up(np.ascontiguousarray(img.transpose(1, 0, 2)[::-1]), scale)


def quality_image(A, scale=1):
    """RGB top view: finest spacing reached, worst patch per column."""
    F, Q = A["F"], A["Q"]
    sh = F["reach"].shape
    worst = np.zeros(sh)
    m = A["graded"] & _band(Q["centre"][:, 2], F)
    ij = np.floor(Q["centre"][m, :2] / F["res"]).astype(int) - F["gl"]
    ok = ((ij >= 0) & (ij < np.array(sh))).all(1)
    ach = np.where(np.isfinite(Q["achieved"][m]), Q["achieved"][m], 0.4)
    np.maximum.at(worst, (ij[ok, 0], ij[ok, 1]), ach[ok])
    img = np.full(sh + (3,), 255, np.uint8)
    img[F["open"]] = (240, 243, 247)
    if not A["summary"]["things_counted"]:
        img[A["thing_cols"]] = (236, 228, 214)
    for lv, col in ((0.025, (20, 130, 60)), (0.05, (130, 190, 80)), (0.1, (245, 190, 60)),
                    (0.2, (232, 120, 20)), (0.4, (200, 40, 60))):
        img[np.isclose(worst, lv)] = col
    return _up(np.ascontiguousarray(img.transpose(1, 0, 2)[::-1]), scale)


def to_json(A, path=None):
    F = A["F"]
    doc = dict(summary=A["summary"],
               actions=[{k: v for k, v in a.items() if k not in ("cell", "t", "t_full")} for a in A["actions"]],
               openings=[{k: v for k, v in o.items() if not k.startswith("_")} for o in F["openings"]],
               frame=dict(origin_xy=[float(v) for v in F["gl"] * F["res"]], res_m=F["res"],
                          shape=list(F["reach"].shape)))
    if path:
        with open(path, "w") as f:
            json.dump(doc, f, indent=1)
    return doc


LEVEL_RGB = ((20, 130, 60), (130, 190, 80), (245, 190, 60), (232, 120, 20), (200, 40, 60))
# fill threshold for 0, 1 (the tool's rule), 2 and 3 empty 5 cm cells of the sixteen in a patch
TOLS = (1.0, 0.9, 0.85, 0.8)
ROWS = (("wall", "Walls", "building", 0), ("floor", "Floor", "building", 1), ("ceiling", "Ceiling", "building", 2),
        ("other_flat", "Tilted flat surfaces", "building", 3), ("edge", "Edges and corners", "building", 4),
        ("top", "Table tops and shelves", "things", 5), ("clutter", "Clutter", "things", 6),
        ("outside", "Failing surface outside the windows", "outside", None))
OV_BITS = dict(things=1, under=2, behind=4, unseen=8, open=16)


def _orient(a):
    """grid (x, y, ...) -> image (row from the top, column)"""
    return np.ascontiguousarray(np.swapaxes(a, 0, 1)[::-1])


def base_image(A):
    """Neutral plan under the quality layers."""
    F = A["F"]
    img = np.full(F["reach"].shape + (3,), 255, np.uint8)
    img[F["unknown"]] = (230, 231, 235)
    img[F["open"]] = (247, 248, 250)
    img[F["nofloor"]] = (228, 239, 243)
    img[F["blocked"]] = (178, 181, 189)
    return _orient(img)


def analysis(A):
    """For the 'Surface quality' tab: area per surface class by the finest spacing reached, at
    four tolerances, and the same as map layers (worst patch per 10 cm spot), plus the
    estimated hidden surfaces. Layers are packed 3 bits per tolerance and zlib-compressed."""
    Q, F, s = A["Q"], A["F"], A["summary"]
    levels = Q["levels"]
    nl = len(levels)
    ks = int(np.argmin(np.abs(levels - s["s_req_m"])))
    f = Q["fills"][:, :nl]
    codes = np.array([np.where(ok.any(1), np.argmax(ok, 1) + 1, nl + 1) for ok in (f >= t - 1e-9 for t in TOLS)], np.uint8)
    cls, area = A["cls"], Q["area"]
    out = A["outside"] & A["indom"]
    din = A["indom"] & ~A["outside"]
    sh = F["reach"].shape
    g = A["B"]["store"].g
    kk = int(round(g / F["res"]))
    base = Q["ijk"][:, :2].astype(np.int64) * kk - F["gl"]

    def paint(lay, lat, val, op):
        for a in range(kk):
            for b in range(kk):
                ii, jj = lat[:, 0] + a, lat[:, 1] + b
                ok = (ii >= 0) & (ii < sh[0]) & (jj >= 0) & (jj < sh[1])
                op(lay, ii[ok], jj[ok], val[ok] if np.ndim(val) else val)

    def vmax(lay, i, j, v):
        np.maximum.at(lay, (i, j), v)

    def vor(lay, i, j, v):
        lay[i, j] |= v

    packed = np.zeros((len(ROWS),) + sh, np.uint16)
    rows = []
    for r, (key, name, group, k) in enumerate(ROWS):
        ix = np.flatnonzero(out if k is None else din & (cls == k))
        for t in range(len(TOLS)):
            lay = np.zeros(sh, np.uint8)
            paint(lay, base[ix], codes[t, ix], vmax)
            packed[r] |= lay.astype(np.uint16) << (3 * t)
        a = area[ix]
        rows.append(dict(key=key, name=name, group=group, m2=round(float(a.sum()), 1),
                         lv=[[round(float(a[codes[t, ix] == c].sum()), 2) for c in range(1, nl + 2)]
                             for t in range(len(TOLS))]))
    ov = np.zeros(sh, np.uint8)
    ov[A["thing_cols"]] |= OV_BITS["things"]
    ov[A["thing_cols"] & ~F["floor_seen"]] |= OV_BITS["under"]
    hs = s["hidden"]
    hid = dict(things=hs["things_footprint_m2"], under=hs["floor_under_things_m2"],
               behind=hs["wall_behind_things_m2"], unseen=hs["wall_not_seen_m2"], open=0.0)
    P = A.get("hidden_cells")
    if P is not None and len(P):
        for code, key in ((2, "behind"), (3, "unseen"), (0, "open")):
            q = P[P[:, 3] == code]
            paint(ov, np.floor(q[:, :2] / g).astype(np.int64) * kk - F["gl"], np.uint8(OV_BITS[key]), vor)
        hid["open"] = round(float((P[:, 3] == 0).sum()) * g * g, 1)
    buf = np.ascontiguousarray(packed.transpose(0, 2, 1)[:, ::-1, :]).astype("<u2").tobytes() + _orient(ov).tobytes()
    return dict(w=int(sh[0]), h=int(sh[1]), b64=base64.b64encode(zlib.compress(buf, 9)).decode(),
                rows=rows, lab=[f"{100 * l:g} cm" for l in levels] + ["worse"], ks=ks, tool_tol=1,
                lcol=["rgb(%d,%d,%d)" % c for c in LEVEL_RGB], lrgb=[list(c) for c in LEVEL_RGB], hid=hid)


def to_html(A, path, title=None):
    F = A["F"]; s = A["summary"]
    scale = 2                                   # page units per grid cell (0.1 m)
    H = F["reach"].shape[1] * scale; W = F["reach"].shape[0] * scale
    x0, y0 = F["gl"] * F["res"]

    def px(xy):
        return ((xy[0] - x0) / F["res"] * scale, H - (xy[1] - y0) / F["res"] * scale)
    tr = A["B"]["traj"][::10]
    path_d = "M" + " L".join(f"{px(p)[0]:.0f},{px(p)[1]:.0f}" for p in tr)

    def w(xy):
        return [round(float(xy[0]), 2), round(float(xy[1]), 2)]
    g = A["B"]["store"].g                       # patch size, m
    cpx = g / F["res"] * scale                  # ... in page units

    def cells(a):
        """Top-view squares (upper-left corners) of the patches action a is about."""
        c = A["Q"]["centre"][A["idx"][a["t"]], :2]
        q = np.unique(np.floor(c / g).astype(int), axis=0)
        return np.column_stack([(q[:, 0] * g - x0) / F["res"] * scale,
                                H - ((q[:, 1] + 1) * g - y0) / F["res"] * scale]).round().astype(int).ravel().tolist()
    items = [dict(id=a["id"], kind=a["kind"], x=px(a["xy"])[0], y=px(a["xy"])[1],
                  tx=px(a["target"])[0], ty=px(a["target"])[1], gain=a["gain_m2"], walk=a["walk_m"],
                  z=a["target_z"], floor=s["floor_z"], w=w(a["xy"]), tw=w(a["target"]),
                  area=a["area_m2"], new=a["new_m2"], closer=a["closer_m2"], same=a["same_m2"],
                  types=sorted([[TYPE_TEXT[k], v] for k, v in a["types"].items() if v >= 0.05], key=lambda t: -t[1]),
                  pts=cells(a)) for a in A["actions"]]
    for i, o in enumerate(F["openings"], 1):
        items.append(dict(id=f"O{i}", kind="explore", x=px(o["xy"])[0], y=px(o["xy"])[1], tx=None, ty=None,
                          gain=None, walk=o["walk_m"], opening=o["width_m"], beyond=o["beyond_m2"],
                          w=w(o["xy"]), tw=None))
    # fr: page unit -> scan coordinates, for places the operator adds on the map
    data = dict(items=items, W=W, H=H, s=s, text=dict(KIND_TEXT, added="Added by you: the list missed this"),
                col=dict({k: "rgb(%d,%d,%d)" % v for k, v in KIND_RGB.items()}, added="rgb(11,114,133)"),
                fr=dict(x0=float(x0), y0=float(y0), m=float(F["res"]) / scale),
                cpx=cpx, rpx=s["range_m"] / F["res"] * scale, upm=scale / F["res"], qa=analysis(A))
    html = _PAGE.replace("__TITLE__", title or f"Scan check: {s['scan']}") \
        .replace("__FLOOR__", _png(floor_image(A))).replace("__BASE__", _png(base_image(A))) \
        .replace("__PATH__", path_d).replace("__DATA__", json.dumps(data))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#16181d;--mute:#667085;--line:#e3e6ec;--acc:#1f6feb}
@media (prefers-color-scheme:dark){:root{--bg:#111318;--card:#1a1d24;--ink:#eceff4;--mute:#98a2b3;--line:#2a2f3a;--acc:#6ea8ff}}
*{box-sizing:border-box;min-width:0}html,body{max-width:100vw;overflow-x:hidden}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif}
header{display:flex;flex-wrap:wrap;align-items:flex-end;gap:8px 16px;padding:12px 16px 10px}
header .ttl{flex:1 1 320px}h1{font-size:17px;margin:0 0 2px}.sub{color:var(--mute);font-size:13px}
.tabs{display:flex;gap:6px;flex:1 1 300px;max-width:420px}
.tabs button{flex:1;padding:8px 10px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);font:inherit;cursor:pointer}
.tabs button.on{border-color:var(--acc);color:var(--acc);font-weight:600}
.mapwrap{position:relative;margin:0 16px;height:62vh;min-height:320px;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#fff;touch-action:none;cursor:grab;outline:none;user-select:none;-webkit-user-select:none}
.mapwrap:focus-visible{box-shadow:0 0 0 2px var(--acc)}.mapwrap.drag{cursor:grabbing}.mapwrap.adding{cursor:crosshair}
#svg{position:absolute;inset:0;width:100%;height:100%;display:block}
.ctl{position:absolute;right:10px;top:10px;display:flex;flex-direction:column;gap:6px}
.ctl button{width:36px;height:36px;border-radius:8px;border:1px solid #d0d5dd;background:#fff;color:#1d2433;font:600 18px/1 system-ui,sans-serif;box-shadow:0 1px 3px rgba(0,0,0,.18);cursor:pointer;padding:0}
.ctl button:hover{background:#f2f4f7}
.scale{position:absolute;left:10px;bottom:10px;font-size:11px;color:#1d2433;background:rgba(255,255,255,.88);padding:2px 6px 3px;border-radius:4px;pointer-events:none}
.scale i{display:block;height:5px;border:1.5px solid #1d2433;border-top:0;margin-top:1px}
.hint{position:absolute;right:10px;bottom:10px;font-size:11px;color:#475467;background:rgba(255,255,255,.88);padding:3px 7px;border-radius:4px;pointer-events:none}
@media (pointer:coarse){.hint{display:none}}
.panel{padding:4px 0 8px}
.tiles{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;padding:8px 16px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 10px}
.tile b{display:block;font-size:19px;font-variant-numeric:tabular-nums}.tile span{font-size:12px;color:var(--mute)}
.bar{display:flex;height:10px;border-radius:5px;overflow:hidden;margin:4px 16px 2px;background:var(--line)}
.legend{display:flex;flex-wrap:wrap;gap:4px 12px;padding:2px 16px 8px;font-size:12px;color:var(--mute)}
.legend i,.lgd i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
h2{font-size:14px;margin:16px 16px 6px;color:var(--mute);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.list{padding:0 16px 16px;display:grid;gap:8px}
.act{display:grid;grid-template-columns:38px 1fr auto;gap:10px;align-items:center;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px;cursor:pointer}
.act:hover{border-color:color-mix(in srgb,var(--acc) 45%,var(--line))}
.act.sel{border-color:var(--acc);box-shadow:0 0 0 2px color-mix(in srgb,var(--acc) 30%,transparent)}
.num{width:34px;height:34px;border-radius:17px;color:#fff;display:grid;place-items:center;font-weight:700;font-size:13px}
.act small{color:var(--mute);display:block}.gain{font-variant-numeric:tabular-nums;font-weight:600;white-space:nowrap}
.gain small{font-weight:400;text-align:right}
.why{grid-column:1/-1;font-size:13px;color:var(--mute);border-top:1px solid var(--line);padding-top:8px}.why b{color:var(--ink)}
.done{opacity:.55}.done .what{text-decoration:line-through}
.chips{grid-column:1/-1;display:flex;flex-wrap:wrap;gap:6px}
.chips button{padding:7px 11px;border-radius:16px;border:1px solid var(--line);background:var(--bg);color:var(--ink);font:inherit;font-size:13px;cursor:pointer}
.chips button.on,.tools button.on{background:var(--acc);border-color:var(--acc);color:#fff}
.act small.tag{color:var(--acc);font-weight:600}
.tools{display:flex;gap:8px;padding:10px 16px 0}.tools button{flex:1;padding:9px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);font:inherit;font-size:13px;cursor:pointer}
.note{padding:6px 16px 0;color:var(--mute);font-size:12px}
.foot{padding:0 16px 20px;color:var(--mute);font-size:12px}
.box{margin:10px 16px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.box h3{margin:0 0 6px;font-size:12px;color:var(--mute);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.big{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}.big b{font-size:30px;font-variant-numeric:tabular-nums}
.mut{color:var(--mute);font-size:13px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;vertical-align:middle}
.seg button{border:0;border-right:1px solid var(--line);background:var(--card);color:var(--ink);padding:5px 13px;font:inherit;cursor:pointer}
.seg button:last-child{border-right:0}.seg button.on{background:var(--acc);color:#fff}
.tolrow{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-top:10px;font-size:13px}
.tolx{display:flex;gap:10px;align-items:center;margin-top:8px}
.cells{display:grid;grid-template-columns:repeat(4,9px);gap:2px;flex:none}
.cells i{width:9px;height:9px;border-radius:2px;background:var(--mute)}.cells i.e{background:none;box-shadow:inset 0 0 0 1.5px var(--acc)}
.cr{display:grid;grid-template-columns:38px 1fr 58px 46px 38px;gap:3px 8px;align-items:center;padding:7px 0 6px;border-top:1px solid var(--line)}
.cr.hd{border-top:0;font-size:11px;color:var(--mute);padding:0 0 2px}
.cr .n{font-size:14px}.cr .v{text-align:right;font-variant-numeric:tabular-nums;font-size:13px}
.cr .lv{grid-column:2/5;display:flex;height:7px;border-radius:4px;overflow:hidden;background:var(--line)}
.cr.off .n,.cr.off .v{color:var(--mute)}.cr.off .lv{opacity:.45}
.cr.tot{border-top:2px solid var(--line)}.cr.tot .n,.cr.tot .v{font-weight:700}
.grp{font-size:11px;color:var(--mute);padding:8px 0 3px;text-transform:uppercase;letter-spacing:.04em;border-top:1px solid var(--line)}
.sw{position:relative;display:inline-block;width:34px;height:20px;flex:none}
.sw input{opacity:0;width:0;height:0;position:absolute}
.sw span{position:absolute;inset:0;background:#c3c8d2;border-radius:10px;transition:.15s;cursor:pointer}
.sw span:before{content:"";position:absolute;width:16px;height:16px;left:2px;top:2px;background:#fff;border-radius:50%;transition:.15s;box-shadow:0 1px 2px rgba(0,0,0,.3)}
.sw input:checked+span{background:var(--acc)}.sw input:checked+span:before{transform:translateX(14px)}
.sw input:focus-visible+span{outline:2px solid var(--acc);outline-offset:2px}
.hr{display:grid;grid-template-columns:12px 1fr auto 38px;gap:8px;align-items:center;padding:7px 0;border-top:1px solid var(--line);font-size:14px}
.hr:first-of-type{border-top:0}.hr i{width:12px;height:12px;border-radius:3px;display:block}.hr .v{font-variant-numeric:tabular-nums;font-size:13px;color:var(--mute)}
.lgd{display:flex;flex-wrap:wrap;gap:4px 12px;font-size:12px;color:var(--mute);margin-top:8px}
.linkbtn{border:0;background:none;color:var(--acc);font:inherit;cursor:pointer;padding:0;text-decoration:underline}
details{margin:10px 16px 20px;font-size:13px;color:var(--mute)}summary{cursor:pointer;color:var(--ink);font-weight:600}details p{margin:6px 0}
@media (min-width:900px){
 body{height:100vh;display:flex;flex-direction:column;overflow:hidden}
 .main{flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,1fr) 440px;grid-template-rows:minmax(0,1fr)}
 .mapwrap{height:auto;min-height:0;margin:0 0 16px 16px}
 .panel{overflow-y:auto;padding:0 0 8px}
}
</style></head><body>
<header><div class="ttl"><h1>__TITLE__</h1><div class="sub" id="sub"></div></div>
<div class="tabs"><button id="t0" class="on">Where to go</button><button id="t1">Surface quality</button></div></header>
<div class="main">
<div class="mapwrap" id="map" tabindex="0" aria-label="Map. Drag to move, scroll or plus and minus to zoom, arrow keys to move, 0 for the whole map.">
<svg id="svg" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
<image id="img0" href="data:image/png;base64,__FLOOR__"/>
<g id="qg" style="display:none"><image id="imgb" href="data:image/png;base64,__BASE__"/><image id="imgq"/></g>
<path id="walk" d="__PATH__" fill="none" stroke="#1b7a3a" stroke-width="2" opacity=".75" vector-effect="non-scaling-stroke"/>
<g id="hl" style="pointer-events:none"></g><g id="marks"></g></svg>
<div class="ctl"><button id="zi" title="Zoom in (+)" aria-label="Zoom in">+</button><button id="zo" title="Zoom out (−)" aria-label="Zoom out">−</button><button id="zf" title="Whole map (0)" aria-label="Whole map">⤢</button></div>
<div class="scale" id="scale"></div>
<div class="hint">drag to move · scroll to zoom · arrows, + − 0</div>
</div>
<div class="panel" id="panel">
<div id="pP">
<div class="tiles" id="tiles"></div>
<div class="bar" id="bar"></div><div class="legend" id="legend"></div>
<div class="tools"><button id="add">+ Add a place the list missed</button><button id="save">Save labels</button></div>
<div class="note" id="saved"></div>
<h2>What is left to do</h2><div class="list" id="list"></div>
<div class="foot" id="foot"></div>
</div>
<div id="pQ" style="display:none"></div>
</div></div>
<script>
const D=__DATA__, s=D.s, QA=D.qa, NS='http://www.w3.org/2000/svg';
const $=id=>document.getElementById(id), svg=$('svg'), wrap=$('map'), marks=$('marks'), hl=$('hl'), list=$('list');
const fmt=v=>v>=100?Math.round(v).toLocaleString('en'):v.toFixed(1);
for(const i of ['img0','imgb','imgq']){const e=$(i);e.setAttribute('width',D.W);e.setAttribute('height',D.H);e.setAttribute('preserveAspectRatio','none');e.style.imageRendering='pixelated';}
$('sub').textContent=`Required point spacing ${s.s_req_m*100} cm · scan from within ${s.range_m} m`+(s.outside_m2>0?` · ${Math.round(s.outside_m2)} m² outside the windows not counted`:'')+
 (s.things_counted?' · furniture counted':` · ${Math.round(s.things_m2)} m² of furniture and clutter not counted`);

// ---------- map: pan and zoom like a web map ----------
let vb={x:0,y:0,w:D.W,h:D.H}, anim=0, last=null;
const CW=()=>wrap.clientWidth||1, CH=()=>wrap.clientHeight||1;
const kFit=()=>Math.max(D.W/CW(),D.H/CH())*1.04, clampK=k=>Math.min(kFit()*1.6,Math.max(0.04,k));
const cur=()=>({cx:vb.x+vb.w/2,cy:vb.y+vb.h/2,k:vb.w/CW()});
function setView(cx,cy,k){k=clampK(k); cx=Math.min(D.W,Math.max(0,cx)); cy=Math.min(D.H,Math.max(0,cy));
 vb={x:cx-k*CW()/2,y:cy-k*CH()/2,w:k*CW(),h:k*CH()}; last={cx,cy,k}; svg.setAttribute('viewBox',`${vb.x} ${vb.y} ${vb.w} ${vb.h}`); onView(k);}
const fitView=()=>({cx:D.W/2,cy:D.H/2,k:kFit()});
function animTo(t,ms=280){cancelAnimationFrame(anim); const a=cur(),t0=performance.now(),l0=Math.log(a.k),l1=Math.log(clampK(t.k));
 const step=now=>{let u=Math.min(1,(now-t0)/ms); u=1-Math.pow(1-u,3);
  setView(a.cx+(t.cx-a.cx)*u,a.cy+(t.cy-a.cy)*u,Math.exp(l0+(l1-l0)*u)); if(u<1) anim=requestAnimationFrame(step);};
 anim=requestAnimationFrame(step);}
function zoomTarget(px,py,f){const c=cur(),k2=clampK(c.k/f),ux=vb.x+px*c.k,uy=vb.y+py*c.k;
 return {cx:ux-px*k2+k2*CW()/2,cy:uy-py*k2+k2*CH()/2,k:k2};}
const zoomAt=(px,py,f)=>{const t=zoomTarget(px,py,f); setView(t.cx,t.cy,t.k);};
const local=e=>{const r=wrap.getBoundingClientRect(); return [e.clientX-r.left,e.clientY-r.top];};
function onView(k){
 for(const g of marks.children) if(g.dataset&&g.dataset.x) g.setAttribute('transform',`translate(${g.dataset.x} ${g.dataset.y}) scale(${k})`);
 const mpp=k/D.upm, want=110*mpp, nice=[0.5,1,2,5,10,20,50,100,200].filter(v=>v<=want).pop()||0.5;
 $('scale').innerHTML=`${nice} m<i style="width:${Math.round(nice/mpp)}px"></i>`;}
const pts=new Map(); let drag=null, moved=false, pinch=null, hover=false;
wrap.addEventListener('pointerdown',e=>{
 if(e.pointerType==='mouse'&&e.button!==0) return;
 if(e.target.closest('.ctl')) return;
 cancelAnimationFrame(anim); pts.set(e.pointerId,{x:e.clientX,y:e.clientY});
 if(pts.size===1){drag={x:e.clientX,y:e.clientY,vx:vb.x,vy:vb.y,id:e.pointerId}; moved=false;}
 else if(pts.size===2){const [a,b]=[...pts.values()]; pinch={d:Math.hypot(a.x-b.x,a.y-b.y),m:{x:(a.x+b.x)/2,y:(a.y+b.y)/2}}; moved=true;
  for(const id of pts.keys()) try{wrap.setPointerCapture(id)}catch(_){}}});
wrap.addEventListener('pointermove',e=>{ if(!pts.has(e.pointerId)) return; pts.set(e.pointerId,{x:e.clientX,y:e.clientY});
 if(pts.size>=2&&pinch){const [a,b]=[...pts.values()], d=Math.hypot(a.x-b.x,a.y-b.y), m={x:(a.x+b.x)/2,y:(a.y+b.y)/2}, r=wrap.getBoundingClientRect(), c=cur();
  setView(c.cx-(m.x-pinch.m.x)*c.k,c.cy-(m.y-pinch.m.y)*c.k,c.k); zoomAt(m.x-r.left,m.y-r.top,d/Math.max(pinch.d,1)); pinch={d,m}; return;}
 if(drag&&e.pointerId===drag.id){const dx=e.clientX-drag.x, dy=e.clientY-drag.y;
  if(!moved&&Math.hypot(dx,dy)>4){moved=true; wrap.classList.add('drag'); try{wrap.setPointerCapture(e.pointerId)}catch(_){}}
  if(moved){const k=cur().k; setView(drag.vx-dx*k+vb.w/2,drag.vy-dy*k+vb.h/2,k);}}});
const up=e=>{pts.delete(e.pointerId); if(pts.size<2) pinch=null;
 if(!pts.size){wrap.classList.remove('drag'); drag=null;}
 else{const [id,p]=[...pts.entries()][0]; drag={x:p.x,y:p.y,vx:vb.x,vy:vb.y,id};}};
wrap.addEventListener('pointerup',up); wrap.addEventListener('pointercancel',up);
wrap.addEventListener('click',e=>{if(moved){e.stopPropagation(); e.preventDefault(); moved=false;}},true);
wrap.addEventListener('wheel',e=>{e.preventDefault(); cancelAnimationFrame(anim); const [x,y]=local(e), dy=e.deltaMode===1?e.deltaY*33:e.deltaMode===2?e.deltaY*400:e.deltaY;
 zoomAt(x,y,Math.exp(-Math.max(-300,Math.min(300,dy))*0.0022));},{passive:false});
wrap.addEventListener('dblclick',e=>{if(e.target.closest('.ctl,#marks')) return; const [x,y]=local(e); animTo(zoomTarget(x,y,e.shiftKey?0.5:2),250);});
wrap.addEventListener('mouseenter',()=>hover=true); wrap.addEventListener('mouseleave',()=>hover=false);
document.addEventListener('keydown',e=>{
 if(e.ctrlKey||e.metaKey||e.altKey) return;
 const inMap=document.activeElement===wrap||(hover&&!(e.target.closest&&e.target.closest('input,button,select,textarea')));
 if(!inMap) return;
 const c=cur(), st=0.3; let t=null;
 switch(e.key){
  case 'ArrowLeft': t={cx:c.cx-vb.w*st,cy:c.cy,k:c.k}; break; case 'ArrowRight': t={cx:c.cx+vb.w*st,cy:c.cy,k:c.k}; break;
  case 'ArrowUp': t={cx:c.cx,cy:c.cy-vb.h*st,k:c.k}; break; case 'ArrowDown': t={cx:c.cx,cy:c.cy+vb.h*st,k:c.k}; break;
  case '+': case '=': t=zoomTarget(CW()/2,CH()/2,1.8); break; case '-': case '_': t=zoomTarget(CW()/2,CH()/2,1/1.8); break;
  case '0': case 'Home': t=fitView(); break;
  case 'Escape': if(adding) setadd(false); else if(sel!=null){sel=null; draw();} e.preventDefault(); return;
  default: return;}
 e.preventDefault(); animTo(t,180);});
$('zi').onclick=()=>animTo(zoomTarget(CW()/2,CH()/2,1.8)); $('zo').onclick=()=>animTo(zoomTarget(CW()/2,CH()/2,1/1.8)); $('zf').onclick=()=>animTo(fitView());
const reflow=()=>{if(!wrap.clientWidth||!wrap.clientHeight) return; if(!last){const f=fitView(); setView(f.cx,f.cy,f.k);} else setView(last.cx,last.cy,last.k);};
if(window.ResizeObserver) new ResizeObserver(reflow).observe(wrap); else window.addEventListener('resize',reflow);

// ---------- tab 1: where to go ----------
const k=s.deficit_by_kind;
$('tiles').innerHTML=
 `<div class="tile"><b>${s.met_pct}%</b><span>of ${fmt(s.surface_m2)} m² scanned ${s.things_counted?'surface':'building surface'} meets it</span></div>`+
 `<div class="tile"><b>${fmt(s.actions_gain_m2)} m²</b><span>expected from the list</span></div>`+
 `<div class="tile"><b id="ntogo"></b><span>places to go</span></div>`;
const parts=[['met',s.met_m2,'#2e9e5b','meets requirement'],['closer',k.closer,D.col.closer,'stand closer'],['enter',k.enter,D.col.enter,'enter a space'],
 ['explore',k.explore,D.col.explore,'not seen yet'],['blocked',k.blocked,D.col.blocked,'view blocked'],['beyond',k.beyond,D.col.beyond,'out of reach'],
 ['limit',k.limit,'#b9a26a','seen from within '+(s.range_m/2)+' m, still gappy']];
$('bar').innerHTML=parts.map(p=>`<div style="flex:${p[1]};background:${p[2]}"></div>`).join('');
$('legend').innerHTML=parts.map(p=>`<span><i style="background:${p[2]}"></i>${p[3]} ${fmt(p[1])} m²</span>`).join('');
// Labels: what the operator says about each place. 'missed' confirms it; the others take it off the list.
// Kept in the browser, written to labels_<scan>.json with "Save labels". Positions are in scan coordinates,
// so a label still finds its place when the scan is assessed again and the numbering changes.
const LAB=[['missed','Missed'],['done','Rescanned'],['outside','Outside'],['glass','Behind glass'],['closed','No access'],['unwanted','Not wanted'],['wrong','Nothing there']];
const LABN=Object.fromEntries(LAB), KEY='labels-'+s.scan;
let st={labels:{},added:[]}; try{const o=JSON.parse(localStorage.getItem(KEY)||'null'); if(o&&o.labels&&o.added) st=o;}catch(e){}
for(const id of Object.keys(st.labels)){const a=D.items.concat(st.added).find(i=>String(i.id)===id), l=st.labels[id];
 if(!a||Math.hypot(a.w[0]-l.xy[0],a.w[1]-l.xy[1])>0.5) delete st.labels[id];}
const keep=()=>{try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}};
const all=()=>D.items.concat(st.added), lab=a=>(st.labels[a.id]||{l:[]}).l, off=a=>lab(a).some(k=>k!=='missed');
function tog(a,k){const e=st.labels[a.id]||(st.labels[a.id]={xy:a.w,l:[]}), i=e.l.indexOf(k); i<0?e.l.push(k):e.l.splice(i,1);
 if(!e.l.length) delete st.labels[a.id]; keep(); draw();}
let sel=null;
function draw(){
 marks.innerHTML=''; hl.innerHTML=''; list.innerHTML='';
 $('ntogo').textContent=all().filter(a=>!off(a)).length;
 const cs=all().find(i=>i.id===sel);
 if(cs&&cs.pts){let d=''; const q=D.cpx; for(let i=0;i<cs.pts.length;i+=2) d+=`M${cs.pts[i]} ${cs.pts[i+1]}h${q}v${q}h${-q}z`;
  hl.innerHTML=`<circle cx="${cs.x}" cy="${cs.y}" r="${D.rpx}" fill="none" stroke="${D.col[cs.kind]}" stroke-width="2" stroke-dasharray="7 6" vector-effect="non-scaling-stroke" opacity=".85"/>`+
   `<path d="${d}" fill="#ffe600" stroke="#5b4300" stroke-width=".8" vector-effect="non-scaling-stroke"/>`;}
 for(const a of all()) if(a.tx!=null&&sel!==a.id){const l=document.createElementNS(NS,'line');
  for(const [n,v] of [['x1',a.x],['y1',a.y],['x2',a.tx],['y2',a.ty],['stroke',D.col[a.kind]],['stroke-width',2.5],['stroke-dasharray','6 5'],['vector-effect','non-scaling-stroke'],['opacity',off(a)?.3:.9]]) l.setAttribute(n,v);
  l.style.pointerEvents='none'; marks.appendChild(l);}
 const order=all().slice().sort((a,b)=>(a.id===sel)-(b.id===sel));
 for(const a of order){
  const c=D.col[a.kind], g=document.createElementNS(NS,'g'), r=sel===a.id?17:13; g.dataset.x=a.x; g.dataset.y=a.y; g.style.cursor='pointer';
  g.innerHTML=`<circle r="${r}" fill="${c}" stroke="#fff" stroke-width="2.5" opacity="${off(a)?.4:1}"/>`+
   `<text y="4.5" text-anchor="middle" font-size="${String(a.id).length>2?10:12}" font-weight="700" fill="#fff" font-family="system-ui,sans-serif">${a.id}</text>`;
  const t=document.createElementNS(NS,'title'); t.textContent=`${a.id}: ${a.opening?'Opening not walked into':D.text[a.kind]}${a.gain!=null?', +'+fmt(a.gain)+' m² expected':''}`; g.appendChild(t);
  g.addEventListener('click',e=>{e.stopPropagation(); pick(a.id,true);}); marks.appendChild(g);}
 for(const a of all()){
  const c=D.col[a.kind], L=lab(a), el=document.createElement('div'); el.className='act'+(sel===a.id?' sel':'')+(off(a)?' done':''); el.id='a'+a.id;
  let what=D.text[a.kind], more=[];
  if(a.opening){what='Opening not walked into'; more.push(`${a.opening} m wide, ${fmt(a.beyond)} m² unseen behind`);}
  if(a.types&&a.types.length) more.push('mostly '+a.types.slice(0,2).map(t=>t[0]).join(' and '));
  if(a.z){const lo=a.z[0]-a.floor, hi=a.z[1]-a.floor; more.push(`${Math.max(0,lo).toFixed(1)}–${hi.toFixed(1)} m above floor`);}
  if(a.walk!=null) more.push(`${a.walk} m from your path`);
  el.innerHTML=`<div class="num" style="background:${c}">${a.id}</div><div><span class="what">${what}</span><small>${more.join(' · ')}</small>`+
   (L.length?`<small class="tag">${L.map(k=>LABN[k]).join(' · ')}</small>`:'')+`</div>`+
   `<div class="gain">${a.gain!=null?'+'+fmt(a.gain)+' m²<small>expected</small>':''}</div>`;
  el.onclick=()=>pick(a.id,false);
  if(sel===a.id&&a.area!=null){const d=document.createElement('div'); d.className='why';
   d.innerHTML=`<b>${fmt(a.area)} m²</b> in view from here fails the requirement (yellow on the map):`+
    `<br>${fmt(a.new)} m² was never seen from within ${s.range_m} m`+
    (a.closer>0?`; ${fmt(a.closer)} m² was, but from further away than from here`:'')+
    (a.same>0?`; ${fmt(a.same)} m² was already seen from as close, so a second pass counts for about a third`:'')+`.`+
    `<br>${a.types.map(t=>`${t[0]} ${fmt(t[1])}`).join(' · ')} m²`;
   el.appendChild(d);}
  if(sel===a.id){const row=document.createElement('div'); row.className='chips';
   const chip=(txt,on,f)=>{const b=document.createElement('button'); b.textContent=txt; if(on) b.className='on'; b.onclick=e=>{e.stopPropagation(); f();}; row.appendChild(b);};
   for(const [k,txt] of LAB) if(!(a.kind==='added'&&k==='wrong')) chip(txt,L.includes(k),()=>tog(a,k));
   if(a.kind==='added') chip('Remove',false,()=>{st.added=st.added.filter(i=>i.id!==a.id); delete st.labels[a.id]; sel=null; keep(); draw();});
   el.appendChild(row);}
  list.appendChild(el);}
 onView(cur().k);}
function pick(id,fromMap){sel=sel===id?null:id; draw(); if(sel==null) return;
 const a=all().find(i=>i.id===id);
 if(fromMap) $('a'+id).scrollIntoView({behavior:'smooth',block:'nearest'});
 else animTo({cx:a.x,cy:a.y,k:Math.min(cur().k,2.6*D.rpx/Math.min(CW(),CH()))});}
const addb=$('add'); let adding=false;
const setadd=v=>{adding=v; addb.className=v?'on':''; wrap.classList.toggle('adding',v); addb.textContent=v?'Now tap the map where it is (Esc: cancel)':'+ Add a place the list missed';};
addb.onclick=()=>setadd(!adding);
svg.addEventListener('click',e=>{ if(!adding||e.target.closest('#marks')) return;
 const [lx,ly]=local(e), kk=cur().k, x=vb.x+lx*kk, y=vb.y+ly*kk;
 let n=1; while(st.added.some(i=>i.id==='M'+n)) n++;
 const a={id:'M'+n,kind:'added',x:Math.round(x),y:Math.round(y),tx:null,ty:null,gain:null,walk:null,
  w:[+(D.fr.x0+x*D.fr.m).toFixed(2),+(D.fr.y0+(D.H-y)*D.fr.m).toFixed(2)],tw:null};
 st.added.push(a); st.labels[a.id]={xy:a.w,l:['missed']}; setadd(false); sel=a.id; keep(); draw();
 $('a'+a.id).scrollIntoView({behavior:'smooth',block:'nearest'});});
$('save').onclick=()=>{
 const out={scan:s.scan,saved:new Date().toISOString(),s_req_m:s.s_req_m,range_m:s.range_m,
  places:D.items.map(a=>({id:a.id,kind:a.opening?'opening':a.kind,xy:a.w,target:a.tw,gain_m2:a.gain,labels:lab(a)})),
  added:st.added.map(a=>({id:a.id,xy:a.w,labels:lab(a)}))};
 const u=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:'application/json'})), l=document.createElement('a');
 l.href=u; l.download='labels_'+s.scan+'.json'; document.body.appendChild(l); l.click(); l.remove(); setTimeout(()=>URL.revokeObjectURL(u),2000);
 const n=out.places.filter(p=>p.labels.length).length;
 $('saved').textContent=`Saved labels_${s.scan}.json: ${n} of ${out.places.length} places labelled, ${out.added.length} added. Put the file in the scan's result folder.`;};
$('foot').textContent=`A 20 cm patch of surface meets the requirement when at most one of its sixteen ${s.s_req_m*100} cm cells has no point. `+
 `A place is credited with the failing patches it views from within ${s.range_m} m, under ${s.max_incidence_deg}°, with a free line of sight. `+
 `Expected gain: on this sensor about ${Math.round(10*s.p_fix.flat)} in 10 flat patches were fixed once seen that way for the first time or from clearly closer than before; a second pass from the same distance fixed about a third. `+
 (s.things_counted?'':`Building only: walls, floor, ceiling and the edges between them; furniture and clutter (sand-coloured) are left out. The tab "Surface quality" shows them with their numbers. `)+
 (s.closer_not_listed_m2>0?`${fmt(s.closer_not_listed_m2)} m² more could be fixed from the walked floor, in bits too small to list. `:'')+
 `Blue floor: reachable from your path. Orange floor: seen, never entered. Pale blue-green: seen through a window, no floor there. Grey: not observed. Green line: your walk. `+
 `Tap a place, then tap what applies to it; anything except "Missed" takes it off the list. Sensor ${s.sensor_height_m} m above the floor; ${s.floor.reachable_m2} m² reachable, ${s.floor.cut_off_m2} m² seen but cut off.`;

// ---------- tab 2: surface quality ----------
const DEF={tol:QA.tool_tol,count:{},show:{},ov:{things:true,under:false,behind:false,unseen:false,open:false},walk:true};
for(const r of QA.rows){DEF.count[r.key]=r.group==='building'||(r.group==='things'&&!!s.things_counted); DEF.show[r.key]=['wall','floor','other_flat','edge'].includes(r.key);}
const VK='view-'+s.scan; let V=JSON.parse(JSON.stringify(DEF)), tab=0;
try{const o=JSON.parse(localStorage.getItem(VK)||'null'); if(o&&o.count&&o.show&&o.ov){V=Object.assign(V,o); if(o.tab) tab=o.tab;}}catch(e){}
const keepV=()=>{try{localStorage.setItem(VK,JSON.stringify(Object.assign({},V,{tab})))}catch(e){}};
const HID=[['things','rgb(222,206,178)','Furniture and clutter: floor it stands on'],['under','rgb(176,136,80)','… and no floor was seen under it'],
 ['behind','rgb(192,38,211)','Wall behind furniture, no points (taken to continue)'],['unseen','rgb(37,99,235)','Wall not seen, nothing in front'],
 ['open','rgb(13,148,136)','Openings in walls: doors, windows, glass']];
const GRP={building:'Building',things:'Furniture and clutter',outside:'Outside'};
function buildQ(){
 const lvl=QA.lab.map((l,i)=>`<span><i style="background:${QA.lcol[i]}"></i>${l}</span>`).join('');
 let rows='', g0=null;
 for(const r of QA.rows){ if(r.group!==g0){rows+=`<div class="grp">${GRP[r.group]}</div>`; g0=r.group;}
  rows+=`<div class="cr" id="r-${r.key}"><label class="sw" title="Count it in the share above"><input type="checkbox" data-c="${r.key}"><span></span></label>`+
   `<span class="n">${r.name}</span><span class="v" id="m-${r.key}"></span><span class="v" id="p-${r.key}"></span>`+
   `<label class="sw" title="Show it on the map"><input type="checkbox" data-s="${r.key}"><span></span></label><div class="lv" id="b-${r.key}"></div></div>`;}
 const hid=HID.map(h=>`<div class="hr"><i style="background:${h[1]}"></i><span>${h[2]}</span><span class="v">${fmt(QA.hid[h[0]])} m²</span>`+
  `<label class="sw" title="Show it on the map"><input type="checkbox" data-o="${h[0]}"><span></span></label></div>`).join('')+
  `<div class="hr"><i style="background:#1d2433;opacity:.7"></i><span>Your walk</span><span class="v"></span><label class="sw"><input type="checkbox" data-w="1"><span></span></label></div>`;
 const kd=s.deficit_by_kind;
 $('pQ').innerHTML=
 `<div class="box"><div class="big"><b id="qpct"></b><span class="mut" id="qof"></span></div><div class="mut" id="qtool"></div>`+
 `<div class="tolrow"><b>Gaps allowed in a patch</b>`+
 `<span class="seg" id="tol">${[0,1,2,3].map(t=>`<button data-t="${t}" title="${t===QA.tool_tol?'Used by the tool':''}">${t}</button>`).join('')}</span></div>`+
 `<div class="tolx"><span class="cells" id="tolgrid"></span><span class="mut" id="toltxt"></span></div></div>`+
 `<div class="box"><h3>What was scanned, and how well</h3>`+
 `<div class="cr hd"><span>count</span><span>surface</span><span class="v">m²</span><span class="v">meets</span><span>map</span></div>${rows}`+
 `<div class="cr tot"><span></span><span class="n">Counted</span><span class="v" id="m-tot"></span><span class="v" id="p-tot"></span><span></span></div>`+
 `<div class="lgd">finest spacing without gaps: ${lvl}</div>`+
 `<div class="mut" style="margin-top:6px">The map shows the worst patch at each spot among the surfaces switched on. Floor and ceiling lie on top of each other; show one at a time.</div></div>`+
 `<div class="box"><h3>Why the building fails (tool's setting)</h3><div class="mut">`+
 `${fmt(s.deficit_m2)} m² fails: ${fmt(s.deficit_never_seen_close_m2)} m² was never seen from within ${s.range_m} m, ${fmt(s.deficit_seen_close_m2)} m² was and fails all the same.<br>`+
 `Stand closer ${fmt(kd.closer)} · enter a space ${fmt(kd.enter)} · not seen yet ${fmt(kd.explore)} · view blocked ${fmt(kd.blocked)} · out of reach from the floor ${fmt(kd.beyond)} · seen from within ${s.range_m/2} m and still gappy ${fmt(kd.limit)} m². `+
 `The places on "Where to go" deal with the first three.</div></div>`+
 `<div class="box"><h3>Not in the numbers: estimated</h3>${hid}`+
 `<div class="mut" style="margin-top:6px">Wall cells without points between the floor and the top of the wall seen above them. "Behind furniture": something stands within 1 m in front. Openings: a ray passed through.</div></div>`+
 `<div class="note" id="qnote">The switches change this tab only. The places on "Where to go" use the tool's setting: building only, one empty cell allowed.</div>`+
 `<details><summary>How surfaces are sorted</summary>`+
 `<p>Every 20 cm patch gets a plane from its points and its neighbours, its height above the floor and its neighbours' classes.</p>`+
 `<p><b>Walls</b>: flat and vertical (a cupboard front counts as wall). <b>Floor</b>: flat, horizontal, within 0.3 m of the floor. <b>Ceiling</b>: flat, horizontal, 2.2 m or more above the floor. <b>Tilted</b>: flat, neither. <b>Edges and corners</b>: not flat, next to two building surfaces (wall and floor, wall and ceiling, two walls).</p>`+
 `<p><b>Table tops and shelves</b>: flat, horizontal, 0.3–2.2 m above the floor (a window sill or a step lands here too). <b>Clutter</b>: not flat and not an edge: chairs, plants, cables, lamps.</p>`+
 `<p><b>Outside</b>: failing surface that can only be viewed from where no floor was seen, i.e. through a window. Outside surface that happens to pass is not recognised and stays in its class.</p>`+
 `<p>A patch meets ${s.s_req_m*100} cm when at most the chosen number of its sixteen ${s.s_req_m*100} cm cells has no point. The same rule decides the finer and coarser levels in the legend.</p></details>`;
 $('pQ').addEventListener('change',e=>{const t=e.target, d=t.dataset;
  if(d.c) V.count[d.c]=t.checked; else if(d.s) V.show[d.s]=t.checked; else if(d.o) V.ov[d.o]=t.checked; else if(d.w) V.walk=t.checked;
  keepV(); renderQ();});
 $('pQ').addEventListener('click',e=>{const b=e.target.closest('[data-t]'); if(b){V.tol=+b.dataset.t; keepV(); renderQ();}
  if(e.target.id==='qreset'){V.tol=DEF.tol; V.count=Object.assign({},DEF.count); keepV(); renderQ();}});}
function statsQ(){
 const t=V.tol, ks=QA.ks; let A=0,M=0;
 for(const r of QA.rows){const lv=r.lv[t], met=lv.slice(0,ks+1).reduce((a,b)=>a+b,0);
  $('m-'+r.key).textContent=fmt(r.m2); $('p-'+r.key).textContent=r.m2>0?Math.round(100*met/r.m2)+'%':'–';
  $('b-'+r.key).innerHTML=lv.map((v,i)=>`<div style="flex:${v};background:${QA.lcol[i]}" title="${QA.lab[i]}: ${fmt(v)} m²"></div>`).join('');
  $('r-'+r.key).classList.toggle('off',!V.count[r.key]); if(V.count[r.key]){A+=r.m2; M+=met;}}
 $('m-tot').textContent=fmt(A); $('p-tot').textContent=A>0?Math.round(100*M/A)+'%':'–';
 $('qpct').textContent=A>0?(100*M/A).toFixed(1)+' %':'–';
 $('qof').textContent=`of ${fmt(A)} m² counted surface meets ${s.s_req_m*100} cm`;
 const same=V.tol===DEF.tol&&QA.rows.every(r=>V.count[r.key]===DEF.count[r.key]);
 $('qtool').innerHTML=same?`The tool's setting: ${s.things_counted?'everything inside':'building only'}, one empty cell allowed.`:
  `The tool's setting gives ${(+s.met_pct).toFixed(1)} %. <button class="linkbtn" id="qreset">Back to it</button>`;
 for(const b of $('tol').children) b.className=+b.dataset.t===t?'on':'';
 const E=[5,14,2].slice(0,t), cm=s.s_req_m*100, tt=QA.tool_tol;
 $('tolgrid').innerHTML=[...Array(16).keys()].map(i=>`<i class="${E.includes(i)?'e':''}"></i>`).join('');
 $('toltxt').innerHTML=(t===0?`A 20 cm patch meets ${cm} cm only when every one of its sixteen ${cm} cm cells holds a point.`:
  `A 20 cm patch meets ${cm} cm when at most ${t} of its sixteen ${cm} cm cells ${t===1?'has':'have'} no point.`)+
  ` ${t===tt?'This is the tool’s setting.':t<tt?`Stricter than the tool, which allows ${tt}.`:`More lenient than the tool, which allows ${tt}.`}`+
  ` It changes the shares above and below and the colours on the map.`;
 for(const i of $('pQ').querySelectorAll('input')){const d=i.dataset; i.checked=d.c?!!V.count[d.c]:d.s?!!V.show[d.s]:d.o?!!V.ov[d.o]:!!V.walk;}
 $('walk').style.display=(tab===1&&!V.walk)?'none':'';}
let QL=null, qerr=null, qbusy=false;
async function ensureQ(){ if(QL||qerr||qbusy) return renderQ(); qbusy=true;
 try{const bin=Uint8Array.from(atob(QA.b64),c=>c.charCodeAt(0));
  const buf=await new Response(new Blob([bin]).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer();
  const n=QA.w*QA.h, R=QA.rows.length; QL={lay:new Uint16Array(buf,0,R*n),ov:new Uint8Array(buf,R*n*2,n)};}
 catch(e){qerr=e; $('qnote').textContent='This browser cannot draw the quality layers (it lacks DecompressionStream); the numbers are still right.';}
 qbusy=false; renderQ();}
const OVB={things:1,under:2,behind:4,unseen:8,open:16}, OVC={things:[222,206,178],under:[176,136,80],behind:[192,38,211],unseen:[37,99,235],open:[13,148,136]};
function renderQ(){ statsQ(); if(!QL) return;
 const w=QA.w,h=QA.h,n=w*h,R=QA.rows.length,sh=3*V.tol,on=QA.rows.map(r=>!!V.show[r.key]),L=QA.lrgb;
 const under=['things','under'].filter(k=>V.ov[k]), over=['open','unseen','behind'].filter(k=>V.ov[k]);
 const cv=document.createElement('canvas'); cv.width=w; cv.height=h; const ctx=cv.getContext('2d'), im=ctx.createImageData(w,h), p=im.data;
 for(let i=0;i<n;i++){let c=0; for(let r=0;r<R;r++) if(on[r]){const v=(QL.lay[r*n+i]>>sh)&7; if(v>c) c=v;}
  let col=c?L[c-1]:null; const o=QL.ov[i];
  if(o){ if(!c) for(const k of under) if(o&OVB[k]) col=OVC[k]; for(const k of over) if(o&OVB[k]) col=OVC[k]; }
  if(col){const j=i<<2; p[j]=col[0]; p[j+1]=col[1]; p[j+2]=col[2]; p[j+3]=255;}}
 ctx.putImageData(im,0,0); $('imgq').setAttribute('href',cv.toDataURL());}
function setTab(i){tab=i; keepV();
 $('t0').className=i?'':'on'; $('t1').className=i?'on':'';
 $('img0').style.display=i?'none':''; $('qg').style.display=i?'':'none';
 marks.style.display=i?'none':''; hl.style.display=i?'none':'';
 $('pP').style.display=i?'none':''; $('pQ').style.display=i?'':'none';
 $('walk').style.display=(i&&!V.walk)?'none':'';
 $('walk').setAttribute('stroke',i?'#1d2433':'#1b7a3a'); $('walk').setAttribute('opacity',i?.55:.75);
 const pn=$('panel'); if(pn.scrollHeight>pn.clientHeight) pn.scrollTop=0;
 if(adding) setadd(false); if(i) ensureQ();}
$('t0').onclick=()=>setTab(0); $('t1').onclick=()=>setTab(1);
buildQ(); draw(); reflow(); setTab(tab);
</script></body></html>
"""
