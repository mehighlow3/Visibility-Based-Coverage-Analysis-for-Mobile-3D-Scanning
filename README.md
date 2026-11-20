# Scan check for handheld LiDAR scans

From a handheld LiDAR scan and its trajectory, the tool finds which surfaces were sampled finely enough and where to stand to fix the ones that were not. The result is one HTML page that works offline on a PC or a phone.

The requirement is a bound on gaps: **no gap between points larger than 5 cm.** Every 20 cm patch of surface is checked against it.

## The pages

| Scan | Page | Result |
|---|---|---|
| Institute floor, 24-min walk | [`out/institutGesamt/scanq/scan_check.html`](out/institutGesamt/scanq/scan_check.html) | 89.0 % of 3,275 m² of building surface meets 5 cm; 27 places to go |
| Office, 3-min walk | [`out/20260827172211/scanq/scan_check.html`](out/20260827172211/scanq/scan_check.html) | 75.9 % of 509 m²; 11 places |

GitHub shows these files as source code. To view one, download it (the "Download raw file" button) and open it in a current browser: Chrome, Edge, Firefox or Safari. The page loads nothing from the internet. `quality.json` next to each page holds the same numbers for a program.

**Moving around.** The map works like a web map:
- **Mouse:** drag to move, scroll to zoom at the cursor, double-click to zoom in (Shift+double-click to zoom out).
- **Keyboard, with the pointer over the map:** the arrow keys move, + and − zoom, 0 shows the whole map, Esc closes a selected place.
- **Phone:** one finger moves, two fingers zoom.

**Tab "Where to go": what to scan again**
- **Top:** the share of building surface that meets 5 cm, the area the list is expected to fix, and the number of places. The bar below splits the surface by state: meets the requirement, stand closer, enter a space, not seen yet, view blocked, out of reach, and seen close but still with gaps.
- **Map:** the floor plan from above.
  - Blue floor can be reached from the walk; orange floor was seen but never entered; pale blue-green was seen through a window; grey was not observed.
  - Sand is furniture; the green line is the walk.
  - The numbered circles are the places, coloured by what to do there.
- **List:** each place with what to do, which surfaces, how high above the floor, how far from the walk, and the expected m². Clicking a place moves the map to it and marks in yellow the failing patches it would fix.
- **Labels:** the buttons under a selected place, "Add a place" and "Save labels" are for recording on site what was really there. They write `labels_<scan>.json`. Labels stay in your browser until saved.

**Tab "Surface quality": how well everything was scanned**
- **Map:** each patch is coloured by the finest spacing it reaches without gaps: 2.5, 5, 10 or 20 cm, or coarser.
- **Gaps allowed in a patch (0–3):** how many of a patch's sixteen 5 cm cells may be empty. The tool uses 1. The text under the switch says what the chosen value means.
- **One row per kind of surface:**
  - building: walls, floor, ceiling, tilted surfaces, edges and corners;
  - furniture and clutter: table tops and shelves, clutter;
  - outside: failing surface that could only be seen through windows.

  Each row shows its area, the share that meets 5 cm, and a bar of the spacing levels. The left switch counts the row in the headline share; the right switch draws it on the map.
- **Estimated hidden surface (not in the numbers):** where furniture stands, floor never seen under it, wall behind furniture, wall not seen with nothing in front, and openings (doors, windows, glass). Each has a map switch.
- **Saved settings:** the switches change only this tab. "Back to it" restores the tool's setting, and the page remembers your choices.

## How the numbers are computed

1. **Points.** Each LiDAR point gets its own sensor pose, interpolated from the trajectory at the point's timestamp. Points go into an occupancy octree at 10 cm, with ray casting, and into a store of 20 cm surface patches. The store keeps each patch's points in 5 cm cells, the plane, and the distance and direction it was seen from.
2. **Gaps.** A patch meets *s* = 5 cm when at most one of its sixteen 5 cm cells holds no point. The same rule at 2.5, 10 and 20 cm gives the finest level a patch reaches.
3. **Building or thing.** Each patch is sorted by simple rules: whether it is flat, its orientation, its height above the floor and its neighbours.
   - Flat and vertical is wall; flat and horizontal near the floor is floor; flat and horizontal at 2.2 m or more is ceiling.
   - Not flat between two building surfaces is an edge.
   - Flat and horizontal at 0.3–2.2 m is a table top or shelf. Anything else not flat is clutter.

   Only the building is graded by default; `--things` counts furniture and clutter too.
4. **Where to stand.** At walking pace this sensor reaches 5 cm up to about 4 m (2.5 cm: 1.5 m; 10 cm: 10 m), measured on the two scans. For every failing patch the tool finds the floor positions within that range, under 60° incidence and with a free line of sight. Positions on floor reached by the walk mean *stand closer*; on floor that was seen but never entered, *enter a space*; where nothing was observed, *go and look*. Failing surface that can only be viewed from where no floor was seen is outside a window, and is left out.
5. **Expected gain.** Each place is credited with the failing area in view, weighted by how often such a view fixed a patch on these scans. A first close view fixed about 9 in 10 flat patches, edges 0.85, table tops 0.8 and clutter 0.4. A second pass from the same distance counts for about a third. Places are listed in order of expected gain.
6. **Hidden surface.** In each wall column, from the floor to the top of the wall, the wall cells without points are checked:
   - a ray passed through: an opening;
   - something stands within 1 m in front: behind furniture;
   - otherwise: not seen.

## Run it

Each scan is a folder holding `LIDAR_*.bag` (ROS1 bags, topic `/vanjee_722z`; Raven / Vanjee WLR-722Z) and `recalculate_trj.txt` (the trajectory from RayStudio "Data Processing", TUM format). The raw scans are not published.

```powershell
python -m venv .venv; .venv\Scripts\pip install -r requirements.txt     # Python 3.12
$env:P1_SCANS = "<folder with the scan folders>"
.venv\Scripts\python assess.py --scan institutGesamt             # 5 cm requirement, building only
.venv\Scripts\python assess.py --scan institutGesamt --s 0.10    # coarser requirement
.venv\Scripts\python assess.py --scan institutGesamt --things    # count furniture and clutter too
```

**Run times.** The first run builds the octree and the patch store, which takes a few minutes. They are kept in `out/<scan>/scanq/full/`, and later runs take about a minute.

**Output.** Each run writes `out/<scan>/scanq/scan_check.html` and `quality.json`.

| File | What it does |
|---|---|
| `assess.py` | command line |
| `scanq/io.py` | sweeps from the bag, each point with its own pose |
| `scanq/store.py` | per-patch statistics (incremental, mergeable) |
| `scanq/build.py`, `session.py` | build the octree and the store once, keep them on disk |
| `scanq/quality.py` | point spacing per patch against the requirement |
| `scanq/shell.py` | building or thing; estimated hidden surface |
| `scanq/grid.py` | occupancy, floor map, line of sight |
| `scanq/plan.py` | standpoints that would fix each failing patch, grouped into places |
| `scanq/assess.py` | the numbers and the ranked list |
| `scanq/report.py` | `quality.json` and the page |

## Limits

- **One building only.** The rules that separate building from furniture, and outside from not visited, were checked on one building. A cupboard front counts as wall.
- **Weights from the same scans.** The expected-gain weights come from the same two scans; a second building is needed.
- **Validation so far.** Deleting ten branches of the walk one at a time, the tool proposed a place on the deleted path in nine cases. Nobody has yet scanned by following the list.

Earlier versions of this repository (October 2026, `coverage.py`) are in the git history.
