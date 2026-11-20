"""scanq: scan-quality assessment for mobile LiDAR scans.

    io.py      sweeps from the bag, each point with its own pose (motion-corrected)
    store.py   PatchStore: incremental, mergeable per-patch statistics
    quality.py achieved point spacing per surface patch, against a requirement
    shell.py   building or thing: surface classes, estimated hidden surface
    grid.py    dense occupancy labels from the octree, floor map, line of sight
    plan.py    standpoints that would fix each deficient patch -> ranked actions
    assess.py  ties it together: the numbers and the list of places
    report.py  JSON and the page

See README.md.
"""
