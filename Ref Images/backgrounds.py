"""Load metre-frame curb rasters for trajectory overlays.

    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path("Ref Images").resolve()))
    from backgrounds import load_background
    img, extent = load_background("roundabout")
    ax.imshow(img, extent=extent, origin="upper", zorder=0)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_background(site: str):
    """Return ``(rgba_array, [xmin, xmax, ymin, ymax])`` for ``freeway``, ``tgsim``, or ``roundabout``."""
    from matplotlib.image import imread

    meta = json.loads((HERE / "extents.json").read_text(encoding="utf-8"))
    if site not in meta:
        raise KeyError(f"Unknown site {site!r}; have {sorted(meta)}")
    info = meta[site]
    img = imread(HERE / info["file"])
    extent = [float(x) for x in info["extent"]]
    return np.asarray(img), extent
