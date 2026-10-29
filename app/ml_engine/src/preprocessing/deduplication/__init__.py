"""Duplicate detection and cleaning for 2D brain MRI classification datasets.

Expected layout: ``<data_dir>/<Train|Test>/<label>/<image>``.

Pipeline:

1. ``fingerprint``: decoded-pixel SHA-256, pHash and 32x32 thumbnails of the full and the
   background-cropped image (cropping aligns copies with different margins; the full
   image stays stable when compression noise shifts the crop box).
2. ``matching``: pHash candidates (Hamming distance, direct or mirrored) confirmed by
   thumbnail correlation, which ignores brightness/contrast changes. Union-find builds
   duplicate clusters (same slice) and wider related groups (neighbouring slices).
3. ``selection``: one image kept per duplicate cluster; augmentations and minority
   labels dropped.
4. ``report``: ``manifest.csv`` (with ``group_id`` for grouped splitting),
   ``review_pairs.csv`` and ``summary.json`` in a timestamped folder.

Raw images are never modified. Run from ``app/ml_engine``::

    python -m src.preprocessing.deduplication --data-dir src/data
"""

from src.preprocessing.deduplication.pipeline import run

__all__ = ["run"]
