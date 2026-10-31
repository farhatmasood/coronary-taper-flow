# Data

Both files are small, plain-text inputs to the model.

## `coronary_tree_geometry.csv`

The seventeen-segment reference coronary tree. Lengths and lumen radii come
from the one-dimensional model of Duanmu et al. (2019, *Front. Physiol.*
10:853, doi:10.3389/fphys.2019.00853), an open-access article published
under the Creative Commons Attribution (CC BY) licence. The branching follows
the clinical (AHA/SYNTAX) anatomy. The tree is right-dominant: the posterior
descending artery arises from the RCA.

| Column | Unit | Meaning |
|---|---|---|
| `segment_id` | | Short identifier used throughout the code (`LMCA`, `LAD1`, ...) |
| `segment_name` | | Name in the source model |
| `clinical_name` | | Clinical segment name |
| `side` | | `L` or `R` coronary system |
| `parent_id` | | Upstream segment; `AORTA_L` / `AORTA_R` mark the two ostia |
| `length_mm` | mm | Centreline length |
| `radius_prox_mm` | mm | Lumen radius at the proximal end |
| `radius_dist_mm` | mm | Lumen radius at the distal end |
| `is_terminal` | 0/1 | 1 if the segment has no named daughters |
| `geometry_source` | | Provenance of the row |

To run the model on another tree, write a file with the same columns and
pass it to `geometry.CoronaryTree.from_csv`, then to
`coronary.model.build(tree=...)`. The loader rejects a tree whose declared
terminals have children, whose non-terminals have none, or whose parent
references do not resolve. The outflow law assumes radii in millimetres and
works best when the taper of each segment is measured rather than set
constant: a segment reported without taper sheds no flow along its length.

## `reference_measurements.csv`

The published human measurements that the model is compared against
(`coronary.experiments.validation_table`). Each row gives a mean and standard
deviation, or a band (`low`, `high`), with the cohort, the measurement method,
the source and its DOI. The `bibkey` column is a short citation key.
