# Sliding Base-Pair Tracker

Example usage of `sliding_bp_tracker.py`.

This script tracks the closest DNA phosphate bead to a protein anchor and maps
it onto a strand-invariant base-pair coordinate.

## Example command

```bash
python openmm_mdanalysis/sliding_bp_tracker.py \
    --top nucleosome_cg.pdb \
    --traj md_position.dcd \
    --anchor-serial 772,774 \
    --chainA A \
    --chainB B \
    --sel-dnaP "(name P or name DP)" \
    --n-bp 197 \
    --stride 1 \
    --plot \
    --out-ts sliding_bp.csv \
    --out-path sliding_bp.txt \
    --fig sliding_bp.png
```

## Output

The script generates:

- `sliding_bp.csv`: time, base-pair index, chain, residue, and closest distance
- `sliding_bp.txt`: two-column time series
- `sliding_bp.png`: optional trajectory and occupancy plot

## Notes

Both DNA strands are mapped onto a single 1..N coordinate so that movement along
either strand is represented consistently.
