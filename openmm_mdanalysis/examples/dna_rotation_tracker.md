# DNA Rotation Tracker

Example usage of `dna_rotation_tracker.py`.

This script estimates azimuthal DNA displacement in nucleosome-like
coarse-grained simulations using a protein-anchored reference frame.

## Example command

```bash
python openmm_mdanalysis/dna_rotation_tracker.py \
    --top nucleosome_cg.pdb \
    --traj md_position.dcd \
    --chain-a-sel "chainID A" \
    --chain-b-sel "chainID B" \
    --bead-name DP \
    --center-atom-id 836 \
    --half-block-bp 5 \
    --protein-core-sel "name CA and not (chainID A or chainID B)" \
    --time-unit ns \
    --out-prefix rot \
    --output-dir dna_rotation_output
```

## Output

The script generates:

- `rot_delta_bp_paired.txt`: estimated base-pair displacement over time
- `rot_meta.txt`: metadata describing the pairing, calibration, and analysis setup

## Notes

The base-pair displacement is estimated from azimuthal rotation around a
protein-anchored reference frame. It is most useful for relative comparisons
between related simulations.
