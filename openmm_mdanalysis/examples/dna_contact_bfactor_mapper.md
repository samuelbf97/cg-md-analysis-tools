# DNA Contact B-Factor Mapper

Example usage of `dna_contact_bfactor_mapper.py`.

This script computes the probability that each DNA bead is contacted by one or
more protein selections and writes PDB files with the probability encoded in the
B-factor field.

## Example command

```bash
python openmm_mdanalysis/dna_contact_bfactor_mapper.py \
    --top nucleosome_cg.pdb \
    --traj md_position.dcd \
    --dna-chains A,B \
    --dna-beads P,S,B,DP,DS \
    --target H14:"chainid K" \
    --target H3_tail_C:"chainid C and resid 1-36" \
    --target H3_tail_G:"chainid G and resid 1-36" \
    --cutoff 8.0 \
    --combined H3_tail_C H3_tail_G \
    --output-dir dna_contact_bfactor_output
```

## Output

The script generates:

- `dna_contact_probability_<target>.csv`
- `dna_contact_probability_<target>_bfactor.pdb`
- optional combined B-factor maps using maximum and clipped-sum probabilities

## Notes

The B-factor PDB files can be opened in VMD, PyMOL, or ChimeraX to visualize
DNA regions contacted by each protein selection.
