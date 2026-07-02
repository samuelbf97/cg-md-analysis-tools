# Protein–RNA Contact Analysis

Example usage:

```bash
python protein_rna_contact_analysis.py result.lammpstrj \
    --protein-length 305 \
    --rna-length 250 \
    --n-proteins 60 \
    --cutoff 8.0 \
    --skip-frames 0 \
    --rna-count-mode binary \
    --rna-normalization-mode all_frames \
    --output-dir contact_analysis_output
```

## Output

The script generates:

- `protein_protein_contacts_raw.txt`
- `protein_protein_contacts_normalized.txt`
- `protein_rna_contacts_raw.txt`
- `protein_rna_contacts_normalized.txt`
- `protein_protein_contact_map.png`
- `protein_rna_contact_profile.png`

## Description

The script computes:

- Residue-resolved protein–protein contact maps.
- Residue-resolved protein–RNA contact frequencies.
- Normalized contact matrices.
- Publication-quality figures.

Protein and RNA molecules are identified automatically according to the number of beads per molecule.
