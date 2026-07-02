# Protein Contact Maps

Example usage of `protein_contact_maps.py`.

This script computes residue-resolved contact maps between two protein species
and optionally computes DNA contact profiles.

## Example command

```bash
python openmm_mdanalysis/protein_contact_maps.py \
    --top proof.pdb \
    --traj todo_position.dcd \
    --chain-a A \
    --chain-b B \
    --n-prot-a 100 \
    --n-prot-b 100 \
    --res-per-a 305 \
    --res-per-b 352 \
    --start-frame 100 \
    --cutoff 6.5 \
    --wrap \
    --output-dir protein_contact_maps_output
```

## Optional DNA contact analysis

```bash
python openmm_mdanalysis/protein_contact_maps.py \
    --top proof.pdb \
    --traj todo_position.dcd \
    --chain-a A \
    --chain-b B \
    --n-prot-a 100 \
    --n-prot-b 100 \
    --res-per-a 305 \
    --res-per-b 352 \
    --start-frame 100 \
    --cutoff 6.5 \
    --dna-resname-prefix D \
    --n-dna 20 \
    --output-dir protein_contact_maps_output
```

## Output

The script generates:

- raw and normalized A-A, A-B, and B-B contact maps
- heatmap PNG files
- global contact frequency summary
- optional protein-DNA contact profiles
