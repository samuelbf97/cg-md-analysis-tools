# Density Profile Averaging

Example usage of `density_profile_average.py` for averaging LAMMPS density profiles generated with `fix ave/chunk`.

## Example command

```bash
python lammps/density_profile_average.py density.profile \
    --start-frame 100 \
    -o density_average.txt
```

Alternatively, averaging can start from a specific timestep:

```bash
python lammps/density_profile_average.py density.profile \
    --start-timestep 5000000 \
    -o density_average.txt
```

## Input

The input file should be a LAMMPS density profile generated with `fix ave/chunk`, containing blocks with:

```text
timestep number_of_bins total_count
bin coordinate count density
```

## Output

The script writes a two-column text file:

```text
distance density_avg
```

Example output file:

```text
density_average.txt
```

## Description

The script averages density profiles over multiple frames and allows the user to discard initial frames or start the averaging from a selected timestep. This is useful for analyzing one-dimensional density profiles of condensed and dilute phases in slab or direct-coexistence simulations.
