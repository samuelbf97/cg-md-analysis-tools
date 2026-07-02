#!/usr/bin/env python3
"""
protein_rna_contact_analysis.py

Compute protein-protein contact maps and protein-RNA contact profiles from
coarse-grained molecular dynamics trajectories written in LAMMPS dump format.

Features
--------
- Reads LAMMPS .lammpstrj trajectory files.
- Automatically identifies protein and RNA molecules by bead count.
- Computes residue-residue protein-protein contact maps.
- Computes residue-resolved protein-RNA contact frequencies.
- Supports periodic boundary conditions.
- Writes raw and normalized contact data.
- Generates contact-map and contact-profile figures.

Author
------
Samuel Blázquez Fernández
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Argument parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute protein-protein contact maps and protein-RNA contact "
            "profiles from coarse-grained LAMMPS trajectories."
        )
    )

    parser.add_argument(
        "dump_file",
        help="Input LAMMPS trajectory file in .lammpstrj format."
    )

    parser.add_argument(
        "--protein-length",
        type=int,
        required=True,
        help="Number of beads/residues per protein molecule."
    )

    parser.add_argument(
        "--rna-length",
        type=int,
        required=True,
        help="Number of beads/nucleotides per RNA molecule."
    )

    parser.add_argument(
        "--n-proteins",
        type=int,
        required=True,
        help="Expected number of protein molecules in each frame."
    )

    parser.add_argument(
        "--cutoff",
        type=float,
        default=8.0,
        help="Contact cutoff distance in the same units as the trajectory."
    )

    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Number of initial frames to skip."
    )

    parser.add_argument(
        "--rna-count-mode",
        choices=["binary", "pair"],
        default="binary",
        help=(
            "RNA contact counting mode. 'binary': each residue counts once per "
            "frame/protein if it contacts any RNA bead. 'pair': count all "
            "residue-RNA bead pairs."
        )
    )

    parser.add_argument(
        "--rna-normalization-mode",
        choices=["all_frames", "frames_with_rna"],
        default="all_frames",
        help=(
            "Normalization mode for RNA contacts. 'all_frames': normalize by "
            "all analyzed frames. 'frames_with_rna': normalize only by frames "
            "containing RNA."
        )
    )

    parser.add_argument(
        "--output-dir",
        default="contact_analysis_output",
        help="Directory where output files will be written."
    )

    return parser.parse_args()


# ============================================================
# LAMMPS trajectory reader
# ============================================================

def read_lammpstrj(filename):
    """
    Read a LAMMPS dump trajectory frame by frame.

    Yields
    ------
    step : int
        Timestep.
    box_lengths : np.ndarray
        Box lengths in x, y, z.
    data : dict
        Dictionary containing trajectory columns and a 'coords' array.
    """
    with open(filename, "r", encoding="utf-8", errors="replace") as file:
        while True:
            line = file.readline()

            if not line:
                break

            if not line.startswith("ITEM: TIMESTEP"):
                raise ValueError("Expected 'ITEM: TIMESTEP'.")

            step = int(file.readline().strip())

            line = file.readline().strip()
            if not line.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError("Expected 'ITEM: NUMBER OF ATOMS'.")

            natoms = int(file.readline().strip())

            line = file.readline().strip()
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Expected 'ITEM: BOX BOUNDS'.")

            bounds = []
            for _ in range(3):
                parts = file.readline().split()
                lower, upper = float(parts[0]), float(parts[1])
                bounds.append((lower, upper))

            bounds = np.array(bounds, dtype=float)
            box_lengths = bounds[:, 1] - bounds[:, 0]

            line = file.readline().strip()
            if not line.startswith("ITEM: ATOMS"):
                raise ValueError("Expected 'ITEM: ATOMS'.")

            columns = line.split()[2:]
            data = {column: np.empty(natoms, dtype=float) for column in columns}

            for atom_idx in range(natoms):
                parts = file.readline().split()
                for column, value in zip(columns, parts):
                    data[column][atom_idx] = float(value)

            for key in ["id", "mol", "type"]:
                if key in data:
                    data[key] = data[key].astype(int)

            if all(key in data for key in ["xu", "yu", "zu"]):
                coords = np.column_stack([data["xu"], data["yu"], data["zu"]])
            elif all(key in data for key in ["x", "y", "z"]):
                coords = np.column_stack([data["x"], data["y"], data["z"]])
            elif all(key in data for key in ["xs", "ys", "zs"]):
                coords = (
                    np.column_stack([data["xs"], data["ys"], data["zs"]])
                    * box_lengths
                )
            else:
                raise ValueError(
                    "Coordinates not found. Expected xu/yu/zu, x/y/z, or xs/ys/zs."
                )

            data["coords"] = coords

            yield step, box_lengths, data


# ============================================================
# Helper functions
# ============================================================

def get_molecule_indices(mol_ids):
    """Return atom indices grouped by molecule ID."""
    molecule_to_indices = {}

    for molecule_id in np.unique(mol_ids):
        molecule_to_indices[molecule_id] = np.where(mol_ids == molecule_id)[0]

    return molecule_to_indices


def classify_molecules(data, protein_length, rna_length):
    """
    Classify molecules as proteins or RNA molecules using bead count.
    """
    molecule_to_indices = get_molecule_indices(data["mol"])

    protein_molecules = []
    rna_molecules = []

    for molecule_id, indices in molecule_to_indices.items():
        number_of_beads = len(indices)

        if number_of_beads == protein_length:
            protein_molecules.append(molecule_id)
        elif number_of_beads == rna_length:
            rna_molecules.append(molecule_id)

    protein_molecules.sort()
    rna_molecules.sort()

    return protein_molecules, rna_molecules


def get_ordered_coordinates_for_molecule(data, molecule_id):
    """
    Return coordinates of a molecule ordered by atom ID.

    This defines the bead/residue index within the molecule.
    """
    indices = np.where(data["mol"] == molecule_id)[0]
    order = np.argsort(data["id"][indices])

    return data["coords"][indices][order]


def compute_contact_matrix_pbc(coords_a, coords_b, cutoff, box_lengths):
    """
    Compute a binary contact matrix between two molecules using PBC.
    """
    displacement = coords_a[:, None, :] - coords_b[None, :, :]
    displacement -= box_lengths[None, None, :] * np.round(
        displacement / box_lengths[None, None, :]
    )

    distance_squared = np.sum(displacement * displacement, axis=2)

    return (distance_squared < cutoff * cutoff).astype(np.int64)


def compute_protein_rna_contacts_pbc(
    protein_coords,
    rna_coords,
    cutoff,
    box_lengths,
    mode="binary",
):
    """
    Compute protein-RNA contacts for one protein.

    Parameters
    ----------
    protein_coords : np.ndarray
        Coordinates of one protein, shape (n_residues, 3).
    rna_coords : np.ndarray or None
        Coordinates of all RNA beads, shape (n_rna_beads, 3).
    cutoff : float
        Contact cutoff.
    box_lengths : np.ndarray
        Simulation box lengths.
    mode : str
        'binary' or 'pair'.

    Returns
    -------
    np.ndarray
        Residue-resolved RNA contact counts.
    """
    if rna_coords is None or len(rna_coords) == 0:
        return np.zeros(protein_coords.shape[0], dtype=np.int64)

    displacement = protein_coords[:, None, :] - rna_coords[None, :, :]
    displacement -= box_lengths[None, None, :] * np.round(
        displacement / box_lengths[None, None, :]
    )

    distance_squared = np.sum(displacement * displacement, axis=2)
    contact_mask = distance_squared < cutoff * cutoff

    if mode == "binary":
        return np.any(contact_mask, axis=1).astype(np.int64)

    if mode == "pair":
        return np.sum(contact_mask, axis=1).astype(np.int64)

    raise ValueError("mode must be either 'binary' or 'pair'.")


def make_output_directory(path):
    """Create output directory if it does not already exist."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ============================================================
# Main analysis
# ============================================================

def run_contact_analysis(args):
    output_dir = make_output_directory(args.output_dir)

    protein_protein_upper = np.zeros(
        (args.protein_length, args.protein_length),
        dtype=np.int64,
    )

    protein_rna_contacts = np.zeros(args.protein_length, dtype=np.int64)

    n_frames_total = 0
    n_frames_analyzed = 0
    n_frames_with_rna = 0
    n_protein_pairs_sampled = 0

    for frame_idx, (step, box_lengths, data) in enumerate(
        read_lammpstrj(args.dump_file)
    ):
        n_frames_total += 1

        if frame_idx < args.skip_frames:
            continue

        if "mol" not in data or "id" not in data:
            raise ValueError(
                "The LAMMPS dump must contain at least 'id' and 'mol' columns."
            )

        protein_molecules, rna_molecules = classify_molecules(
            data,
            args.protein_length,
            args.rna_length,
        )

        if len(protein_molecules) != args.n_proteins:
            raise ValueError(
                f"Frame {frame_idx}: found {len(protein_molecules)} proteins "
                f"with {args.protein_length} beads, but expected "
                f"{args.n_proteins}."
            )

        protein_coordinates = {}

        for molecule_id in protein_molecules:
            coordinates = get_ordered_coordinates_for_molecule(data, molecule_id)

            if coordinates.shape[0] != args.protein_length:
                raise ValueError(
                    f"Molecule {molecule_id} does not have "
                    f"{args.protein_length} beads."
                )

            protein_coordinates[molecule_id] = coordinates

        if len(rna_molecules) > 0:
            rna_coordinates_all = []

            for molecule_id in rna_molecules:
                coordinates = get_ordered_coordinates_for_molecule(data, molecule_id)

                if coordinates.shape[0] != args.rna_length:
                    raise ValueError(
                        f"RNA molecule {molecule_id} does not have "
                        f"{args.rna_length} beads."
                    )

                rna_coordinates_all.append(coordinates)

            rna_coordinates_all = np.vstack(rna_coordinates_all)
            n_frames_with_rna += 1

        else:
            rna_coordinates_all = None

        # Protein-protein contacts
        protein_ids = protein_molecules

        for i, molecule_i in enumerate(protein_ids):
            coordinates_i = protein_coordinates[molecule_i]

            for molecule_j in protein_ids[i + 1:]:
                coordinates_j = protein_coordinates[molecule_j]

                contact_matrix = compute_contact_matrix_pbc(
                    coordinates_i,
                    coordinates_j,
                    args.cutoff,
                    box_lengths,
                )

                protein_protein_upper += contact_matrix
                n_protein_pairs_sampled += 1

        # Protein-RNA contacts
        if rna_coordinates_all is not None:
            for molecule_id in protein_ids:
                residue_contacts = compute_protein_rna_contacts_pbc(
                    protein_coordinates[molecule_id],
                    rna_coordinates_all,
                    args.cutoff,
                    box_lengths,
                    mode=args.rna_count_mode,
                )

                protein_rna_contacts += residue_contacts

        n_frames_analyzed += 1

    if n_frames_analyzed == 0:
        raise RuntimeError("No frames were analyzed. Check --skip-frames.")

    protein_protein_raw = protein_protein_upper + protein_protein_upper.T

    pairs_per_frame = args.n_proteins * (args.n_proteins - 1) // 2
    protein_protein_norm = protein_protein_raw / (
        n_frames_analyzed * pairs_per_frame
    )

    if args.rna_normalization_mode == "all_frames":
        rna_denominator = n_frames_analyzed * args.n_proteins
    else:
        rna_denominator = n_frames_with_rna * args.n_proteins

    if rna_denominator > 0:
        protein_rna_norm = protein_rna_contacts / rna_denominator
    else:
        protein_rna_norm = np.zeros_like(protein_rna_contacts, dtype=float)

    save_results(
        output_dir,
        protein_protein_raw,
        protein_protein_norm,
        protein_rna_contacts,
        protein_rna_norm,
    )

    plot_protein_protein_contact_map(
        protein_protein_norm,
        output_dir / "protein_protein_contact_map.png",
    )

    plot_protein_rna_contact_profile(
        protein_rna_norm,
        output_dir / "protein_rna_contact_profile.png",
    )

    print_summary(
        args,
        n_frames_total,
        n_frames_analyzed,
        n_frames_with_rna,
        pairs_per_frame,
        n_protein_pairs_sampled,
        output_dir,
    )


# ============================================================
# Output functions
# ============================================================

def save_results(
    output_dir,
    protein_protein_raw,
    protein_protein_norm,
    protein_rna_raw,
    protein_rna_norm,
):
    """Save raw and normalized contact data."""
    np.savetxt(
        output_dir / "protein_protein_contacts_raw.txt",
        protein_protein_raw,
        fmt="%d",
    )

    np.savetxt(
        output_dir / "protein_protein_contacts_normalized.txt",
        protein_protein_norm,
        fmt="%.8f",
    )

    np.savetxt(
        output_dir / "protein_rna_contacts_raw.txt",
        protein_rna_raw,
        fmt="%d",
    )

    np.savetxt(
        output_dir / "protein_rna_contacts_normalized.txt",
        protein_rna_norm,
        fmt="%.8f",
    )


def plot_protein_protein_contact_map(contact_map, output_file):
    """Save a protein-protein contact map as a PNG image."""
    plt.figure(figsize=(8, 6))

    image = plt.imshow(
        contact_map,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
    )

    plt.colorbar(image, label="Normalized contact frequency")
    plt.xlabel("Residue index")
    plt.ylabel("Residue index")
    plt.title("Protein-protein contact map")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_protein_rna_contact_profile(contact_profile, output_file):
    """Save a residue-resolved protein-RNA contact profile."""
    residues = np.arange(1, len(contact_profile) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(residues, contact_profile, linewidth=1.5)
    plt.xlabel("Residue index")
    plt.ylabel("Normalized RNA contact frequency")
    plt.title("Protein-RNA contact profile")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def print_summary(
    args,
    n_frames_total,
    n_frames_analyzed,
    n_frames_with_rna,
    pairs_per_frame,
    n_protein_pairs_sampled,
    output_dir,
):
    """Print analysis summary."""
    print("========================================")
    print("Contact analysis completed successfully")
    print("========================================")
    print(f"Input trajectory:                       {args.dump_file}")
    print(f"Total frames in trajectory:             {n_frames_total}")
    print(f"Initial frames skipped:                 {args.skip_frames}")
    print(f"Analyzed frames:                        {n_frames_analyzed}")
    print(f"Frames containing RNA:                  {n_frames_with_rna}")
    print(f"Proteins per frame:                     {args.n_proteins}")
    print(f"Protein pairs per frame:                {pairs_per_frame}")
    print(f"Total protein-protein pairs sampled:    {n_protein_pairs_sampled}")
    print(f"Contact cutoff:                         {args.cutoff}")
    print(f"RNA count mode:                         {args.rna_count_mode}")
    print(f"RNA normalization mode:                 {args.rna_normalization_mode}")
    print(f"Output directory:                       {output_dir}")


# ============================================================
# Entry point
# ============================================================

def main():
    args = parse_args()
    run_contact_analysis(args)


if __name__ == "__main__":
    main()
