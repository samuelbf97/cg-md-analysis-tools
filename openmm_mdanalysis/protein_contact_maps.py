#!/usr/bin/env python3
"""
protein_contact_maps.py

Compute residue-resolved contact maps between two protein species and optional
protein-DNA contact profiles from MDAnalysis-compatible trajectories.

This script was designed for coarse-grained biomolecular condensate simulations
containing many copies of two protein types and, optionally, coarse-grained DNA.

Features
--------
- Computes A-A, B-B, and A-B residue contact maps.
- Computes global contact frequencies between protein types.
- Optionally computes DNA contact profiles for each protein species.
- Writes raw and normalized contact matrices.
- Generates heatmaps and summary bar plots.

Author
------
Samuel Blázquez Fernández
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.transformations import wrap


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute residue-resolved protein contact maps and optional "
            "protein-DNA contact profiles from MD trajectories."
        )
    )

    parser.add_argument("--top", required=True, help="Topology file.")
    parser.add_argument("--traj", required=True, help="Trajectory file.")
    parser.add_argument("--chain-a", default="A", help="Chain ID for protein species A.")
    parser.add_argument("--chain-b", default="B", help="Chain ID for protein species B.")
    parser.add_argument("--n-prot-a", type=int, required=True, help="Number of A proteins.")
    parser.add_argument("--n-prot-b", type=int, required=True, help="Number of B proteins.")
    parser.add_argument("--res-per-a", type=int, required=True, help="Residues per A protein.")
    parser.add_argument("--res-per-b", type=int, required=True, help="Residues per B protein.")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame to analyze.")
    parser.add_argument("--cutoff", type=float, default=6.5, help="Contact cutoff in Angstrom.")
    parser.add_argument("--output-dir", default="protein_contact_maps_output", help="Output directory.")
    parser.add_argument("--wrap", action="store_true", help="Wrap atoms into the primary unit cell.")

    parser.add_argument(
        "--dna-resname-prefix",
        default=None,
        help=(
            "Optional DNA residue-name prefix. If provided, DNA contact profiles "
            "will be computed for residues whose resname starts with this prefix."
        )
    )

    parser.add_argument(
        "--n-dna",
        type=int,
        default=None,
        help="Number of DNA molecules used for normalizing DNA contact profiles."
    )

    return parser.parse_args()


def split_residues_into_copies(residues, n_copies, residues_per_copy, label):
    """Split a ResidueGroup into a list of atom-index arrays, one per molecule."""
    expected = n_copies * residues_per_copy

    if len(residues) != expected:
        raise ValueError(
            f"Unexpected number of residues for species {label}: found "
            f"{len(residues)}, expected {expected}."
        )

    return [
        residues[i * residues_per_copy:(i + 1) * residues_per_copy].atoms.indices
        for i in range(n_copies)
    ]


def pairwise_contact_matrix(coords_a, coords_b, cutoff):
    """Compute a binary contact matrix between two residue/bead coordinate arrays."""
    displacement = coords_a[:, None, :] - coords_b[None, :, :]
    distance_squared = np.sum(displacement * displacement, axis=-1)
    return (distance_squared < cutoff * cutoff).astype(np.int64)


def update_contact_maps(
    positions,
    indices_a,
    indices_b,
    contact_aa,
    contact_ab,
    contact_bb,
    cutoff,
):
    """Update A-A, A-B, and B-B contact maps for one frame."""
    total_aa = 0
    total_ab = 0
    total_bb = 0

    for i, idx_a in enumerate(indices_a):
        coords_a = positions[idx_a]

        for idx_b in indices_b:
            coords_b = positions[idx_b]
            mask = pairwise_contact_matrix(coords_a, coords_b, cutoff)
            contact_ab += mask
            total_ab += int(mask.sum())

        for idx_a2 in indices_a[i + 1:]:
            coords_a2 = positions[idx_a2]
            mask = pairwise_contact_matrix(coords_a, coords_a2, cutoff)
            contact_aa += mask
            total_aa += int(mask.sum())

    for i, idx_b in enumerate(indices_b):
        coords_b = positions[idx_b]

        for idx_b2 in indices_b[i + 1:]:
            coords_b2 = positions[idx_b2]
            mask = pairwise_contact_matrix(coords_b, coords_b2, cutoff)
            contact_bb += mask
            total_bb += int(mask.sum())

    return total_aa, total_ab, total_bb


def normalize_by_maximum(matrix):
    """Normalize a matrix by its maximum value for visualization."""
    max_value = matrix.max()
    return matrix / max_value if max_value > 0 else matrix


def plot_heatmap(matrix, title, output_file, xlabel, ylabel):
    """Write a heatmap figure."""
    plt.figure(figsize=(8, 6))
    image = plt.imshow(matrix, origin="lower", interpolation="nearest", vmin=0, vmax=1)
    plt.colorbar(image, label="Normalized contact frequency")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_contact_frequency_bars(frequencies, output_file):
    """Plot global contact frequencies between protein species."""
    labels = ["A-A", "A-B", "B-B"]
    values = [frequencies["AA"], frequencies["AB"], frequencies["BB"]]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylabel("Relative contact frequency")
    plt.title("Global contact frequencies between protein species")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def select_dna_residues(universe, prefix):
    """Select DNA residues according to a residue-name prefix."""
    residues = [res for res in universe.residues if res.resname.startswith(prefix)]
    return mda.core.groups.ResidueGroup(residues)


def compute_dna_contact_profile(universe, residues, dna_indices, start_frame, cutoff, n_dna):
    """Compute residue-resolved DNA contacts for one protein species."""
    n_residues = len(residues)
    contacts = np.zeros(n_residues, dtype=float)
    n_frames = 0

    for _ in universe.trajectory[start_frame:]:
        n_frames += 1
        positions = universe.atoms.positions
        dna_positions = positions[dna_indices]

        for i, residue in enumerate(residues):
            residue_positions = residue.atoms.positions
            mask = pairwise_contact_matrix(residue_positions, dna_positions, cutoff)
            contacts[i] += mask.sum()

    denominator = n_dna * n_frames if n_dna and n_frames > 0 else n_frames
    contact_frequency = contacts / denominator if denominator else contacts
    normalized = normalize_by_maximum(contact_frequency)

    return contacts, contact_frequency, normalized


def plot_contact_profile(profile, output_file, title):
    """Plot a one-dimensional residue contact profile."""
    plt.figure(figsize=(8, 1.8))
    plt.imshow(profile[np.newaxis, :], aspect="auto", vmin=0, vmax=1)
    plt.yticks([])
    plt.xlabel("Residue index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    universe = mda.Universe(args.top, args.traj)

    if args.wrap:
        universe.trajectory.add_transformations(wrap(universe.atoms))

    residues_a = universe.select_atoms(f"chainID {args.chain_a}").residues
    residues_b = universe.select_atoms(f"chainID {args.chain_b}").residues

    indices_a = split_residues_into_copies(
        residues_a,
        args.n_prot_a,
        args.res_per_a,
        "A",
    )
    indices_b = split_residues_into_copies(
        residues_b,
        args.n_prot_b,
        args.res_per_b,
        "B",
    )

    contact_ab = np.zeros((args.res_per_a, args.res_per_b), dtype=float)
    contact_aa = np.zeros((args.res_per_a, args.res_per_a), dtype=float)
    contact_bb = np.zeros((args.res_per_b, args.res_per_b), dtype=float)

    total_contacts = {"AA": 0, "AB": 0, "BB": 0}
    n_frames = 0

    print("Processing residue-resolved protein contact maps...")

    for frame_number, _ in enumerate(universe.trajectory[args.start_frame:]):
        n_frames += 1
        positions = universe.atoms.positions

        aa, ab, bb = update_contact_maps(
            positions,
            indices_a,
            indices_b,
            contact_aa,
            contact_ab,
            contact_bb,
            args.cutoff,
        )

        total_contacts["AA"] += aa
        total_contacts["AB"] += ab
        total_contacts["BB"] += bb

        if n_frames % 10 == 0:
            print(f"  Processed {n_frames} frames.")

    if n_frames == 0:
        raise RuntimeError("No frames were analyzed. Check --start-frame.")

    pd.DataFrame(contact_ab).to_csv(output_dir / "contact_AB_raw.csv", index=False)
    pd.DataFrame(contact_aa).to_csv(output_dir / "contact_AA_raw.csv", index=False)
    pd.DataFrame(contact_bb).to_csv(output_dir / "contact_BB_raw.csv", index=False)

    contact_ab_norm = normalize_by_maximum(contact_ab)
    contact_aa_norm = normalize_by_maximum(contact_aa)
    contact_bb_norm = normalize_by_maximum(contact_bb)

    pd.DataFrame(contact_ab_norm).to_csv(output_dir / "contact_AB_normalized.csv", index=False)
    pd.DataFrame(contact_aa_norm).to_csv(output_dir / "contact_AA_normalized.csv", index=False)
    pd.DataFrame(contact_bb_norm).to_csv(output_dir / "contact_BB_normalized.csv", index=False)

    plot_heatmap(contact_ab_norm, "A-B contact map", output_dir / "contact_AB_map.png", "Residue B", "Residue A")
    plot_heatmap(contact_aa_norm, "A-A contact map", output_dir / "contact_AA_map.png", "Residue A", "Residue A")
    plot_heatmap(contact_bb_norm, "B-B contact map", output_dir / "contact_BB_map.png", "Residue B", "Residue B")

    total_comparisons = (
        args.n_prot_a * args.n_prot_b
        + args.n_prot_a * (args.n_prot_a - 1) // 2
        + args.n_prot_b * (args.n_prot_b - 1) // 2
    ) * n_frames

    frequencies = {
        key: value / total_comparisons
        for key, value in total_contacts.items()
    }

    with open(output_dir / "global_contact_frequencies.txt", "w", encoding="utf-8") as handle:
        handle.write("Global contact frequencies between protein species\n")
        handle.write(f"A-A: {frequencies['AA']:.8f}\n")
        handle.write(f"A-B: {frequencies['AB']:.8f}\n")
        handle.write(f"B-B: {frequencies['BB']:.8f}\n")

    plot_contact_frequency_bars(
        frequencies,
        output_dir / "global_contact_frequencies.png",
    )

    if args.dna_resname_prefix:
        if args.n_dna is None:
            raise ValueError("--n-dna is required when --dna-resname-prefix is used.")

        dna_residues = select_dna_residues(universe, args.dna_resname_prefix)
        dna_indices = dna_residues.atoms.indices

        if len(dna_indices) == 0:
            raise ValueError("No DNA atoms found using the provided residue-name prefix.")

        for label, residues in [("A", residues_a), ("B", residues_b)]:
            raw, frequency, normalized = compute_dna_contact_profile(
                universe,
                residues,
                dna_indices,
                args.start_frame,
                args.cutoff,
                args.n_dna,
            )

            pd.DataFrame(raw).to_csv(output_dir / f"contact_DNA_{label}_raw.csv", index=False)
            pd.DataFrame(frequency).to_csv(output_dir / f"contact_DNA_{label}_frequency.csv", index=False)
            pd.DataFrame(normalized).to_csv(output_dir / f"contact_DNA_{label}_normalized.csv", index=False)

            plot_contact_profile(
                normalized,
                output_dir / f"contact_DNA_{label}_profile.png",
                f"DNA contacts with protein {label}",
            )

    print("Protein contact-map analysis completed successfully.")
    print(f"Analyzed frames: {n_frames}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
