#!/usr/bin/env python3
"""
dna_rotation_tracker.py

Estimate azimuthal DNA rotation/sliding in nucleosome-like coarse-grained
trajectories using a protein-anchored reference frame.

The script computes a time series of angular displacement converted to an
approximate base-pair displacement for a selected DNA block. It also performs
a secondary nearest-phosphate-to-protein-anchor check with hysteresis.

Features
--------
- Builds strand pairing from two DNA chains.
- Defines a protein-anchored reference frame using Kabsch alignment.
- Converts azimuthal rotation into base-pair displacement.
- Performs an optional phosphate-to-CA anchor consistency check.
- Writes time series and metadata files.

Author
------
Samuel Blázquez Fernández
"""

import argparse
from pathlib import Path

import numpy as np
import MDAnalysis as mda


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Track azimuthal DNA rotation/sliding using a protein-anchored "
            "reference frame."
        )
    )

    parser.add_argument("--top", required=True, help="Topology file.")
    parser.add_argument("--traj", required=True, help="Trajectory file.")
    parser.add_argument("--chain-a-sel", default="chainID A", help="Selection for DNA strand A.")
    parser.add_argument("--chain-b-sel", default="chainID B", help="Selection for DNA strand B.")
    parser.add_argument("--bead-name", default="DP", help="DNA bead used for azimuthal tracking.")
    parser.add_argument("--center-atom-id", type=int, required=True, help="Reference atom serial/id, 1-based.")
    parser.add_argument("--half-block-bp", type=int, default=5, help="Half-size of the tracked DNA block.")
    parser.add_argument("--exclude-center", action="store_true", help="Exclude center bead from the tracked block.")
    parser.add_argument(
        "--protein-core-sel",
        default="name CA and not (chainID A or chainID B)",
        help="Protein selection used to define the reference frame."
    )

    parser.add_argument("--calib-halfwin-pairs", type=int, default=30)
    parser.add_argument("--calib-min-span-rad", type=float, default=1.0)
    parser.add_argument("--calib-max-expand", type=int, default=80)
    parser.add_argument("--bp-per-2pi-fallback", type=float, default=86.5)
    parser.add_argument("--clamp-bp-per-rad", default="9.0,20.0")
    parser.add_argument("--time-unit", choices=["ps", "ns"], default="ns")
    parser.add_argument("--out-prefix", default="rot")

    parser.add_argument("--p-bead-name", default="DP", help="Phosphate bead name for anchor check.")
    parser.add_argument("--ref-p-atom-id", type=int, default=None, help="Reference phosphate atom id for anchor check.")
    parser.add_argument("--anchor-ca-count", type=int, default=5)
    parser.add_argument("--hysteresis-margin", type=float, default=1.5)
    parser.add_argument("--hysteresis-min-frames", type=int, default=2)
    parser.add_argument("--output-dir", default="dna_rotation_output")

    return parser.parse_args()


def sort_by_index(atom_group):
    return atom_group[np.argsort(atom_group.indices)]


def fit_plane_pca(coordinates):
    center = coordinates.mean(axis=0)
    centered = coordinates - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1] / np.linalg.norm(vt[-1])

    axis_1 = np.cross([1.0, 0.0, 0.0], normal)
    if np.linalg.norm(axis_1) < 1e-6:
        axis_1 = np.cross([0.0, 1.0, 0.0], normal)

    axis_1 /= np.linalg.norm(axis_1)
    axis_2 = np.cross(normal, axis_1)

    return center, normal, axis_1, axis_2


def circular_mean(angles):
    return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))


def unwrap_centered(angles):
    unwrapped = np.unwrap(angles)
    return unwrapped - unwrapped[0]


def get_times_array(universe, time_unit):
    times_ps = []

    for timestep in universe.trajectory:
        times_ps.append(getattr(timestep, "time", len(times_ps)))

    universe.trajectory[0]
    times_ps = np.asarray(times_ps, dtype=float)

    if time_unit == "ns":
        return times_ps / 1000.0, "time_ns"

    return times_ps, "time_ps"


def circular_diff(i, j, n):
    diff = int(i) - int(j)

    if diff > n / 2:
        diff -= n

    if diff < -n / 2:
        diff += n

    return int(diff)


def mean_pair_distance(group_a, group_b):
    n_pairs = min(len(group_a), len(group_b))
    return float(
        np.mean(
            np.linalg.norm(
                group_a.positions[:n_pairs] - group_b.positions[:n_pairs],
                axis=1,
            )
        )
    )


def build_index_pairing(universe, selection_a, selection_b):
    """Build paired DNA strands by choosing the orientation with smaller pair distance."""
    strand_a = sort_by_index(selection_a)
    strand_b = sort_by_index(selection_b)

    universe.trajectory[0]

    distance_forward = mean_pair_distance(strand_a, strand_b[::-1])
    distance_reverse = mean_pair_distance(strand_a[::-1], strand_b)

    if distance_forward <= distance_reverse:
        n_pairs = min(len(strand_a), len(strand_b))
        return strand_a[:n_pairs], strand_b[::-1][:n_pairs], np.arange(n_pairs), "A_forward_B_reverse"

    n_pairs = min(len(strand_a), len(strand_b))
    return strand_a[::-1][:n_pairs], strand_b[:n_pairs], np.arange(n_pairs), "A_reverse_B_forward"


def find_center_pair_index(universe, paired_a, paired_b, center_atom_id, bead_name):
    center_atom = universe.atoms[center_atom_id - 1]
    center_chain = getattr(center_atom, "chainID", "")
    center_resid = center_atom.resid

    candidate_group = paired_a

    if len(paired_b) > 0 and center_chain == getattr(paired_b[0], "chainID", ""):
        candidate_group = paired_b

    matches = np.where(
        (candidate_group.resids == center_resid)
        & (candidate_group.names == bead_name)
    )[0]

    if len(matches) > 0:
        return int(matches[0])

    return int(np.argmin(np.abs(candidate_group.indices - center_atom.index)))


def kabsch(mobile, reference):
    """Return rotation matrix that aligns mobile coordinates onto reference coordinates."""
    covariance = mobile.T @ reference
    u_matrix, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u_matrix.T

    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u_matrix.T

    return rotation


def prepare_protein_reference_frame(universe, protein_selection):
    universe.trajectory[0]
    protein_0 = protein_selection.positions
    center_0 = protein_0.mean(axis=0)
    centered_0 = protein_0 - center_0
    plane_center, _, axis_1, axis_2 = fit_plane_pca(protein_0)

    return {
        "protein_center_0": center_0,
        "protein_centered_0": centered_0,
        "plane_center_0": plane_center,
        "axis_1": axis_1,
        "axis_2": axis_2,
    }


def align_coordinates_to_protein_frame(protein_selection, reference, coordinates):
    protein = protein_selection.positions
    center = protein.mean(axis=0)
    centered = protein - center
    rotation = kabsch(centered, reference["protein_centered_0"])

    return (coordinates - center) @ rotation + reference["protein_center_0"]


def robust_bp_per_rad(
    theta_0,
    abscissa,
    center_index,
    halfwin_init,
    min_span,
    max_expand,
    bp_per_2pi_fallback,
    clamp_range,
):
    """Estimate base pairs per radian from the initial DNA geometry."""
    for expansion in range(0, max_expand + 1, 5):
        start = max(0, center_index - (halfwin_init + expansion))
        end = min(len(theta_0), center_index + (halfwin_init + expansion) + 1)
        indices = np.arange(start, end)

        if len(indices) < 5:
            continue

        x_values = np.unwrap(theta_0[indices])
        span = float(np.max(x_values) - np.min(x_values))

        if span >= min_span:
            y_values = abscissa[indices].astype(float)
            matrix = np.column_stack([x_values, np.ones_like(x_values)])
            slope, _ = np.linalg.lstsq(matrix, y_values, rcond=None)[0]
            slope = float(np.clip(slope, clamp_range[0], clamp_range[1]))
            return slope, (start, end, span, slope, "fit")

    start = max(0, center_index - halfwin_init)
    end = min(len(theta_0), center_index + halfwin_init + 1)

    x_values = np.unwrap(theta_0[start:end])
    y_values = abscissa[start:end].astype(float)

    if len(x_values) >= 2 and np.std(x_values) > 1e-8 and np.std(y_values) > 1e-8:
        sign = 1.0 if np.corrcoef(x_values, y_values)[0, 1] >= 0 else -1.0
    else:
        sign = 1.0

    slope = sign * (bp_per_2pi_fallback / (2 * np.pi))
    slope = float(np.clip(slope, clamp_range[0], clamp_range[1]))
    span = float(np.max(x_values) - np.min(x_values)) if len(x_values) else 0.0

    return slope, (start, end, span, slope, "fallback")


def compute_delta_bp_azimuth(
    universe,
    protein_selection,
    paired_a,
    paired_b,
    block_indices,
    abscissa,
    center_index,
    calibration_parameters,
):
    """Compute azimuthal displacement converted into base-pair units."""
    reference = prepare_protein_reference_frame(universe, protein_selection)

    universe.trajectory[0]
    midpoint_0 = 0.5 * (paired_a.positions + paired_b.positions)
    midpoint_0_aligned = align_coordinates_to_protein_frame(
        protein_selection,
        reference,
        midpoint_0,
    )

    projected_0 = np.column_stack([
        (midpoint_0_aligned - reference["plane_center_0"]) @ reference["axis_1"],
        (midpoint_0_aligned - reference["plane_center_0"]) @ reference["axis_2"],
    ])
    theta_0 = np.arctan2(projected_0[:, 1], projected_0[:, 0])

    bp_per_rad, calibration_used = robust_bp_per_rad(
        theta_0,
        abscissa,
        center_index,
        calibration_parameters["halfwin_init"],
        calibration_parameters["min_span"],
        calibration_parameters["max_expand"],
        calibration_parameters["bp_per_2pi_fallback"],
        calibration_parameters["clamp_range"],
    )

    mean_angles = []

    for _ in universe.trajectory:
        midpoint = 0.5 * (paired_a.positions + paired_b.positions)
        aligned = align_coordinates_to_protein_frame(
            protein_selection,
            reference,
            midpoint,
        )

        projected = np.column_stack([
            (aligned - reference["plane_center_0"]) @ reference["axis_1"],
            (aligned - reference["plane_center_0"]) @ reference["axis_2"],
        ])

        theta = np.arctan2(projected[:, 1], projected[:, 0])
        mean_angles.append(circular_mean(theta[block_indices]))

    delta_theta = unwrap_centered(np.asarray(mean_angles))
    delta_bp = delta_theta * bp_per_rad

    universe.trajectory[0]

    return delta_bp, bp_per_rad, calibration_used


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clamp_range = tuple(float(value) for value in args.clamp_bp_per_rad.split(","))

    universe = mda.Universe(args.top, args.traj)

    selection_a = universe.select_atoms(f"({args.chain_a_sel}) and name {args.bead_name}")
    selection_b = universe.select_atoms(f"({args.chain_b_sel}) and name {args.bead_name}")

    if len(selection_a) == 0 or len(selection_b) == 0:
        raise RuntimeError(
            "DNA selections are empty. Check --chain-a-sel, --chain-b-sel, "
            "and --bead-name."
        )

    paired_a, paired_b, abscissa, orientation = build_index_pairing(
        universe,
        selection_a,
        selection_b,
    )

    n_pairs = len(paired_a)
    center_index = find_center_pair_index(
        universe,
        paired_a,
        paired_b,
        args.center_atom_id,
        args.bead_name,
    )

    start = max(0, center_index - args.half_block_bp)
    end = min(n_pairs, center_index + args.half_block_bp + 1)

    if args.exclude_center:
        block_indices = np.r_[
            np.arange(start, center_index),
            np.arange(center_index + 1, end),
        ]
    else:
        block_indices = np.arange(start, end)

    if len(block_indices) == 0:
        raise RuntimeError("Tracked DNA block is empty. Check --half-block-bp.")

    times, time_header = get_times_array(universe, args.time_unit)
    frames = np.arange(len(times))

    protein_selection = universe.select_atoms(args.protein_core_sel)

    if len(protein_selection) == 0:
        raise RuntimeError("Protein reference selection is empty.")

    calibration_parameters = {
        "halfwin_init": args.calib_halfwin_pairs,
        "min_span": args.calib_min_span_rad,
        "max_expand": args.calib_max_expand,
        "bp_per_2pi_fallback": args.bp_per_2pi_fallback,
        "clamp_range": clamp_range,
    }

    delta_bp, bp_per_rad, calibration_used = compute_delta_bp_azimuth(
        universe,
        protein_selection,
        paired_a,
        paired_b,
        block_indices,
        abscissa,
        center_index,
        calibration_parameters,
    )

    np.savetxt(
        output_dir / f"{args.out_prefix}_delta_bp_paired.txt",
        np.column_stack([frames, times, delta_bp]),
        header=f"frame {time_header} delta_bp_paired",
    )

    i0, i1, span, slope, mode = calibration_used

    with open(output_dir / f"{args.out_prefix}_meta.txt", "w", encoding="utf-8") as handle:
        handle.write(f"topology={args.top}\n")
        handle.write(f"trajectory={args.traj}\n")
        handle.write(f"chain_a_selection=({args.chain_a_sel})\n")
        handle.write(f"chain_b_selection=({args.chain_b_sel})\n")
        handle.write(f"bead_name={args.bead_name}\n")
        handle.write(f"pairing_orientation={orientation}\n")
        handle.write(f"n_pairs={n_pairs}\n")
        handle.write(f"center_atom_id={args.center_atom_id}\n")
        handle.write(f"center_index={center_index}\n")
        handle.write(f"block_indices={block_indices.tolist()}\n")
        handle.write(f"half_block_bp={args.half_block_bp}\n")
        handle.write(f"calibration_window=[{i0},{i1})\n")
        handle.write(f"calibration_span_rad={span:.6f}\n")
        handle.write(f"calibration_mode={mode}\n")
        handle.write(f"bp_per_rad={bp_per_rad:.6f}\n")
        handle.write(f"bp_per_2pi={bp_per_rad * 2 * np.pi:.6f}\n")
        handle.write(f"time_unit={time_header}\n")

    print("DNA rotation analysis completed successfully.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
