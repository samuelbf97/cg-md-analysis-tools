#!/usr/bin/env python3
"""
sliding_bp_tracker.py

Track the closest DNA phosphate bead to a protein anchor and map it onto a
strand-invariant base-pair index along a double-stranded DNA molecule.

This script is useful for following sliding or translocation of a protein along
coarse-grained DNA trajectories.

Features
--------
- Supports MDAnalysis-compatible topology and trajectory files.
- Uses either two atom serials or an MDAnalysis selection as the protein anchor.
- Maps both DNA strands onto a single 1..N base-pair coordinate.
- Writes time series of closest base pair, chain, residue, and distance.
- Optionally generates a trajectory plot and a base-pair occupancy histogram.

Author
------
Samuel Blázquez Fernández
"""

import argparse

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Track the closest DNA phosphate bead to a protein anchor and map it "
            "onto a strand-invariant base-pair index."
        )
    )

    parser.add_argument("--top", required=True, help="Topology file.")
    parser.add_argument("--traj", help="Trajectory file. If omitted, only the topology is used.")

    parser.add_argument(
        "--anchor-serial",
        help="Two PDB atom serials defining the anchor, e.g. 772,774."
    )

    parser.add_argument(
        "--sel-anchor",
        help="Alternative MDAnalysis selection defining the anchor. Must select two atoms."
    )

    parser.add_argument("--chainA", default="A", help="First DNA strand chain ID.")
    parser.add_argument("--chainB", default="B", help="Second DNA strand chain ID.")

    parser.add_argument(
        "--sel-dnaP",
        default="(name P or name DP)",
        help="Selection for DNA phosphate beads."
    )

    parser.add_argument(
        "--n-bp",
        type=int,
        required=True,
        help="Total number of DNA base pairs."
    )

    parser.add_argument("--stride", type=int, default=1, help="Trajectory stride.")

    parser.add_argument(
        "--out-ts",
        default="sliding_bp.csv",
        help="CSV output with time, bp, chain, resid, and distance."
    )

    parser.add_argument(
        "--out-path",
        default="sliding_bp.txt",
        help="Two-column text output: time and base-pair index."
    )

    parser.add_argument("--plot", action="store_true", help="Generate a PNG plot.")
    parser.add_argument("--fig", default="sliding_bp.png", help="Output figure name.")

    return parser.parse_args()


def get_anchor(universe, args):
    """Return the atom group defining the protein anchor."""
    if args.anchor_serial:
        serial_1, serial_2 = [int(value.strip()) for value in args.anchor_serial.split(",")]
        anchor = universe.select_atoms(f"bynum {serial_1} {serial_2}")

        if anchor.n_atoms != 2:
            raise ValueError(
                f"--anchor-serial {serial_1},{serial_2} selected "
                f"{anchor.n_atoms} atoms, but exactly 2 are required."
            )

        return anchor

    if args.sel_anchor:
        anchor = universe.select_atoms(args.sel_anchor)

        if anchor.n_atoms != 2:
            raise ValueError(
                f"--sel-anchor selected {anchor.n_atoms} atoms, "
                "but exactly 2 are required."
            )

        return anchor

    raise ValueError("Specify either --anchor-serial or --sel-anchor.")


def select_by_chain_or_segid(universe, base_selection, chain_id):
    """Select atoms using chainid first and segid as fallback."""
    selection = universe.select_atoms(f"({base_selection}) and (chainid {chain_id})")

    if selection.n_atoms == 0:
        selection = universe.select_atoms(f"({base_selection}) and (segid {chain_id})")

    return selection


def build_base_pair_mapper(universe, phosphate_selection, chain_a, chain_b, n_bp):
    """
    Build a function mapping chain/residue identifiers to a 1..N base-pair index.
    """
    phosphates_a = select_by_chain_or_segid(universe, phosphate_selection, chain_a)
    phosphates_b = select_by_chain_or_segid(universe, phosphate_selection, chain_b)

    if phosphates_a.n_atoms == 0 or phosphates_b.n_atoms == 0:
        raise ValueError(
            "Could not find phosphate beads in both DNA strands. "
            "Check --chainA, --chainB, and --sel-dnaP."
        )

    phosphates_all = phosphates_a + phosphates_b

    ranks_a = {
        resid: index + 1
        for index, resid in enumerate(sorted(set(phosphates_a.resids)))
    }
    ranks_b = {
        resid: index + 1
        for index, resid in enumerate(sorted(set(phosphates_b.resids)))
    }

    def to_base_pair(chain_id, resid):
        if chain_id == chain_a:
            base_pair = ranks_a.get(resid)
        elif chain_id == chain_b:
            rank = ranks_b.get(resid)
            base_pair = n_bp - rank + 1 if rank is not None else None
        else:
            # Fallback for topologies where chainID is missing.
            if resid in ranks_a:
                base_pair = ranks_a[resid]
            elif resid in ranks_b:
                base_pair = n_bp - ranks_b[resid] + 1
            else:
                raise KeyError(f"Residue {resid} is not found in either DNA strand.")

        if base_pair is None:
            raise KeyError(f"Residue {resid} is not found in chain {chain_id}.")

        return int(np.clip(base_pair, 1, n_bp))

    return phosphates_all, to_base_pair


def write_outputs(times, base_pairs, chains, resids, distances, args):
    """Write CSV and two-column time series outputs."""
    with open(args.out_ts, "w", encoding="utf-8") as handle:
        handle.write("# time_ps,bp,chain,resid,dist_A\n")

        for time, bp, chain, resid, distance in zip(
            times,
            base_pairs,
            chains,
            resids,
            distances,
        ):
            handle.write(f"{time:.3f},{bp},{chain},{resid},{distance:.3f}\n")

    with open(args.out_path, "w", encoding="utf-8") as handle:
        handle.write("# time_ps bp\n")

        for time, bp in zip(times, base_pairs):
            handle.write(f"{time:.3f} {bp}\n")


def plot_results(times, base_pairs, args):
    """Generate a time-series plot and a base-pair occupancy histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times_ns = np.asarray(times) / 1000.0

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

    axes[0].plot(times_ns, base_pairs, linewidth=1.0)
    axes[0].set_xlabel("Time (ns)")
    axes[0].set_ylabel("Base-pair index")
    axes[0].set_title("Protein sliding along DNA")

    n_bins = min(args.n_bp, max(40, int(np.sqrt(len(base_pairs)))))
    axes[1].hist(base_pairs, bins=n_bins, density=True, histtype="step", linewidth=1.8)
    axes[1].set_xlabel("Base-pair index")
    axes[1].set_ylabel("Probability density")

    fig.tight_layout()
    fig.savefig(args.fig, dpi=300)
    plt.close(fig)


def main():
    args = parse_args()

    if args.traj:
        universe = mda.Universe(args.top, args.traj)
    else:
        universe = mda.Universe(args.top)

    anchor = get_anchor(universe, args)
    phosphates, to_base_pair = build_base_pair_mapper(
        universe,
        args.sel_dnaP,
        args.chainA,
        args.chainB,
        args.n_bp,
    )

    try:
        chain_ids = phosphates.chainIDs
    except Exception:
        chain_ids = np.array([""] * phosphates.n_atoms)

    resids = phosphates.resids

    times = []
    base_pairs = []
    chains_out = []
    resids_out = []
    min_distances = []

    for timestep in universe.trajectory[::args.stride]:
        anchor_position = anchor.positions.mean(axis=0).reshape(1, 3)
        distances = distance_array(anchor_position, phosphates.positions)[0]

        closest_index = int(np.argmin(distances))
        chain_id = str(chain_ids[closest_index])
        resid = int(resids[closest_index])
        base_pair = to_base_pair(chain_id, resid)

        times.append(timestep.time)
        base_pairs.append(base_pair)
        chains_out.append(chain_id)
        resids_out.append(resid)
        min_distances.append(float(distances[closest_index]))

    write_outputs(times, base_pairs, chains_out, resids_out, min_distances, args)

    if args.plot:
        plot_results(times, base_pairs, args)

    print(f"Written: {args.out_ts}")
    print(f"Written: {args.out_path}")

    if args.plot:
        print(f"Written: {args.fig}")


if __name__ == "__main__":
    main()
