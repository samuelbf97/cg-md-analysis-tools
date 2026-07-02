#!/usr/bin/env python3
"""
dna_contact_bfactor_mapper.py

Compute per-bead DNA contact probabilities from MD trajectories and write the
results as CSV files and PDB files with probabilities encoded in the B-factor
field.

This script is useful for visualizing where proteins contact coarse-grained DNA
in VMD, PyMOL, or ChimeraX.

Features
--------
- Supports MDAnalysis-compatible topologies and trajectories.
- Selects DNA beads by chain/segid and bead names.
- Computes contact probability per DNA bead for one or more target selections.
- Writes CSV files with per-bead probabilities.
- Writes PDB files with probabilities encoded as B-factors.
- Optionally writes combined maps using max or clipped sum of two targets.

Author
------
Samuel Blázquez Fernández
"""

import argparse
from pathlib import Path

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-bead DNA contact probabilities and write PDB files "
            "with probabilities encoded in B-factors."
        )
    )

    parser.add_argument("--top", required=True, help="Topology file, e.g. PDB.")
    parser.add_argument("--traj", required=True, help="Trajectory file, e.g. DCD.")

    parser.add_argument(
        "--dna-chains",
        default="A,B",
        help="Comma-separated DNA chain IDs. Default: A,B."
    )

    parser.add_argument(
        "--dna-beads",
        default="P,S,B,DP,DS",
        help="Comma-separated DNA bead names to include. Default: P,S,B,DP,DS."
    )

    parser.add_argument(
        "--target",
        action="append",
        required=True,
        help=(
            "Target definition in the format name:selection. "
            "Example: H14:'chainid K' or H3_tail_C:'chainid C and resid 1-36'. "
            "Can be supplied multiple times."
        )
    )

    parser.add_argument(
        "--cutoff",
        type=float,
        default=8.0,
        help="Contact cutoff distance in Angstrom. Default: 8.0."
    )

    parser.add_argument(
        "--output-dir",
        default="dna_contact_bfactor_output",
        help="Directory for output files."
    )

    parser.add_argument(
        "--invert",
        action="store_true",
        help="Write 1 - probability instead of probability in the B-factor field."
    )

    parser.add_argument(
        "--combined",
        nargs=2,
        action="append",
        metavar=("TARGET_A", "TARGET_B"),
        help=(
            "Optional pair of target names to combine using max and clipped sum. "
            "Can be supplied multiple times."
        )
    )

    return parser.parse_args()


def parse_targets(target_args):
    """Parse target definitions supplied as name:selection."""
    targets = {}

    for item in target_args:
        if ":" not in item:
            raise ValueError(
                f"Invalid target '{item}'. Expected format: name:selection."
            )

        name, selection = item.split(":", 1)
        name = name.strip()
        selection = selection.strip().strip("'").strip('"')

        if not name or not selection:
            raise ValueError(
                f"Invalid target '{item}'. Both name and selection are required."
            )

        targets[name] = selection

    return targets


def selection_token_for_chain(universe, chain_id):
    """
    Return the selection keyword ('chainid' or 'segid') that works for a chain.
    """
    for token in ("chainid", "segid"):
        try:
            if universe.select_atoms(f"{token} {chain_id}").n_atoms > 0:
                return token
        except Exception:
            pass

    raise ValueError(
        f"No atoms found for chain/segid '{chain_id}'. Check the topology."
    )


def select_dna(universe, chain_ids, bead_names):
    """Select DNA beads using chain IDs and bead names."""
    chain_clauses = []

    for chain_id in chain_ids:
        token = selection_token_for_chain(universe, chain_id)
        chain_clauses.append(f"{token} {chain_id}")

    bead_clause = " or ".join(f"name {name}" for name in bead_names)
    chain_clause = " or ".join(chain_clauses)

    selection = f"({chain_clause}) and ({bead_clause})"
    dna = universe.select_atoms(selection)

    if dna.n_atoms == 0:
        fallback = f"({chain_clause}) and nucleic"
        dna = universe.select_atoms(fallback)

    if dna.n_atoms == 0:
        raise ValueError(
            "DNA selection is empty. Check --dna-chains and --dna-beads."
        )

    return dna


def contact_probability_per_dna_bead(universe, dna, target, cutoff):
    """
    Compute the probability that each DNA bead contacts a target atom group.
    """
    if target.n_atoms == 0:
        raise ValueError("Target selection is empty.")

    counts = np.zeros(dna.n_atoms, dtype=np.int64)
    n_frames = 0

    universe.trajectory[0]

    for _ in universe.trajectory:
        distance_matrix = distances.distance_array(
            dna.positions,
            target.positions,
            backend="OpenMP",
        )
        touched = np.any(distance_matrix < cutoff, axis=1)
        counts += touched.astype(np.int64)
        n_frames += 1

    return counts / max(1, n_frames), n_frames


def ensure_topology_attribute(universe, attr_name, fill_value=0.0):
    """Add a topology attribute if missing."""
    try:
        values = getattr(universe.atoms, attr_name)
        if values is None or len(values) != universe.atoms.n_atoms:
            raise AttributeError
    except Exception:
        universe.add_TopologyAttr(
            attr_name,
            np.full(universe.atoms.n_atoms, fill_value, dtype=float),
        )

    return getattr(universe.atoms, attr_name)


def write_bfactor_pdb(universe, dna, values, output_file, invert=False):
    """Write a PDB file with values encoded in the B-factor field for DNA beads."""
    bfactor_values = 1.0 - values if invert else values

    beta = ensure_topology_attribute(universe, "tempfactors", 0.0).copy()
    beta[:] = 0.0
    beta[dna.indices] = bfactor_values
    universe.atoms.tempfactors = beta

    universe.trajectory[0]

    with mda.coordinates.PDB.PDBWriter(str(output_file), multiframe=False) as writer:
        writer.write(universe.atoms)


def write_probability_csv(dna, probabilities, n_frames, output_file):
    """Write per-DNA-bead probabilities to CSV."""
    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write("atom_index,chain,segid,resid,resname,name,probability,n_frames\n")

        for atom, probability in zip(dna.atoms, probabilities):
            chain = getattr(atom, "chainID", "")
            segid = getattr(atom, "segid", "")
            handle.write(
                f"{atom.index},{chain},{segid},{atom.resid},{atom.resname},"
                f"{atom.name},{probability:.6f},{n_frames}\n"
            )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    universe = mda.Universe(args.top, args.traj)

    chain_ids = [item.strip() for item in args.dna_chains.split(",") if item.strip()]
    bead_names = [item.strip() for item in args.dna_beads.split(",") if item.strip()]
    targets = parse_targets(args.target)

    dna = select_dna(universe, chain_ids, bead_names)
    print(f"Selected {dna.n_atoms} DNA beads.")

    probabilities = {}
    n_frames_by_target = {}

    for name, selection in targets.items():
        target = universe.select_atoms(selection)
        print(f"Target '{name}': {target.n_atoms} atoms selected.")

        probs, n_frames = contact_probability_per_dna_bead(
            universe,
            dna,
            target,
            args.cutoff,
        )

        probabilities[name] = probs
        n_frames_by_target[name] = n_frames

        csv_file = output_dir / f"dna_contact_probability_{name}.csv"
        pdb_file = output_dir / f"dna_contact_probability_{name}_bfactor.pdb"

        write_probability_csv(dna, probs, n_frames, csv_file)
        write_bfactor_pdb(universe, dna, probs, pdb_file, invert=args.invert)

        print(f"  Wrote {csv_file}")
        print(f"  Wrote {pdb_file}")

    if args.combined:
        for target_a, target_b in args.combined:
            if target_a not in probabilities or target_b not in probabilities:
                raise ValueError(
                    f"Cannot combine '{target_a}' and '{target_b}'. "
                    "Both names must correspond to defined targets."
                )

            max_values = np.maximum(probabilities[target_a], probabilities[target_b])
            sum_values = np.clip(
                probabilities[target_a] + probabilities[target_b],
                0.0,
                1.0,
            )

            prefix = f"{target_a}_and_{target_b}"
            write_bfactor_pdb(
                universe,
                dna,
                max_values,
                output_dir / f"dna_contact_probability_{prefix}_max_bfactor.pdb",
                invert=args.invert,
            )
            write_bfactor_pdb(
                universe,
                dna,
                sum_values,
                output_dir / f"dna_contact_probability_{prefix}_sum_bfactor.pdb",
                invert=args.invert,
            )

            print(f"  Wrote combined maps for {target_a} and {target_b}.")

    print("DNA contact mapping completed successfully.")


if __name__ == "__main__":
    main()
