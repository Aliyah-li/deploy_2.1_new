"""
Offline (headless) wheel FK error checker.

Usage:
  1. Record qpos from the running simulation:
     RECORD_QPOS_PATH=/tmp/qpos.npz RECORD_QPOS_INTERVAL=50 \
       python3 src/M20_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py

  2. After the simulation finishes (or you Ctrl+C), run this script:
     python3 check_wheel_fk_error_offline.py \
       --xml src/M20_sdk_deploy/model/M20/mjcf/M20.xml \
       --qpos-file /tmp/qpos.npz \
       --eps 1e-6
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

WHEELS = {
    "fl": {"body": "fl_wheel", "site": "fl_wheel_center"},
    "fr": {"body": "fr_wheel", "site": "fr_wheel_center"},
    "hl": {"body": "hl_wheel", "site": "hl_wheel_center"},
    "hr": {"body": "hr_wheel", "site": "hr_wheel_center"},
}


def get_body_id(model, name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise RuntimeError(f"Body not found: {name}")
    return body_id


def get_site_id(model, name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id < 0:
        raise RuntimeError(f"Site not found: {name}")
    return site_id


def get_fk_wheel_center_from_body(model, data, body_id, site_id):
    """Manual FK: p_site_world = p_body_world + R_body_world @ p_site_local."""
    body_pos_world = data.xpos[body_id].copy()
    body_rot_world = data.xmat[body_id].reshape(3, 3).copy()
    site_local_pos = model.site_pos[site_id].copy()
    return body_pos_world + body_rot_world @ site_local_pos


def build_ids(model):
    ids = {}
    for leg, names in WHEELS.items():
        ids[leg] = {
            "body_name": names["body"],
            "site_name": names["site"],
            "body_id": get_body_id(model, names["body"]),
            "site_id": get_site_id(model, names["site"]),
        }
    return ids


def main():
    parser = argparse.ArgumentParser(
        description="Offline (headless) wheel FK error checker"
    )
    parser.add_argument("--xml", type=str, required=True, help="Path to MuJoCo XML")
    parser.add_argument(
        "--qpos-file", type=str, required=True, help="Path to recorded .npz qpos file"
    )
    parser.add_argument(
        "--eps", type=float, default=1e-6, help="Only print errors above this threshold"
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=0,
        help="Print progress every N snapshots (0 = silent)",
    )
    args = parser.parse_args()

    # ---- load recorded qpos ----
    qpos_path = Path(args.qpos_file)
    if not qpos_path.exists():
        raise FileNotFoundError(f"qpos file not found: {qpos_path}")
    data_npz = np.load(str(qpos_path))
    qpos_all = data_npz["qpos"]  # shape: (n_snapshots, nq)
    n_snapshots = qpos_all.shape[0]
    print(f"[INFO] Loaded {n_snapshots} qpos snapshots (shape={qpos_all.shape})")

    # ---- load model (HEADLESS — no viewer, no GPU) ----
    xml_path = Path(args.xml)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    ids = build_ids(model)

    print(f"[INFO] Model loaded. Checking FK errors, eps={args.eps}")
    print(f"[INFO] Wheels: {list(ids.keys())}")

    # ---- per-wheel accumulators ----
    max_z_error = {leg: 0.0 for leg in ids}
    max_pos_error = {leg: 0.0 for leg in ids}
    sum_z_error = {leg: 0.0 for leg in ids}
    sum_pos_error = {leg: 0.0 for leg in ids}
    total_exceeded = {leg: 0 for leg in ids}

    for i in range(n_snapshots):
        # Set qpos and run forward kinematics only (no physics integration)
        data.qpos[:] = qpos_all[i]
        mujoco.mj_forward(model, data)

        for leg, item in ids.items():
            fk_pos = get_fk_wheel_center_from_body(
                model, data, item["body_id"], item["site_id"]
            )
            sim_pos = data.site_xpos[item["site_id"]].copy()
            error_vec = fk_pos - sim_pos
            pos_error = float(np.linalg.norm(error_vec))
            z_error = float(abs(error_vec[2]))

            # update stats
            if z_error > max_z_error[leg]:
                max_z_error[leg] = z_error
            if pos_error > max_pos_error[leg]:
                max_pos_error[leg] = pos_error
            sum_z_error[leg] += z_error
            sum_pos_error[leg] += pos_error

            if z_error > args.eps or pos_error > args.eps:
                total_exceeded[leg] += 1
                print(
                    f"  [{i}] {leg}: "
                    f"z_err={z_error:.6e}, pos_err={pos_error:.6e}, "
                    f"fk_z={fk_pos[2]:.9f}, sim_z={sim_pos[2]:.9f}"
                )

        if args.print_every > 0 and (i + 1) % args.print_every == 0:
            print(f"[PROGRESS] {i + 1}/{n_snapshots} snapshots processed")

    # ---- summary ----
    print("\n" + "=" * 65)
    print("SUMMARY — Wheel FK Error (manual FK vs MuJoCo site_xpos)")
    print("=" * 65)
    print(f"{'Wheel':>6s}  {'max|z_err|':>14s}  {'max|pos_err|':>14s}  "
          f"{'mean|z_err|':>14s}  {'mean|pos_err|':>14s}  {'#exceeded':>10s}")
    print("-" * 65)
    for leg in sorted(ids.keys()):
        mean_z = sum_z_error[leg] / n_snapshots
        mean_pos = sum_pos_error[leg] / n_snapshots
        print(
            f"{leg:>6s}  {max_z_error[leg]:>14.6e}  {max_pos_error[leg]:>14.6e}  "
            f"{mean_z:>14.6e}  {mean_pos:>14.6e}  {total_exceeded[leg]:>10d}"
        )
    print("-" * 65)
    print(f"Total snapshots processed: {n_snapshots}")
    print("=" * 65)


if __name__ == "__main__":
    main()
