#!/usr/bin/env python3
"""Plot stand-up height tracking and height-to-joint-position relationships."""

import argparse
import csv
from pathlib import Path

LEGS = {
    "FL": (1, 2, 1.0),
    "FR": (5, 6, 1.0),
    "HL": (9, 10, -1.0),
    "HR": (13, 14, -1.0),
}


def read_csv(path: Path) -> dict[str, list]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"CSV contains no data: {path}")
    data: dict[str, list] = {key: [] for key in rows[0]}
    for row in rows:
        for key, value in row.items():
            data[key].append(value if key == "phase" else float(value))
    return data


def select(values: list, mask: list[bool]) -> list:
    return [value for value, keep in zip(values, mask) if keep]


def plot_overview(data: dict[str, list], output: Path) -> None:
    time = data["time_s"]
    mask = [phase == "height_interpolation" for phase in data["phase"]]
    command_height = select(data["command_height_m"], mask)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    axes[0, 0].plot(time, data["command_height_m"], label="command")
    axes[0, 0].plot(time, data["base_height_m"], label="measured")
    axes[0, 0].set(xlabel="Time (s)", ylabel="Height (m)", title="Height tracking")
    axes[0, 0].legend()

    for leg, (hip, knee, sign) in LEGS.items():
        hip_pos = [sign * value for value in select(data[f"joint_{hip}_pos"], mask)]
        hip_target = [sign * value for value in select(data[f"joint_{hip}_target"], mask)]
        axes[0, 1].plot(command_height, hip_pos, label=f"{leg} actual")
        axes[0, 1].plot(command_height, hip_target, "--", alpha=0.7)

        knee_pos = [sign * value for value in select(data[f"joint_{knee}_pos"], mask)]
        knee_target = [sign * value for value in select(data[f"joint_{knee}_target"], mask)]
        axes[1, 0].plot(command_height, knee_pos, label=f"{leg} actual")
        axes[1, 0].plot(command_height, knee_target, "--", alpha=0.7)

        hip_error = [actual - target for actual, target in zip(hip_pos, hip_target)]
        knee_error = [actual - target for actual, target in zip(knee_pos, knee_target)]
        axes[1, 1].plot(command_height, hip_error, label=f"{leg} hip-y")
        axes[1, 1].plot(command_height, knee_error, "--", label=f"{leg} knee")

    axes[0, 1].set(xlabel="Command height (m)", ylabel="Normalized joint position (rad)", title="Hip-y: solid actual, dashed target")
    axes[1, 0].set(xlabel="Command height (m)", ylabel="Normalized joint position (rad)", title="Knee: solid actual, dashed target")
    axes[1, 1].set(xlabel="Command height (m)", ylabel="Actual - target (rad)", title="Joint tracking error")
    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    axes[0, 1].legend(ncol=2, fontsize=8)
    axes[1, 0].legend(ncol=2, fontsize=8)
    axes[1, 1].legend(ncol=2, fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_measured_relation(data: dict[str, list], output: Path) -> None:
    mask = [phase == "height_interpolation" for phase in data["phase"]]
    height = select(data["base_height_m"], mask)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for leg, (hip, knee, sign) in LEGS.items():
        axes[0].plot(height, [sign * x for x in select(data[f"joint_{hip}_pos"], mask)], label=leg)
        axes[1].plot(height, [sign * x for x in select(data[f"joint_{knee}_pos"], mask)], label=leg)
    axes[0].set(title="Measured height vs hip-y", xlabel="Measured base height (m)", ylabel="Normalized joint position (rad)")
    axes[1].set(title="Measured height vs knee", xlabel="Measured base height (m)", ylabel="Normalized joint position (rad)")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", type=Path, default=Path("logs/standup_height_joint_data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs"))
    args = parser.parse_args()
    global plt
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = read_csv(args.csv)
    overview = args.output_dir / "standup_height_joint_overview.png"
    measured = args.output_dir / "standup_measured_height_joint_pos.png"
    plot_overview(data, overview)
    plot_measured_relation(data, measured)
    print(f"Saved: {overview}")
    print(f"Saved: {measured}")


if __name__ == "__main__":
    main()
