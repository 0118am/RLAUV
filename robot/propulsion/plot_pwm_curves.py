"""Plot the measured PWM-to-force curves for every T60 thruster."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from robot.dynamics.parameters import AUV
from robot.propulsion.curves import thruster_body_forces_from_pwm_us


_INSTALLATION_LABELS = (
    "vertical front-right",
    "vertical rear-right",
    "vertical front-left",
    "vertical rear-left",
    "horizontal rear-left",
    "horizontal rear-right",
    "horizontal front-left",
    "horizontal front-right",
)

_COMPONENT_STYLES = (
    ("Fx (forward)", "#d55e00"),
    ("Fy (left)", "#009e73"),
    ("Fz (up)", "#0072b2"),
)


def plot_pwm_curves(output: Path, samples: int = 1001) -> tuple[torch.Tensor, torch.Tensor]:
    """Save eight PWM curves and return the sampled PWM and axial thrust."""

    pwm = torch.linspace(
        AUV.thruster_pwm_center_us - AUV.thruster_pwm_half_range_us,
        AUV.thruster_pwm_center_us + AUV.thruster_pwm_half_range_us,
        samples,
        dtype=torch.float64,
    )
    pwm_by_thruster = pwm[:, None].expand(-1, len(AUV.thruster_labels))
    body_forces = thruster_body_forces_from_pwm_us(pwm_by_thruster)
    shaft_axes = torch.as_tensor(AUV.thruster_axes_body, dtype=body_forces.dtype)
    axial_thrust = torch.sum(body_forces * shaft_axes, dim=-1)

    figure, axes = plt.subplots(2, 4, figsize=(18, 8.5), sharex=True, sharey=True)
    deadband_min = AUV.thruster_pwm_center_us - AUV.thruster_pwm_deadband_us
    deadband_max = AUV.thruster_pwm_center_us + AUV.thruster_pwm_deadband_us
    thrust_limit = 1.12 * float(axial_thrust.abs().max())

    for index, (axis, label, installation) in enumerate(
        zip(axes.flat, AUV.thruster_labels, _INSTALLATION_LABELS, strict=True)
    ):
        curve = axial_thrust[:, index]
        axis.axvspan(deadband_min, deadband_max, color="#94a3b8", alpha=0.22)
        axis.axhline(0.0, color="#64748b", linewidth=0.9)
        axis.axvline(AUV.thruster_pwm_center_us, color="#94a3b8", linewidth=0.9, linestyle=":")
        axis.plot(pwm.numpy(), curve.numpy(), color="#0f6cbd", linewidth=2.4)
        axis.scatter(
            [float(pwm[0]), float(pwm[-1])],
            [float(curve[0]), float(curve[-1])],
            color="#0f6cbd",
            edgecolor="white",
            linewidth=0.8,
            s=34,
            zorder=3,
        )
        axis.set_title(f"{label}  |  {installation}", fontsize=11, fontweight="semibold")
        axis.text(
            0.04,
            0.94,
            f"{pwm[0]:.0f}: {curve[0]:+.2f} N\n{pwm[-1]:.0f}: {curve[-1]:+.2f} N",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#334155",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
        )
        axis.set_xlim(float(pwm[0]), float(pwm[-1]))
        axis.set_ylim(-thrust_limit, thrust_limit)
        axis.set_xticks(torch.linspace(float(pwm[0]), float(pwm[-1]), 5).tolist())
        axis.grid(True, color="#e2e8f0", linewidth=0.8)
        axis.set_axisbelow(True)

    figure.suptitle("T60 Thruster PWM–Axial Thrust Curves", fontsize=18, fontweight="bold")
    figure.text(
        0.5,
        0.945,
        "Axial thrust = measured body force · CAD shaft axis (FLU); shaded region = 1475–1525 µs zero-force deadband",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#475569",
    )
    figure.supxlabel("PWM pulse width (µs)", fontsize=12)
    figure.supylabel("Signed axial thrust (N)", fontsize=12)
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.09, top=0.88, wspace=0.12, hspace=0.22)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return pwm, axial_thrust


def plot_pwm_force_components(
    output: Path,
    samples: int = 1001,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Save the measured body-frame Fx, Fy, and Fz curves for all thrusters."""

    pwm = torch.linspace(
        AUV.thruster_pwm_center_us - AUV.thruster_pwm_half_range_us,
        AUV.thruster_pwm_center_us + AUV.thruster_pwm_half_range_us,
        samples,
        dtype=torch.float64,
    )
    pwm_by_thruster = pwm[:, None].expand(-1, len(AUV.thruster_labels))
    body_forces = thruster_body_forces_from_pwm_us(pwm_by_thruster)

    figure, axes = plt.subplots(2, 4, figsize=(18, 8.5), sharex=True, sharey=True)
    deadband_min = AUV.thruster_pwm_center_us - AUV.thruster_pwm_deadband_us
    deadband_max = AUV.thruster_pwm_center_us + AUV.thruster_pwm_deadband_us
    force_limit = 1.12 * float(body_forces.abs().max())

    for thruster_index, (axis, label, installation) in enumerate(
        zip(axes.flat, AUV.thruster_labels, _INSTALLATION_LABELS, strict=True)
    ):
        axis.axvspan(deadband_min, deadband_max, color="#94a3b8", alpha=0.22)
        axis.axhline(0.0, color="#64748b", linewidth=0.9)
        axis.axvline(AUV.thruster_pwm_center_us, color="#94a3b8", linewidth=0.9, linestyle=":")
        for component_index, (component_label, color) in enumerate(_COMPONENT_STYLES):
            axis.plot(
                pwm.numpy(),
                body_forces[:, thruster_index, component_index].numpy(),
                color=color,
                linewidth=2.2,
                label=component_label,
            )
        axis.set_title(f"{label}  |  {installation}", fontsize=11, fontweight="semibold")
        axis.set_xlim(float(pwm[0]), float(pwm[-1]))
        axis.set_ylim(-force_limit, force_limit)
        axis.set_xticks(torch.linspace(float(pwm[0]), float(pwm[-1]), 5).tolist())
        axis.grid(True, color="#e2e8f0", linewidth=0.8)
        axis.set_axisbelow(True)

    figure.suptitle("T60 Thruster PWM–Body Force Curves", fontsize=18, fontweight="bold")
    figure.text(
        0.5,
        0.945,
        "Measured FLU components (+X forward, +Y left, +Z up); shaded region = 1475–1525 µs zero-force deadband",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#475569",
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=3,
        frameon=False,
        fontsize=10.5,
    )
    figure.supxlabel("PWM pulse width (µs)", fontsize=12)
    figure.supylabel("Body-frame force (N)", fontsize=12)
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.09, top=0.85, wspace=0.12, hspace=0.22)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return pwm, body_forces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--axial-output",
        "--output",
        dest="axial_output",
        type=Path,
        default=Path("artifacts/thruster_pwm_axial_thrust_curves.png"),
        help="Axial-thrust output image path.",
    )
    parser.add_argument(
        "--components-output",
        type=Path,
        default=Path("artifacts/thruster_pwm_force_components.png"),
        help="Fx/Fy/Fz output image path.",
    )
    parser.add_argument("--samples", type=int, default=1001, help="Number of PWM samples.")
    args = parser.parse_args()

    _, axial_thrust = plot_pwm_curves(args.axial_output, args.samples)
    plot_pwm_force_components(args.components_output, args.samples)
    print(f"Saved {args.axial_output.resolve()}")
    print(f"Saved {args.components_output.resolve()}")
    pwm_min = AUV.thruster_pwm_center_us - AUV.thruster_pwm_half_range_us
    pwm_max = AUV.thruster_pwm_center_us + AUV.thruster_pwm_half_range_us
    print(f"Thruster   {pwm_min:.0f} us   {pwm_max:.0f} us")
    for index, label in enumerate(AUV.thruster_labels):
        print(f"{label:>8} {axial_thrust[0, index]:+9.3f} N {axial_thrust[-1, index]:+9.3f} N")


if __name__ == "__main__":
    main()
