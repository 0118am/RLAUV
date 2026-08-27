"""Compare the configured T60 force model with the full-precision OLS model."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from robot.dynamics.parameters import AUV
from robot.propulsion.curves import (
    measured_thruster_body_forces,
    thruster_body_forces_from_pwm_us,
)


PWM_OFFSET_LIMIT_US = 200.0

# Layout: (thruster, component, a_positive, b_positive, a_negative, b_negative).
# These are the full-precision values from ols_quadratic_pwm_flu_coefficients.csv
# supplied with the comparison request.
ACCURATE_COEFFICIENTS = np.asarray(
    (
        (
            (1.0479446747411169e-05, 0.00029049139889470002, 8.4474520804025243e-06, 0.0039640197202361),
            (-2.2416124822308705e-05, 9.1713158089046663e-05, -2.3115342868926689e-05, 0.0049514676540881996),
            (-4.8486422690078336e-05, -0.036265199090186197, 9.4273754131380441e-05, 0.0056177491461797003),
        ),
        (
            (-1.2374551071730197e-05, 0.0019289405138891, -3.4147700627779172e-05, 0.0085922479354993003),
            (-3.3125295417693135e-05, 0.0028634040004971001, -1.5334235466764995e-05, 0.0078817943418689994),
            (-9.2586109270650197e-05, -0.019839952204023199, -6.6593389013093715e-06, 0.042883242917725398),
        ),
        (
            (1.0479446747411169e-05, 0.00029049139889470002, 8.4474520804025243e-06, 0.0039640197202361),
            (2.2416124822308705e-05, -9.1713158089046663e-05, 2.3115342868926689e-05, -0.0049514676540881996),
            (-4.8486422690078336e-05, -0.036265199090186197, 9.4273754131380441e-05, 0.0056177491461797003),
        ),
        (
            (-1.2374551071730197e-05, 0.0019289405138891, -3.4147700627779172e-05, 0.0085922479354993003),
            (3.3125295417693135e-05, -0.0028634040004971001, 1.5334235466764995e-05, -0.0078817943418689994),
            (-9.2586109270650197e-05, -0.019839952204023199, -6.6593389013093715e-06, 0.042883242917725398),
        ),
        (
            (-2.8727300664564519e-05, -0.021064261061346699, -1.8545677155981932e-05, 0.0207220916590331),
            (-5.9984305658458156e-07, -0.0106913668168963, 1.7453036222353297e-05, -0.0067881453100422998),
            (2.5488577481102081e-07, -0.0020886827450356999, -1.7791792921878693e-06, -0.0084607241962699007),
        ),
        (
            (-2.8727300664564519e-05, -0.021064261061346699, -1.8545677155981932e-05, 0.0207220916590331),
            (5.9984305658458156e-07, 0.0106913668168963, -1.7453036222353297e-05, 0.0067881453100422998),
            (2.5488577481102081e-07, -0.0020886827450356999, -1.7791792921878693e-06, -0.0084607241962699007),
        ),
        (
            (3.0266152915549253e-05, 0.028350257707977399, -6.9661024358955455e-05, 0.0042667706930280998),
            (4.6992370553306008e-07, -0.0121290379475416, 1.6878772341498014e-05, -0.00772990265755),
            (2.3123665856172451e-05, -0.014248856239900101, 2.9396098190587793e-05, -0.020797881665092301),
        ),
        (
            (3.0266152915549253e-05, 0.028350257707977399, -6.9661024358955455e-05, 0.0042667706930280998),
            (-4.6992370553306008e-07, 0.0121290379475416, -1.6878772341498014e-05, 0.00772990265755),
            (2.3123665856172451e-05, -0.014248856239900101, 2.9396098190587793e-05, -0.020797881665092301),
        ),
    ),
    dtype=np.float64,
)

_COMPONENT_NAMES = ("Fx", "Fy", "Fz")
_OUTPUT_NAMES = (*_COMPONENT_NAMES, "|F|")
_ACCURATE_COLORS = ("#d55e00", "#009e73", "#0072b2", "#111827")


def accurate_thruster_forces(offset_us: np.ndarray) -> np.ndarray:
    """Evaluate the supplied full-precision model without any PWM sign flip."""

    offset = np.asarray(offset_us, dtype=np.float64)
    q_positive = np.maximum(offset - AUV.thruster_pwm_deadband_us, 0.0)[:, None, None]
    q_negative = np.maximum(-offset - AUV.thruster_pwm_deadband_us, 0.0)[:, None, None]
    a_positive = ACCURATE_COEFFICIENTS[:, :, 0]
    b_positive = ACCURATE_COEFFICIENTS[:, :, 1]
    a_negative = ACCURATE_COEFFICIENTS[:, :, 2]
    b_negative = ACCURATE_COEFFICIENTS[:, :, 3]
    return (
        a_positive * np.square(q_positive)
        + b_positive * q_positive
        + a_negative * np.square(q_negative)
        + b_negative * q_negative
    )


def _with_resultant(forces: np.ndarray) -> np.ndarray:
    return np.concatenate((forces, np.linalg.norm(forces, axis=-1, keepdims=True)), axis=-1)


def _configured_forces_at_offset(offset_us: np.ndarray) -> np.ndarray:
    pwm = torch.as_tensor(
        AUV.thruster_pwm_center_us + offset_us,
        dtype=torch.float64,
    )
    pwm_by_thruster = pwm[:, None].expand(-1, len(AUV.thruster_labels))
    return thruster_body_forces_from_pwm_us(pwm_by_thruster).numpy()


def _configured_forces_at_command(command: np.ndarray) -> np.ndarray:
    commands = torch.as_tensor(command, dtype=torch.float64)[:, None].expand(
        -1, len(AUV.thruster_labels)
    )
    return measured_thruster_body_forces(commands).numpy()


def _decorate_force_axis(axis: plt.Axes, x_min: float, x_max: float) -> None:
    axis.axhline(0.0, color="#94a3b8", linewidth=0.8)
    axis.grid(True, color="#e2e8f0", linewidth=0.7)
    axis.set_xlim(x_min, x_max)
    axis.set_axisbelow(True)


def plot_accurate_curves(output: Path, offset_us: np.ndarray, forces: np.ndarray) -> None:
    """Plot all three fitted components and their derived resultant magnitude."""

    outputs = _with_resultant(forces)
    figure, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True)
    for thruster_index, axis in enumerate(axes.flat):
        axis.axvspan(
            -AUV.thruster_pwm_deadband_us,
            AUV.thruster_pwm_deadband_us,
            color="#cbd5e1",
            alpha=0.35,
        )
        for output_index, (name, color) in enumerate(
            zip(_OUTPUT_NAMES, _ACCURATE_COLORS, strict=True)
        ):
            linestyle = ":" if name == "|F|" else "-"
            axis.plot(
                offset_us,
                outputs[:, thruster_index, output_index],
                color=color,
                linewidth=2.0,
                linestyle=linestyle,
                label=name,
            )
        _decorate_force_axis(axis, -PWM_OFFSET_LIMIT_US, PWM_OFFSET_LIMIT_US)
        axis.set_title(AUV.thruster_labels[thruster_index], fontweight="semibold")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.suptitle("Full-precision T60 three-component force curves", fontsize=18, fontweight="bold")
    figure.text(
        0.5,
        0.925,
        "Solid lines are independently fitted FLU components; |F| is derived from Fx/Fy/Fz only",
        ha="center",
        color="#475569",
    )
    figure.supxlabel("u = PWM_model - 1500 (us)")
    figure.supylabel("Force (N)")
    figure.subplots_adjust(left=0.06, right=0.985, bottom=0.08, top=0.86, wspace=0.18, hspace=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_comparison_grid(
    output: Path,
    x: np.ndarray,
    accurate_outputs: np.ndarray,
    configured_outputs: np.ndarray,
    *,
    title: str,
    subtitle: str,
    xlabel: str,
) -> None:
    """Plot accurate and configured curves for Fx, Fy, Fz, and derived |F|."""

    figure, axes = plt.subplots(8, 4, figsize=(18, 25), sharex=True)
    for thruster_index, thruster in enumerate(AUV.thruster_labels):
        for output_index, output_name in enumerate(_OUTPUT_NAMES):
            axis = axes[thruster_index, output_index]
            axis.plot(
                x,
                accurate_outputs[:, thruster_index, output_index],
                color="#0072b2",
                linewidth=2.0,
                label="accurate (full precision)",
            )
            axis.plot(
                x,
                configured_outputs[:, thruster_index, output_index],
                color="#d55e00",
                linewidth=1.8,
                linestyle="--",
                label="current configured runtime",
            )
            max_error = np.max(
                np.abs(
                    configured_outputs[:, thruster_index, output_index]
                    - accurate_outputs[:, thruster_index, output_index]
                )
            )
            axis.text(
                0.97,
                0.94,
                f"max |delta| = {max_error:.6g} N",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                color="#334155",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cbd5e1"},
            )
            _decorate_force_axis(axis, float(x[0]), float(x[-1]))
            if thruster_index == 0:
                axis.set_title(output_name, fontweight="semibold")
            if output_index == 0:
                axis.set_ylabel(f"{thruster}\nForce (N)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.suptitle(title, fontsize=18, fontweight="bold")
    figure.text(0.5, 0.967, subtitle, ha="center", color="#475569")
    figure.supxlabel(xlabel)
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.035, top=0.94, wspace=0.17, hspace=0.24)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_coefficient_comparison(output: Path) -> None:
    """Write both models in the same (ap, bp, am, bm) layout."""

    configured = np.asarray(AUV.thruster_force_curve_coefficients, dtype=np.float64)
    configured_canonical = np.transpose(configured, (0, 2, 1))
    coefficient_names = ("a_positive", "b_positive", "a_negative", "b_negative")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "thruster",
                "component",
                "coefficient",
                "accurate",
                "current_config_effective",
                "current_minus_accurate",
            )
        )
        for thruster_index, thruster in enumerate(AUV.thruster_labels):
            for component_index, component in enumerate(_COMPONENT_NAMES):
                for coefficient_index, coefficient_name in enumerate(coefficient_names):
                    accurate = ACCURATE_COEFFICIENTS[
                        thruster_index, component_index, coefficient_index
                    ]
                    current = configured_canonical[
                        thruster_index, component_index, coefficient_index
                    ]
                    writer.writerow(
                        (
                            thruster,
                            component,
                            coefficient_name,
                            repr(float(accurate)),
                            repr(float(current)),
                            repr(float(current - accurate)),
                        )
                    )


def write_curve_error_summary(
    output: Path,
    comparisons: tuple[tuple[str, np.ndarray, np.ndarray, np.ndarray], ...],
) -> None:
    """Write per-output errors for same-offset and same-command comparisons."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "comparison_basis",
                "thruster",
                "output",
                "max_abs_error_n",
                "x_at_max_error",
                "rmse_n",
                "accurate_at_x_min_n",
                "configured_at_x_min_n",
                "accurate_at_x_max_n",
                "configured_at_x_max_n",
            )
        )
        for basis, x, accurate, configured in comparisons:
            for thruster_index, thruster in enumerate(AUV.thruster_labels):
                for output_index, output_name in enumerate(_OUTPUT_NAMES):
                    error = configured[:, thruster_index, output_index] - accurate[
                        :, thruster_index, output_index
                    ]
                    maximum_index = int(np.argmax(np.abs(error)))
                    writer.writerow(
                        (
                            basis,
                            thruster,
                            output_name,
                            repr(float(abs(error[maximum_index]))),
                            repr(float(x[maximum_index])),
                            repr(float(np.sqrt(np.mean(np.square(error))))),
                            repr(float(accurate[0, thruster_index, output_index])),
                            repr(float(configured[0, thruster_index, output_index])),
                            repr(float(accurate[-1, thruster_index, output_index])),
                            repr(float(configured[-1, thruster_index, output_index])),
                        )
                    )


def _print_vector_error_table(
    heading: str,
    x: np.ndarray,
    accurate_forces: np.ndarray,
    configured_forces: np.ndarray,
) -> None:
    print(heading)
    print("Thruster  max vector error (N)  x at max    max |F| error (N)")
    accurate_resultant = np.linalg.norm(accurate_forces, axis=-1)
    configured_resultant = np.linalg.norm(configured_forces, axis=-1)
    for thruster_index, thruster in enumerate(AUV.thruster_labels):
        vector_error = np.linalg.norm(
            configured_forces[:, thruster_index] - accurate_forces[:, thruster_index],
            axis=-1,
        )
        maximum_index = int(np.argmax(vector_error))
        resultant_error = np.max(
            np.abs(
                configured_resultant[:, thruster_index]
                - accurate_resultant[:, thruster_index]
            )
        )
        print(
            f"{thruster:>8} {vector_error[maximum_index]:>21.9g}"
            f" {x[maximum_index]:>9.4g} {resultant_error:>20.9g}"
        )


def main() -> None:
    output_dir = Path("artifacts/thruster_curve_comparison")
    offset_us = np.arange(-PWM_OFFSET_LIMIT_US, PWM_OFFSET_LIMIT_US + 1.0, dtype=np.float64)
    command = np.linspace(-1.0, 1.0, offset_us.size, dtype=np.float64)

    accurate_same_offset_forces = accurate_thruster_forces(offset_us)
    configured_same_offset_forces = _configured_forces_at_offset(offset_us)
    accurate_same_command_forces = accurate_thruster_forces(PWM_OFFSET_LIMIT_US * command)
    configured_same_command_forces = _configured_forces_at_command(command)

    accurate_same_offset = _with_resultant(accurate_same_offset_forces)
    configured_same_offset = _with_resultant(configured_same_offset_forces)
    accurate_same_command = _with_resultant(accurate_same_command_forces)
    configured_same_command = _with_resultant(configured_same_command_forces)

    plot_accurate_curves(
        output_dir / "accurate_three_component_curves.png",
        offset_us,
        accurate_same_offset_forces,
    )
    plot_comparison_grid(
        output_dir / "same_pwm_offset_configured_vs_accurate.png",
        offset_us,
        accurate_same_offset,
        configured_same_offset,
        title="Configured vs accurate T60 force curves at the same physical PWM offset",
        subtitle=(
            "u is never sign-flipped; this view isolates branch ordering and coefficient precision "
            "over the valid -200...200 us range"
        ),
        xlabel="u = PWM_model - 1500 (us)",
    )
    plot_comparison_grid(
        output_dir / "same_normalized_command_runtime_vs_accurate.png",
        command,
        accurate_same_command,
        configured_same_command,
        title="Current runtime vs accurate T60 model at the same normalized command",
        subtitle=(
            f"Accurate mapping: u = 200 command; current mapping: u = "
            f"{AUV.thruster_pwm_half_range_us:g} command"
        ),
        xlabel="Normalized command",
    )
    write_coefficient_comparison(output_dir / "coefficient_comparison.csv")
    write_curve_error_summary(
        output_dir / "curve_error_summary.csv",
        (
            ("same_pwm_offset_us", offset_us, accurate_same_offset, configured_same_offset),
            ("same_normalized_command", command, accurate_same_command, configured_same_command),
        ),
    )

    print(f"Accurate PWM offset range: +/-{PWM_OFFSET_LIMIT_US:g} us")
    print(f"Current configured PWM half range: +/-{AUV.thruster_pwm_half_range_us:g} us")
    _print_vector_error_table(
        "\nSame physical PWM offset:",
        offset_us,
        accurate_same_offset_forces,
        configured_same_offset_forces,
    )
    _print_vector_error_table(
        "\nSame normalized command:",
        command,
        accurate_same_command_forces,
        configured_same_command_forces,
    )
    print(f"\nSaved comparison artifacts to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
