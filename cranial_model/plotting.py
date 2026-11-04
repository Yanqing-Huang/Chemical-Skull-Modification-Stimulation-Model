"""Publication-style figures generated only from the theoretical simulation output."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .core import LAYER_NAMES, ScenarioResult, layer_calcium_remaining_fraction
from .diagnostics import thiele_biot_rows


COLORS = {
    "outer_table": "#2166AC",
    "diploe": "#E08214",
    "inner_table": "#1B9E77",
    "acetic_acid": "#762A83",
    "edta": "#2166AC",
    "urea": "#E08214",
}
DOSE_COLORS = ["#666666", "#4393C3", "#1B9E77", "#D6604D"]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
        }
    )


def _panel_label(axis, label: str) -> None:
    axis.text(-0.15, 1.05, label, transform=axis.transAxes, fontweight="bold", fontsize=11)


def _layer_boundaries(axis, scenario: ScenarioResult) -> None:
    for edge in (0.43, 0.63):
        axis.axvline(edge, color="0.75", lw=0.7, ls=":")


def _save(figure, output: Path) -> None:
    figure.savefig(output, facecolor="white")
    plt.close(figure)


def plot_model_schematic(values: dict, output: Path) -> None:
    configure_style()
    figure = plt.figure(figsize=(10.2, 4.5))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.24)
    axis = figure.add_subplot(grid[0, 0])
    axis.set_xlim(-0.25, 1.15)
    axis.set_ylim(-0.1, 1.25)
    axis.axis("off")
    x0 = 0.12
    total_width = 0.72
    y0, height = 0.16, 0.66
    position = x0
    for layer in values["geometry"]["layers"]:
        width = total_width * layer["thickness_mm"] / values["geometry"]["total_thickness_mm"]
        name = layer["name"]
        axis.add_patch(
            Rectangle((position, y0), width, height, facecolor=COLORS[name], alpha=0.22, ec=COLORS[name], lw=1.4)
        )
        axis.text(position + width / 2, y0 + height / 2, name.replace("_", "\n"), ha="center", va="center", color=COLORS[name], fontweight="bold")
        position += width
    axis.text(0.48, 0.035, "Layer thickness: outer 0.43  |  diploe 0.20  |  inner 0.17 mm", ha="center", va="center", fontsize=8)
    axis.annotate("External bath\nfilm + concentration", xy=(x0, 0.5), xytext=(-0.20, 0.5), arrowprops={"arrowstyle": "->", "lw": 1.2}, ha="center", va="center")
    axis.text(1.01, 0.50, "Inner boundary\nzero flux", ha="center", va="center", fontsize=8)
    axis.plot([0.86, 0.86], [0.22, 0.78], color="black", lw=2)
    axis.text(0.36, 1.10, "1  Acid (50%)   →   2  EDTA (pH 7.4)   →   3  Urea (sat.)", color="0.20", fontweight="bold", ha="center", fontsize=9)
    axis.text(0.36, 0.94, "Rinse: retain q and porosity; reset free liquid concentration to zero", ha="center", fontsize=8, color="0.25")

    equations = figure.add_subplot(grid[0, 1])
    equations.axis("off")
    equations.text(
        0.0,
        0.98,
        "Conserved pore-liquid reagent",
        fontweight="bold",
        va="top",
        fontsize=11,
    )
    equations.text(0.0, 0.84, r"$\frac{\partial(\varepsilon c_j)}{\partial t}=\frac{\partial}{\partial x}\left(D_{e,j}\frac{\partial c_j}{\partial x}\right)-\nu_j k_{r,j}c_jq$", va="top", fontsize=11)
    equations.text(0.0, 0.68, r"$\frac{\partial q}{\partial t}=-k_{r,j}c_jq,\qquad D_{e,j}=D_{b,j}\varepsilon/\tau$", va="top", fontsize=11)
    equations.text(0.0, 0.53, r"$\varepsilon=\varepsilon_0+\alpha(q_0-q)$", va="top", fontsize=11)
    equations.text(0.0, 0.38, r"$J_0=\frac{C_b-c_0}{1/k_f+\Delta x_0/(2D_{e,0})},\qquad J_N=0$", va="top", fontsize=11)
    equations.text(0.0, 0.20, "Acid and EDTA consume calcium; urea has k$_r$ = 0.\nSCM and Thiele/Biot relations are independent diagnostics.\nElectrical resistance is evaluated after standardized rinsing.", va="top", fontsize=9, linespacing=1.5)
    _save(figure, output)


def plot_three_stage_results(scenarios: list[ScenarioResult], output: Path) -> None:
    configure_style()
    baseline = min(scenarios, key=lambda item: abs(item.edta_concentration_mol_per_l - 0.1))
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.4))
    figure.subplots_adjust(hspace=0.38, wspace=0.28)
    x_mm = baseline.grid.centers * 1000.0

    axis = axes[0, 0]
    axis.plot(x_mm, baseline.stages["acetic_acid"].calcium[-1] / baseline.grid.q_reference, color="black", ls="--", lw=1.4, label="After acid")
    for index, scenario in enumerate(scenarios):
        result = scenario.stages["edta"]
        axis.plot(x_mm, result.calcium[-1] / scenario.grid.q_reference, color=DOSE_COLORS[index], lw=1.5, label=f"{scenario.edta_concentration_mol_per_l:.2f} M EDTA")
    _layer_boundaries(axis, baseline)
    axis.set(xlabel="Depth from outer surface (mm)", ylabel="Remaining calcium q / q$_0$", ylim=(0, 1.04))
    axis.legend(fontsize=7, ncol=2)
    _panel_label(axis, "A")

    axis = axes[0, 1]
    for index, scenario in enumerate(scenarios):
        result = scenario.stages["edta"]
        axis.plot(x_mm, result.porosity[-1], color=DOSE_COLORS[index], lw=1.5, label=f"{scenario.edta_concentration_mol_per_l:.2f} M")
    _layer_boundaries(axis, baseline)
    axis.set(xlabel="Depth from outer surface (mm)", ylabel="Porosity after EDTA")
    axis.legend(fontsize=8)
    _panel_label(axis, "B")

    axis = axes[1, 0]
    for stage_name in ("acetic_acid", "edta", "urea"):
        result = baseline.stages[stage_name]
        bulk = baseline.values["stages"][stage_name]["bulk_concentration_mol_per_l"] * 1000.0
        axis.plot(x_mm, result.concentration[-1] / bulk, color=COLORS[stage_name], lw=1.6, label=stage_name.replace("_", " ").title())
    _layer_boundaries(axis, baseline)
    axis.set(xlabel="Depth from outer surface (mm)", ylabel="Stage-end pore concentration c / C$_b$", ylim=(-0.02, 1.04))
    axis.legend(fontsize=8)
    _panel_label(axis, "C")

    axis = axes[1, 1]
    for index, scenario in enumerate(scenarios):
        all_time = []
        all_remaining = []
        for stage_name in ("acetic_acid", "edta", "urea"):
            result = scenario.stages[stage_name]
            remaining = np.sum(result.calcium * scenario.grid.widths[None, :], axis=1) / np.sum(scenario.grid.q_reference * scenario.grid.widths)
            all_time.extend(result.global_time_s.tolist())
            all_remaining.extend(remaining.tolist())
        axis.plot(np.asarray(all_time) / 60.0, all_remaining, color=DOSE_COLORS[index], lw=1.5, label=f"{scenario.edta_concentration_mol_per_l:.2f} M")
    acid_end = baseline.values["stages"]["acetic_acid"]["duration_s"] / 60.0
    edta_end = acid_end + baseline.values["stages"]["edta"]["duration_s"] / 60.0
    axis.axvline(acid_end, color="0.65", lw=0.8, ls=":")
    axis.axvline(edta_end, color="0.65", lw=0.8, ls=":")
    axis.text(acid_end / 2, 1.005, "Acid", ha="center", va="bottom", fontsize=8)
    axis.text((acid_end + edta_end) / 2, 1.005, "EDTA", ha="center", va="bottom", fontsize=8)
    axis.text(edta_end + (baseline.values["stages"]["urea"]["duration_s"] / 60.0) / 2, 1.005, "Urea", ha="center", va="bottom", fontsize=8)
    axis.set(xlabel="Sequential process time (min)", ylabel="Whole-skull calcium remaining", ylim=(0.90, 1.015))
    axis.legend(fontsize=8, loc="lower left")
    _panel_label(axis, "D")
    _save(figure, output)


def plot_dose_and_resistance(scenarios: list[ScenarioResult], output: Path) -> None:
    configure_style()
    doses = np.array([scenario.edta_concentration_mol_per_l for scenario in scenarios])
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.4))
    figure.subplots_adjust(hspace=0.38, wspace=0.30)

    axis = axes[0, 0]
    acid_loss = []
    edta_loss = []
    for scenario in scenarios:
        native = np.sum(scenario.grid.q_reference * scenario.grid.widths)
        acid = scenario.stages["acetic_acid"].final_calcium
        edta = scenario.stages["edta"].final_calcium
        acid_loss.append(100 * np.sum((scenario.grid.q_reference - acid) * scenario.grid.widths) / native)
        edta_loss.append(100 * np.sum((scenario.grid.q_reference - edta) * scenario.grid.widths) / native)
    axis.plot(doses, acid_loss, "o--", color=COLORS["acetic_acid"], label="After acid")
    axis.plot(doses, edta_loss, "o-", color=COLORS["edta"], label="After EDTA")
    axis.set(xlabel="EDTA concentration (mol/L)", ylabel="Whole-skull calcium loss (%)")
    axis.legend()
    _panel_label(axis, "A")

    axis = axes[0, 1]
    for layer in LAYER_NAMES:
        remaining = [100 * layer_calcium_remaining_fraction(scenario, "edta", layer) for scenario in scenarios]
        axis.plot(doses, remaining, "o-", color=COLORS[layer], label=layer.replace("_", " ").title())
    axis.set(xlabel="EDTA concentration (mol/L)", ylabel="Layer calcium remaining (%)", ylim=(84, 101))
    axis.legend(fontsize=8)
    _panel_label(axis, "B")

    axis = axes[1, 0]
    for stage_name, marker in (("urea", ":"), ("acetic_acid", "s--"), ("edta", "o-")):
        resistance = [scenario.area_resistance(scenario.stages[stage_name].final_calcium) / scenario.native_area_resistance_ohm_m2 for scenario in scenarios]
        axis.plot(doses, resistance, marker, color=COLORS[stage_name], label=stage_name.replace("_", " ").title(), zorder=3 if stage_name == "edta" else 2)
    axis.set(xlabel="EDTA concentration (mol/L)", ylabel="Post-rinse area resistance / native")
    axis.text(0.04, 0.05, "Urea overlays EDTA: no additional calcium loss\nin the baseline nonreactive-urea model.", transform=axis.transAxes, fontsize=8, color="0.3")
    axis.legend(fontsize=8)
    _panel_label(axis, "C")

    axis = axes[1, 1]
    nonzero = [scenario for scenario in scenarios if scenario.edta_concentration_mol_per_l > 0]
    outer_t50 = []
    for scenario in nonzero:
        rows = thiele_biot_rows(scenario)
        row = next(row for row in rows if row["stage"] == "edta" and row["layer"] == "outer_table")
        outer_t50.append(float(row["outer_table_constant_bath_scm_t50_s"]) / 60.0)
    axis.plot([scenario.edta_concentration_mol_per_l for scenario in nonzero], outer_t50, "o-", color=COLORS["edta"])
    axis.set_yscale("log")
    axis.set(xlabel="EDTA concentration (mol/L)", ylabel="Outer-table SCM t$_{50}$ (min)")
    axis.text(0.04, 0.05, "Constant-bath sharp-front estimate;\nnot a replacement for the three-layer PDE.", transform=axis.transAxes, fontsize=8, color="0.3")
    _panel_label(axis, "D")
    _save(figure, output)
