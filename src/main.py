from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

from .cellular_automaton import CAResult, simulate_ca
from .compartmental_model import simulate_ode, ode_metrics
from .config import ODEParameters, CAParameters, ode_scenarios, ca_scenarios
from .visualization import (
    save_ode_curves,
    save_environment_curve,
    save_scenario_bars,
    save_ca_time_series,
    save_reference_ca_figures,
)


def run_ode(output_dir: Path) -> pd.DataFrame:
    metrics_records = []
    data_by_scenario: Dict[str, pd.DataFrame] = {}

    for name, parameters in ode_scenarios(ODEParameters()).items():
        df = simulate_ode(parameters)
        df.to_csv(output_dir / f"ode_{name}.csv", index=False)
        data_by_scenario[name] = df
        metrics_records.append(ode_metrics(df, name, parameters))

    metrics = pd.DataFrame(metrics_records)
    baseline = float(
        metrics.loc[
            metrics["cenario"] == "referencia",
            "casos_sintomaticos_acumulados",
        ].iloc[0]
    )
    metrics["reducao_percentual_casos"] = (
        (baseline - metrics["casos_sintomaticos_acumulados"]) / baseline * 100.0
    )
    metrics.to_csv(output_dir / "metricas_ode.csv", index=False)

    reference = data_by_scenario["referencia"]
    save_ode_curves(reference, output_dir / "ode_curvas_referencia.png")
    save_environment_curve(
        reference,
        output_dir / "ode_contaminacao_referencia.png",
    )
    save_scenario_bars(
        metrics,
        "casos_sintomaticos_acumulados",
        "Casos sintomáticos acumulados por cenário — modelo ODE",
        "Casos acumulados",
        output_dir / "ode_comparacao_casos.png",
    )
    save_scenario_bars(
        metrics,
        "pico_sintomaticos",
        "Pico de sintomáticos por cenário — modelo ODE",
        "Indivíduos sintomáticos",
        output_dir / "ode_comparacao_picos.png",
    )
    return metrics


def run_ca(output_dir: Path) -> pd.DataFrame:
    metrics_records = []
    results: Dict[str, CAResult] = {}

    for name, parameters in ca_scenarios(CAParameters()).items():
        result = simulate_ca(parameters, name)
        result.history.to_csv(output_dir / f"ca_{name}.csv", index=False)
        results[name] = result
        metrics_records.append(result.metrics)

    metrics = pd.DataFrame(metrics_records)
    baseline = float(
        metrics.loc[
            metrics["cenario"] == "referencia",
            "casos_sintomaticos_acumulados",
        ].iloc[0]
    )
    metrics["reducao_percentual_casos"] = (
        (baseline - metrics["casos_sintomaticos_acumulados"]) / baseline * 100.0
    )
    metrics.to_csv(output_dir / "metricas_ca.csv", index=False)

    reference = results["referencia"]
    save_ca_time_series(reference.history, output_dir / "ca_curvas_referencia.png")
    save_reference_ca_figures(
        reference,
        CAParameters().contamination_threshold,
        output_dir,
    )
    save_scenario_bars(
        metrics,
        "casos_sintomaticos_acumulados",
        "Casos sintomáticos acumulados por cenário — autômato",
        "Casos acumulados",
        output_dir / "ca_comparacao_casos.png",
    )
    save_scenario_bars(
        metrics,
        "area_contaminada_maxima",
        "Área contaminada máxima por cenário — autômato",
        "Número de células",
        output_dir / "ca_comparacao_area.png",
    )
    return metrics


def write_summary(
    output_dir: Path,
    ode_metrics_df: pd.DataFrame | None,
    ca_metrics_df: pd.DataFrame | None,
) -> None:
    lines = ["RESUMO AUTOMÁTICO DOS EXPERIMENTOS", "=" * 40, ""]
    if ode_metrics_df is not None:
        lines += ["MODELO COMPARTIMENTAL", ode_metrics_df.to_string(index=False), ""]
    if ca_metrics_df is not None:
        lines += ["AUTÔMATO CELULAR", ca_metrics_df.to_string(index=False), ""]
    (output_dir / "resumo_resultados.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulação da leptospirose por SEIR ambiental e autômatos celulares."
    )
    parser.add_argument(
        "--model",
        choices=["all", "ode", "ca"],
        default="all",
    )
    parser.add_argument(
        "--output",
        default="results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ode_df = run_ode(output_dir) if args.model in {"all", "ode"} else None
    ca_df = run_ca(output_dir) if args.model in {"all", "ca"} else None

    write_summary(output_dir, ode_df, ca_df)
    (output_dir / "execucao.json").write_text(
        json.dumps(
            {
                "modelo_executado": args.model,
                "diretorio_saida": str(output_dir.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("Simulação concluída.")
    print(f"Resultados salvos em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
