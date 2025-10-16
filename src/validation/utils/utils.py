from tabulate import tabulate

def display_validation_results(results: dict):
    """Affiche les résultats de validation dans un format structuré.
    - Les grandes valeurs sont en notation scientifique (e-n)
    - Les valeurs du fit restent en décimal normal
    """

    def format_val(v, sci_threshold=1e6):
        """Formate les valeurs : e-n si grand/petit, sinon décimal normal."""
        if v is None:
            return "-"
        try:
            if isinstance(v, (int, float)):
                if abs(v) >= sci_threshold or (abs(v) > 0 and abs(v) < 1e-3):
                    return f"{v:.3e}"
                else:
                    return f"{v:.6f}".rstrip("0").rstrip(".")
            return str(v)
        except Exception:
            return str(v)

    print("\n=== Résumé de la validation ===\n")
    print(f"Model type : {results.get('model_type')}")
    print(f"Model spec : {results.get('model_spec')}")
    print(f"Checkpoint : {results.get('checkpoint_path')}")
    print(f"Data path  : {results.get('data_path')}")
    print(f"Samples évalués : {results.get('samples_evaluated')} / {results.get('points_evaluated')} points\n")

    # Métriques globales
    global_metrics = [
        ["MAE", results.get("global_mae")],
        ["MSE", results.get("global_mse")],
        ["RMSE", results.get("global_rmse")],
        ["R²", results.get("global_r2")],
    ]
    global_metrics_fmt = [[k, format_val(v)] for k, v in global_metrics]
    print("=== Métriques globales ===")
    print(tabulate(global_metrics_fmt, headers=["Metric", "Value"], tablefmt="grid"))
    print()

    # Métriques de fit (affichage décimal normal)
    fit_metrics = [
        ["Fit %", results.get("fit_percentage")],
        ["Fit pred samples", results.get("fit_pred_samples")],
        ["Fit pred failures", results.get("fit_pred_failures")],
        ["Fit concentration MAE ratio", results.get("fit_concentration_mae_ratio")],
        ["Fit diameter MAE ratio", results.get("fit_diameter_mae_ratio")],
        ["Fit length MAE ratio", results.get("fit_length_mae_ratio")],
    ]
    fit_metrics_fmt = [[k, format_val(v, sci_threshold=1e9)] for k, v in fit_metrics]
    print("=== Détails du fit ===")
    print(tabulate(fit_metrics_fmt, headers=["Paramètre", "Valeur"], tablefmt="grid"))
    print()

    # Erreurs par variable — ici on autorise la notation scientifique
    variables = ["concentration", "diameter_nm", "length_nm"]
    metrics_table = []
    for var in variables:
        metrics_table.append([
            var,
            format_val(results.get(f"mae_{var}_pred")),
            format_val(results.get(f"mae_{var}_true")),
            format_val(results.get(f"mse_{var}_pred")),
            format_val(results.get(f"mse_{var}_true")),
            format_val(results.get(f"rmse_{var}_pred")),
            format_val(results.get(f"rmse_{var}_true")),
        ])
    print("=== Erreurs par variable ===")
    print(tabulate(metrics_table,
                   headers=["Variable", "MAE_pred", "MAE_true", "MSE_pred", "MSE_true", "RMSE_pred", "RMSE_true"],
                   tablefmt="grid"))
    print()

    # Fichiers
    print("=== Fichiers associés ===")
    print(f"Résumé : {results.get('summary_path')}")
    print(f"Détails fit : {results.get('fit_details_path')}")
    print(f"Détails reconstruction : {results.get('reconstruction_details_path')}")
    print(f"YAML : {results.get('yaml_path')}")
    print()
    print(f"Reconstruction status : {results.get('reconstruction_status')}")
    print(f"Random state : {results.get('random_state')}")
