from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


CONFIGURATIONS = [
    {"run_name": "Run 1", "max_depth": 3, "learning_rate": 0.1},
    {"run_name": "Run 2", "max_depth": 5, "learning_rate": 0.05},
    {"run_name": "Run 3", "max_depth": 7, "learning_rate": 0.01},
]


def main() -> None:
    # Use one database file next to this script, regardless of the shell's cwd.
    project_dir = Path(__file__).resolve().parent
    tracking_db = project_dir / "mlflow.db"
    tracking_uri = f"sqlite:///{tracking_db.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("California Housing - Gradient Boosting")

    print(f"Loading California Housing dataset; cache/network may be needed on first run...")
    housing = fetch_california_housing(as_frame=False)
    X, y = housing.data, housing.target

    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.20,random_state=72)

    print(f"Dataset rows: {len(X)}")
    print(f"Training rows: {len(X_train)} | Validation rows: {len(X_val)}")
    print(f"MLflow tracking URI: {tracking_uri}")

    results = []

    for config in CONFIGURATIONS:
        run_name = config["run_name"]
        max_depth = config["max_depth"]
        learning_rate = config["learning_rate"]

        with mlflow.start_run(
            run_name=f"{run_name} - depth={max_depth} - lr={learning_rate}"
        ) as active_run:
            model = GradientBoostingRegressor(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=100,
                random_state=72,
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)

            # The target is expressed in hundreds of thousands of US dollars.
            metrics = {
                "rmse": float(mean_squared_error(y_val, predictions) ** 0.5),
                "mae": float(mean_absolute_error(y_val, predictions)),
                "r2": float(r2_score(y_val, predictions)),
            }
            params = {
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "n_estimators": 100,
                "model_random_state": 72,
                "split_random_state": 72,
                "validation_size": 0.20,
            }

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("dataset", "California Housing")
            mlflow.set_tag("evaluation_split", "validation")
            mlflow.set_tag("model_family", "GradientBoostingRegressor")

            mlflow.sklearn.log_model(sk_model=model,name="model",skops_trusted_types=["sklearn.tree._tree.Tree"])

            results.append(
                {
                    "run_name": run_name,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    **metrics,
                    "run_id": active_run.info.run_id,
                }
            )
            print(
                f"{run_name}: RMSE={metrics['rmse']:.5f}, "
                f"MAE={metrics['mae']:.5f}, R2={metrics['r2']:.5f}"
            )





    print("\nValidation results (lower RMSE is better):")
    print(f"{'Run':<8} {'max_depth':>10} {'learning_rate':>15} {'RMSE':>12} {'MAE':>12} {'R2':>12}")
    for row in results:
        print(
            f"{row['run_name']:<8} {row['max_depth']:>10} "
            f"{row['learning_rate']:>15.3f} {row['rmse']:>12.5f} "
            f"{row['mae']:>12.5f} {row['r2']:>12.5f}"
        )

    best = min(results, key=lambda row: row["rmse"])
    print(
        f"\nBest run by validation RMSE: {best['run_name']} "
        f"(max_depth={best['max_depth']}, learning_rate={best['learning_rate']}, "
        f"RMSE={best['rmse']:.5f})"
    )


if __name__ == "__main__":
    main()
