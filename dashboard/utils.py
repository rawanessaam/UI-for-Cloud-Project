"""
utils.py — Utility functions for the Vision Evolution Platform dashboard.
Handles API communication, CSV loading, data cleaning, and error management.
"""

import requests
import pandas as pd
import numpy as np
import io
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

API_URL = "http://13.51.70.11:8000/predict"  # Replace with your deployed cloud API endpoint
REQUEST_TIMEOUT = 30       # seconds

# ─── API Communication ─────────────────────────────────────────────────────────


def _mock_prediction(image_file):
    import random
    classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    return {
        "prediction": random.choice(classes),
        "confidence": round(random.uniform(0.82, 0.99), 4),
        "model": "Mock Model"
    }


def predict_image(image_file):
    if USE_MOCK:
        return _mock_prediction(image_file)

    try:
        files = {"file": (image_file.name, image_file.getvalue(), image_file.type)}
        r = requests.post(API_URL, files=files, timeout=30)
        r.raise_for_status()
        return r.json()

    except Exception as e:
        return {"error": str(e)}

    except requests.exceptions.ConnectionError:
        return {"error": "Cannot reach the API server. Check your network or API URL."}
    except requests.exceptions.Timeout:
        return {"error": f"Request timed out after {REQUEST_TIMEOUT}s."}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error {e.response.status_code}: {e.response.text[:200]}"}
    except ValueError:
        return {"error": "API returned invalid JSON."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}





# ─── CSV / Data Loading ────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent


def load_results():
    df = pd.read_csv("evaluation/results.csv")

    # 🔧 normalize column names (VERY IMPORTANT)
    df.columns = [c.strip().lower() for c in df.columns]

    # 🔁 map possible variations → standard schema
    rename_map = {
        "exp_id": "experiment_id",
        "experiment": "experiment_id",
        "model_name": "model",
        "acc": "accuracy",
        "accuracy_score": "accuracy",
    }

    df.rename(columns=rename_map, inplace=True)

    return df


def load_ga_log(path: str = None) -> pd.DataFrame:
    """
    Load GA convergence log CSV into a DataFrame.
    Returns an empty DataFrame with expected columns on failure.
    """
    csv_path = path or DATA_DIR / "ga_log.csv"
    return _safe_load_csv(
        csv_path,
        fallback_columns=[
            "generation", "best_fitness", "avg_fitness", "worst_fitness",
            "population_size", "mutation_rate", "crossover_rate",
            "selected_features", "diversity_score"
        ]
    )


def _safe_load_csv(path, fallback_columns: list) -> pd.DataFrame:
    """Load a CSV file safely, returning a structured empty DataFrame on error."""
    try:
        df = pd.read_csv(path)
        return _clean_dataframe(df)
    except FileNotFoundError:
        return pd.DataFrame(columns=fallback_columns)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=fallback_columns)
    except Exception:
        return pd.DataFrame(columns=fallback_columns)


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, strip whitespace from string columns, reset index."""
    df = df.drop_duplicates().reset_index(drop=True)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


# ─── Summary Statistics ────────────────────────────────────────────────────────

def compute_model_summary(results: pd.DataFrame) -> dict:

    if results.empty:
        return {
            "best_accuracy": 0.0,
            "best_model": "N/A",
            "avg_baseline_acc": 0.0,
            "avg_optimized_acc": 0.0,
            "improvement_pct": 0.0,
            "total_experiments": 0
        }

    # safest mapping
    base_df = results[results["model"].str.contains("Baseline", case=False, na=False)]
    opt_df  = results[results["model"].str.contains("GA", case=False, na=False)]

    best_row = results.loc[results["accuracy"].idxmax()]

    avg_base = base_df["accuracy"].mean() if not base_df.empty else 0.0
    avg_opt  = opt_df["accuracy"].mean() if not opt_df.empty else 0.0

    improvement = ((avg_opt - avg_base) / avg_base * 100) if avg_base > 0 else 0.0

    return {
        "best_accuracy": round(best_row["accuracy"], 4),
        "best_model": best_row["model"],
        "avg_baseline_acc": round(avg_base, 4),
        "avg_optimized_acc": round(avg_opt, 4),
        "improvement_pct": round(improvement, 2),
        "total_experiments": len(results)
    }

def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to UTF-8 CSV bytes for st.download_button."""
    return df.to_csv(index=False).encode("utf-8")
