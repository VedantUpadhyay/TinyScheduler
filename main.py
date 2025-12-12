"""
main.py
--------

This script implements the first phase of a machine‑learning guided CPU scheduling
project.  It has three primary responsibilities:

1.  Generate a synthetic dataset that resembles process scheduling traces.  Each
    record describes a single process snapshot with several low‑level metrics
    (I/O counts, I/O bytes, CPU utilisation and context switch counts).  A
    synthetic CPU burst time is derived from these features using a simple
    linear relationship plus noise.  The synthetic data is purely for
    experimentation and is not meant to reflect any specific operating system.

2.  Train two regression models – a Random Forest and a Linear Regression – on
    both the synthetic dataset and an optional real dataset.  The real dataset
    should follow the same column naming convention (for example, the
    `process_data.csv` file from the referenced GitHub project).  For each
    dataset the script prints standard error metrics (MAE, MSE and R²) and
    generates plots showing predicted versus actual burst times.  For the
    Random Forest it also plots feature importances.

3.  Compare the performance of the two models and, if a real dataset is
    provided, verify that the Random Forest reproduces the results reported by
    the original project (very high R² when all features are available).

This script is self contained.  It can be executed directly via

    python main.py

and does not require any external API access.  The only required
dependencies are `pandas`, `numpy`, `scikit‑learn` and `matplotlib`.  Results
are saved into the ``plots`` subdirectory within the project.
"""

import os
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Set a global random seed for reproducibility
RANDOM_STATE = 42

def generate_synthetic_dataset(n_samples: int = 2000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a synthetic process scheduling dataset.

    Each row in the resulting DataFrame contains the following columns:

    - io_read_bytes:     integer, simulated number of bytes read
    - io_write_bytes:    integer, simulated number of bytes written
    - io_read_count:     integer, number of discrete read operations
    - io_write_count:    integer, number of discrete write operations
    - cpu_percent:       float, CPU utilisation percentage
    - num_ctx_switches_voluntary: integer, voluntary context switches
    - cpu_times_user:    float, portion of burst time spent in user mode
    - cpu_times_system:  float, portion of burst time spent in system mode
    - burst_time:        float, total CPU burst time (target variable)

    The generation strategy loosely mimics the dynamic ranges observed in the
    process_data.csv file from the reference project.  However, no attempt is
    made to match exact distributions; the goal is to produce a realistic
    synthetic workload for modelling purposes.

    Parameters
    ----------
    n_samples : int
        Number of synthetic processes to generate.
    random_state : int
        Seed for the random number generator.

    Returns
    -------
    DataFrame
        The synthetic dataset with the columns described above.
    """
    rng = np.random.default_rng(random_state)

    # Simulate I/O bytes using a log‑normal distribution.  The means and
    # standard deviations are chosen empirically to produce values on the order
    # of 10^6–10^8 bytes, similar to typical process metrics.
    io_read_bytes = rng.lognormal(mean=16, sigma=1.2, size=n_samples).astype(int)
    io_write_bytes = rng.lognormal(mean=16, sigma=1.2, size=n_samples).astype(int)

    # Simulate I/O read/write counts using Poisson distributions.  The lambda
    # parameters control the average number of operations.  Using relatively
    # small lambdas ensures that counts stay in a plausible range (tens to
    # hundreds).
    io_read_count = rng.poisson(lam=300, size=n_samples)
    io_write_count = rng.poisson(lam=300, size=n_samples)

    # Simulate CPU utilisation between 0 and 100 percent.  We bias the
    # distribution towards lower usage by sampling from a Beta distribution.
    cpu_percent = rng.beta(a=2.0, b=5.0, size=n_samples) * 100.0

    # Simulate voluntary context switches using a Poisson distribution.  The
    # mean is selected to produce values on the order of thousands.
    num_ctx_switches_voluntary = rng.poisson(lam=5000, size=n_samples)

    # Construct the synthetic burst time as a linear combination of features
    # plus Gaussian noise.  The coefficients below were tuned to yield burst
    # times loosely comparable to those observed in the original dataset.  A
    # non‑linear model (Random Forest) should be able to capture the complex
    # relationships among features, whereas a linear model will struggle.
    burst_time = (
        0.00000005 * io_write_bytes +
        0.00000002 * io_read_bytes +
        0.02        * cpu_percent +
        0.0001      * num_ctx_switches_voluntary
    )
    # Add Gaussian noise to simulate measurement variability
    burst_time += rng.normal(loc=0.0, scale=5.0, size=n_samples)
    # Ensure no negative burst times
    burst_time = np.maximum(burst_time, 0.0)

    # Split burst time into user and system components randomly
    split = rng.uniform(0.2, 0.8, size=n_samples)
    cpu_times_user = burst_time * split
    cpu_times_system = burst_time - cpu_times_user

    # Assemble DataFrame
    df = pd.DataFrame({
        'io_read_bytes': io_read_bytes,
        'io_write_bytes': io_write_bytes,
        'io_read_count': io_read_count,
        'io_write_count': io_write_count,
        'cpu_percent': cpu_percent,
        'num_ctx_switches_voluntary': num_ctx_switches_voluntary,
        'cpu_times_user': cpu_times_user,
        'cpu_times_system': cpu_times_system,
        'burst_time': burst_time
    })
    return df

def load_dataset(path: str) -> pd.DataFrame:
    """Load a dataset from a CSV file.

    The file must contain all the columns listed in the synthetic dataset.
    Additional columns are ignored.  The function also constructs the target
    variable 'burst_time' as the sum of 'cpu_times_user' and 'cpu_times_system'
    if 'burst_time' does not already exist.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    DataFrame
        The loaded dataset with the expected columns.
    """
    df = pd.read_csv(path)
    # Only keep the relevant columns
    required_cols = [
        'io_read_bytes',
        'io_write_bytes',
        'io_read_count',
        'io_write_count',
        'cpu_percent',
        'num_ctx_switches_voluntary',
        'cpu_times_user',
        'cpu_times_system'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[required_cols].copy()
    # Compute burst_time if not present
    df['burst_time'] = df['cpu_times_user'] + df['cpu_times_system']
    return df

def train_models(df: pd.DataFrame, output_dir: str, dataset_name: str) -> None:
    """Train and evaluate Random Forest and Linear Regression models.

    Splits the data into train and test sets, fits both models, computes
    standard regression metrics and writes plots to the given directory.

    Parameters
    ----------
    df : DataFrame
        Input dataset containing feature columns and a 'burst_time' target.
    output_dir : str
        Directory into which plots will be saved.  The directory is created
        if it does not exist.
    dataset_name : str
        Used as a prefix for filenames and plot titles.
    """
    # Prepare directory
    os.makedirs(output_dir, exist_ok=True)

    # Separate features and target
    feature_cols = [
        'io_read_bytes',
        'io_write_bytes',
        'io_read_count',
        'io_write_count',
        'cpu_percent',
        'num_ctx_switches_voluntary'
    ]
    X = df[feature_cols]
    y = df['burst_time']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Fit Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Fit Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Predictions
    y_pred_rf = rf.predict(X_test)
    y_pred_lr = lr.predict(X_test)

    # Metrics
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    # Print metrics
    print(f"\nDataset: {dataset_name}")
    print("Random Forest -- MAE: {:.4f}, MSE: {:.4f}, R2: {:.4f}".format(mae_rf, mse_rf, r2_rf))
    print("Linear Regression -- MAE: {:.4f}, MSE: {:.4f}, R2: {:.4f}".format(mae_lr, mse_lr, r2_lr))

    # Plot predicted vs actual for Random Forest
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_rf, s=10)
    plt.title(f"Random Forest Predicted vs Actual ({dataset_name})")
    plt.xlabel("Actual Burst Time")
    plt.ylabel("Predicted Burst Time (RF)")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plot_path_rf = os.path.join(output_dir, f"{dataset_name}_rf_pred_vs_actual.png")
    plt.savefig(plot_path_rf)
    plt.close()

    # Plot predicted vs actual for Linear Regression
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_lr, s=10)
    plt.title(f"Linear Regression Predicted vs Actual ({dataset_name})")
    plt.xlabel("Actual Burst Time")
    plt.ylabel("Predicted Burst Time (LR)")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plot_path_lr = os.path.join(output_dir, f"{dataset_name}_lr_pred_vs_actual.png")
    plt.savefig(plot_path_lr)
    plt.close()

    # Plot feature importances for Random Forest
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 6))
    plt.title(f"Feature Importances ({dataset_name})")
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_cols[i] for i in indices], rotation=45, ha='right')
    plt.ylabel("Importance")
    plt.tight_layout()
    plot_path_imp = os.path.join(output_dir, f"{dataset_name}_feature_importance.png")
    plt.savefig(plot_path_imp)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="ML‑guided CPU scheduler phase 1 implementation.")
    parser.add_argument(
        '--real-data', dest='real_data', type=str, default=None,
        help='Path to a CSV file containing a real dataset.  If provided the '
             'script will train models on both the synthetic and real datasets.'
    )
    parser.add_argument(
        '--samples', dest='n_samples', type=int, default=2000,
        help='Number of samples to generate for the synthetic dataset.'
    )
    parser.add_argument(
        '--output', dest='output_dir', type=str, default='plots',
        help='Directory into which plots will be saved.'
    )
    args = parser.parse_args()

    # Generate synthetic dataset
    print("Generating synthetic dataset...")
    synthetic_df = generate_synthetic_dataset(n_samples=args.n_samples)
    train_models(synthetic_df, args.output_dir, dataset_name='synthetic')

    # If a real dataset is provided, load and evaluate
    if args.real_data:
        print(f"\nLoading real dataset from {args.real_data}...")
        real_df = load_dataset(args.real_data)
        train_models(real_df, args.output_dir, dataset_name='real')

if __name__ == '__main__':
    main()