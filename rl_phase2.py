"""
rl_phase2.py
-------------

This script implements a proof‑of‑concept reinforcement learning (RL)
agent for CPU scheduling decisions.  It builds on the first phase of
the ML‑guided CPU scheduler project by reusing the trained Random
Forest model to generate predicted burst times for each process in a
real dataset.  The RL environment presents the agent with pairs of
processes characterised by their predicted and actual burst times.  The
agent must decide which process to schedule next.  A simple Q‑learning
algorithm is used to learn a policy that picks the process with the
shorter actual burst time based solely on the predicted values.

**Important:** This POC is intentionally simple.  The environment is
stateless and each decision is independent of previous decisions.  The
reward is +1 if the agent chooses the process with the smaller actual
burst time, ‑1 otherwise.  There is no discount factor or terminal
state; the objective is merely to maximise the number of correct
choices.  More sophisticated environments could incorporate queue
dynamics, waiting times and context switch overheads, but these are
outside the scope of this demonstration.

Usage
-----

To train the RL agent on the real dataset, first train a Random
Forest model using `main.py` and save the predicted burst times.  Then
run this script.  For example:

```sh
python main.py --real-data process_data.csv --output plots
python rl_phase2.py --real-data process_data.csv --model-burst ml_cpu_scheduler_project/plots/real_rf_pred_vs_actual.png
```

However, the script can also generate its own predicted values by
training a Random Forest on the fly.  See the arguments below.

Arguments
~~~~~~~~~

--real-data PATH     Path to a CSV file containing a real process dataset.
--episodes N         Number of episodes to train the RL agent (default
                     5000).  More episodes yield better policies but
                     increase runtime.
--epsilon EPS        Initial exploration probability for epsilon‑greedy
                     action selection (default 0.3).  Epsilon decays
                     linearly to zero over all episodes.
--alpha ALPHA        Learning rate for Q‑learning updates (default
                     0.5).
--seed SEED          Random seed for reproducibility (default 42).

Output
~~~~~~

After training, the script prints the final Q‑table and the overall
accuracy (fraction of correct choices) of the learned policy on a
held‑out test set of pairs.  A simple report is written to
``rl_report.txt`` in the current directory summarising the training
configuration and results.
"""

import argparse
import os
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define states for discretisation
def discretise_state(pred1: float, pred2: float) -> int:
    """Discretise a pair of predicted burst times into a small state space.

    The state is defined by the relative ordering and magnitude of the
    predictions.  Three discrete states are used:

    0: pred1 < pred2 by more than 10% of their average
    1: |pred1 - pred2| <= 10% of their average (nearly equal)
    2: pred1 > pred2 by more than 10% of their average

    This coarse discretisation allows Q‑learning to operate on a
    manageable number of states while still capturing whether one
    predicted burst is significantly smaller or larger than the other.
    """
    avg = (pred1 + pred2) / 2.0
    if avg == 0:
        return 1
    diff = pred1 - pred2
    if abs(diff) <= 0.1 * avg:
        return 1
    return 0 if diff < 0 else 2

def load_data_and_predict(data_path: str, seed: int = 42) -> tuple:
    """Load a process dataset, train a Random Forest and return predictions.

    Returns
    -------
    tuple
        A tuple (predicted, actual) where both elements are 1D numpy arrays
        of equal length containing predicted and true burst times.
    """
    df = pd.read_csv(data_path)
    feature_cols = [
        'io_read_bytes',
        'io_write_bytes',
        'io_read_count',
        'io_write_count',
        'cpu_percent',
        'num_ctx_switches_voluntary'
    ]
    X = df[feature_cols]
    y = df['cpu_times_user'] + df['cpu_times_system']
    # Train/test split to avoid overfitting the RF on all data; the RL
    # environment will sample from the test portion.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    rf = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    return preds, y_test.values

def generate_pairs(predicted: np.ndarray, actual: np.ndarray) -> list:
    """Generate all unordered pairs of predicted/actual burst times.

    Each pair is represented as a tuple:

        ((pred1, pred2), (true1, true2))

    Pairs are generated without replacement: (i,j) and (j,i) are
    considered distinct because the action space is ordered (action 0
    selects process 1, action 1 selects process 2).
    """
    pairs = []
    n = len(predicted)
    # Use a fixed subset to limit the number of pairs; sampling all
    # combinations would be quadratic in n and unnecessary for the POC.
    # We'll randomly sample 5000 pairs or all possible if fewer.
    rng = np.random.default_rng(42)
    max_pairs = min(5000, n * (n - 1))
    indices = rng.choice(n * (n - 1), size=max_pairs, replace=False)
    # Map flattened index back to (i,j)
    for idx in indices:
        i = idx // (n - 1)
        j = idx % (n - 1)
        if j >= i:
            j += 1
        pairs.append(((predicted[i], predicted[j]), (actual[i], actual[j])))
    return pairs

def q_learning_train(pairs: list, episodes: int, epsilon: float, alpha: float, seed: int = 42) -> tuple:
    """Train a Q‑learning agent on the provided pairs.

    Returns the learned Q‑table and list of episode accuracies.
    """
    rng = np.random.default_rng(seed)
    # Initialise Q‑table: mapping from state index to Q‑values for each action
    # There are 3 states (see discretise_state) and 2 actions (choose process 0 or 1).
    q_table = np.zeros((3, 2), dtype=float)
    accuracies = []
    # Precompute true labels for all pairs: label = 0 if true1 <= true2 else 1
    true_labels = [0 if p[1][0] <= p[1][1] else 1 for p in pairs]

    for episode in range(episodes):
        # Epsilon decays linearly to zero over episodes
        eps = epsilon * (1 - episode / episodes)
        # Sample a random pair
        idx = rng.integers(0, len(pairs))
        (preds, truths) = pairs[idx]
        state = discretise_state(preds[0], preds[1])
        # Epsilon‑greedy action
        if rng.random() < eps:
            action = rng.integers(0, 2)
        else:
            action = int(np.argmax(q_table[state]))
        # Reward is +1 if chosen action matches true label, else ‑1
        reward = 1.0 if action == (0 if truths[0] <= truths[1] else 1) else -1.0
        # For this stateless problem, next state is irrelevant; no discount factor
        q_table[state, action] = q_table[state, action] + alpha * (reward - q_table[state, action])
        # Track training accuracy
        accuracies.append(1 if reward > 0 else 0)
    return q_table, accuracies

def evaluate_policy(q_table: np.ndarray, pairs: list) -> float:
    """Evaluate the learned policy on a set of pairs.  Returns accuracy."""
    correct = 0
    for (preds, truths) in pairs:
        state = discretise_state(preds[0], preds[1])
        action = int(np.argmax(q_table[state]))
        correct += 1 if action == (0 if truths[0] <= truths[1] else 1) else 0
    return correct / len(pairs)

def main():
    parser = argparse.ArgumentParser(description="POC reinforcement learning scheduler")
    parser.add_argument('--real-data', dest='real_data', required=True, help='CSV file with process data')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of training episodes')
    parser.add_argument('--epsilon', type=float, default=0.3, help='Initial exploration rate')
    parser.add_argument('--alpha', type=float, default=0.5, help='Learning rate for Q‑learning')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Load data and obtain predicted and actual bursts
    preds, actual = load_data_and_predict(args.real_data, seed=args.seed)
    # Generate pairs for training and testing
    pairs = generate_pairs(preds, actual)
    # Split into train/test pairs (80/20 split)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(pairs)
    split_idx = int(0.8 * len(pairs))
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]

    # Train the Q‑learning agent
    q_table, accuracies = q_learning_train(
        train_pairs,
        episodes=args.episodes,
        epsilon=args.epsilon,
        alpha=args.alpha,
        seed=args.seed
    )
    # Evaluate on test pairs
    test_accuracy = evaluate_policy(q_table, test_pairs)

    # Prepare report
    report_lines = []
    report_lines.append("Reinforcement Learning Scheduler POC")
    report_lines.append("====================================\n")
    report_lines.append(f"Real dataset: {args.real_data}")
    report_lines.append(f"Number of training pairs: {len(train_pairs)}")
    report_lines.append(f"Number of test pairs: {len(test_pairs)}")
    report_lines.append(f"Episodes: {args.episodes}")
    report_lines.append(f"Initial epsilon: {args.epsilon}")
    report_lines.append(f"Learning rate (alpha): {args.alpha}\n")
    report_lines.append("Final Q‑table (state x action):")
    report_lines.append(str(q_table))
    report_lines.append("\nTraining accuracy (last 100 episodes): {:.3f}".format(sum(accuracies[-100:]) / 100.0))
    report_lines.append("Overall test accuracy: {:.3f}".format(test_accuracy))
    report_content = "\n".join(report_lines)
    report_path = 'rl_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_content)
    print(report_content)
    print(f"\nReport written to {report_path}")

if __name__ == '__main__':
    main()