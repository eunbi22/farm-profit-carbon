"""
학습 loss 곡선 시각화.
  - LSTM: epoch별 train loss
  - XGBoost / SARIMAX / Prophet: Optuna trial별 CV RMSE

저장 위치: train/result/
실행: python plot/plot_loss.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from config import RESULT_DIR

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})
COLORS = {"lstm": "#4C72B0", "xgboost": "#DD8452",
          "sarimax": "#55A868", "prophet": "#C44E52"}


def _load(fname):
    path = os.path.join(RESULT_DIR, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def plot_lstm_loss():
    data = _load("loss_history_lstm.json")
    if data is None:
        print("LSTM loss history 없음, 스킵")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(data["train_loss"], color=COLORS["lstm"], linewidth=1.8)
    ax.set_title("LSTM – 최종 학습 Loss (train)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (scaled)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    trials = [v for v in data["optuna_trials"] if v < float("inf")]
    ax.plot(trials, color=COLORS["lstm"], linewidth=1.5, alpha=0.7)
    ax.axhline(min(trials), linestyle="--", color="gray", linewidth=1)
    ax.set_title("LSTM – Optuna CV Loss per Trial")
    ax.set_xlabel("Trial")
    ax.set_ylabel("CV MSE (scaled)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULT_DIR, "loss_lstm.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"저장: {out}")


def plot_optuna_loss(model_name):
    data = _load(f"loss_history_{model_name}.json")
    if data is None:
        print(f"{model_name} loss history 없음, 스킵")
        return
    trials = [v for v in data["trial_values"] if v < float("inf")]
    if not trials:
        return

    running_best = np.minimum.accumulate(trials)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(range(len(trials)), trials, s=15,
               color=COLORS.get(model_name, "steelblue"), alpha=0.5, label="각 Trial")
    ax.plot(running_best, color="black", linewidth=1.5, label="누적 최솟값")
    ax.set_title(f"{model_name.upper()} – Optuna CV RMSE per Trial")
    ax.set_xlabel("Trial")
    ax.set_ylabel("CV RMSE (kg/10a)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULT_DIR, f"loss_{model_name}.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"저장: {out}")


if __name__ == "__main__":
    plot_lstm_loss()
    for m in ("xgboost", "sarimax", "prophet"):
        plot_optuna_loss(m)
