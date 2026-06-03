"""
PyTorch LSTM 필지별 생산량 예측 (Optuna + Adam).

입력:
  seq_input    (N, 5, 2)  – 영농기 월별 [avg_ta, sum_rn]
  static_input (N, 4)     – [area, cad_con_ra, lat, lon]
출력: yield_per_10a (정규화 값)

실행: python train_lstm.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna

from config import RESULT_DIR, OPTUNA_TRIALS, RANDOM_SEED, CV_FOLDS
from dataset import (
    load_dataset, split_train_test, get_cv_splits,
    fit_scalers, prepare_lstm_sequences, TARGET,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
SAVE_DIR  = os.path.join(RESULT_DIR, "models")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class YieldLSTM(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float,
                 static_dim: int = 4, seq_features: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + static_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, seq, static):
        _, (h_n, _) = self.lstm(seq)
        h_last = h_n[-1]                        # (batch, hidden_size)
        x = torch.cat([h_last, static], dim=1)  # (batch, hidden + static_dim)
        return self.head(x).squeeze(1)


def _make_tensors(seq, static, y):
    return (
        torch.tensor(seq,    dtype=torch.float32),
        torch.tensor(static, dtype=torch.float32),
        torch.tensor(y,      dtype=torch.float32),
    )


def _train_one_fold(params: dict, seq_tr, stat_tr, y_tr,
                    seq_val, stat_val, y_val, n_epochs=50):
    model = YieldLSTM(
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.MSELoss()

    ds = TensorDataset(*_make_tensors(seq_tr, stat_tr, y_tr))
    dl = DataLoader(ds, batch_size=params["batch_size"], shuffle=True)

    best_val = float("inf")
    for _ in range(n_epochs):
        model.train()
        for s, st, yb in dl:
            s, st, yb = s.to(DEVICE), st.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(s, st), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            sv  = torch.tensor(seq_val,  dtype=torch.float32).to(DEVICE)
            stv = torch.tensor(stat_val, dtype=torch.float32).to(DEVICE)
            yv  = torch.tensor(y_val,    dtype=torch.float32).to(DEVICE)
            val_loss = loss_fn(model(sv, stv), yv).item()
        if val_loss < best_val:
            best_val = val_loss

    return best_val


def _objective(trial, train_df, scaler_X, scaler_y, cv_splits):
    params = {
        "hidden_size": trial.suggest_int("hidden_size", 32, 256, step=32),
        "num_layers":  trial.suggest_int("num_layers", 1, 3),
        "dropout":     trial.suggest_float("dropout", 0.0, 0.5),
        "lr":          trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size":  trial.suggest_categorical("batch_size", [64, 128, 256]),
    }
    fold_losses = []
    for tr_mask, val_mask in cv_splits:
        seq_tr, stat_tr, y_tr  = prepare_lstm_sequences(train_df[tr_mask],  scaler_X, scaler_y)
        seq_val, stat_val, y_val = prepare_lstm_sequences(train_df[val_mask], scaler_X, scaler_y)
        fold_losses.append(
            _train_one_fold(params, seq_tr, stat_tr, y_tr, seq_val, stat_val, y_val)
        )
    return float(np.mean(fold_losses))


def train_lstm(parcel_df):
    torch.manual_seed(RANDOM_SEED)
    df       = load_dataset(parcel_df)
    train_df, test_df = split_train_test(df)
    scaler_X, scaler_y = fit_scalers(train_df)
    cv_splits = get_cv_splits(train_df)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(
        lambda t: _objective(t, train_df, scaler_X, scaler_y, cv_splits),
        n_trials=OPTUNA_TRIALS,
        show_progress_bar=True,
    )

    best_p = study.best_params
    print(f"\n최적 LSTM 파라미터: {best_p}")
    print(f"최적 CV Loss (scaled MSE): {study.best_value:.4f}")

    # 전체 train으로 최종 학습 (loss 기록 포함)
    seq_tr, stat_tr, y_tr = prepare_lstm_sequences(train_df, scaler_X, scaler_y)
    final = YieldLSTM(
        hidden_size=best_p["hidden_size"],
        num_layers=best_p["num_layers"],
        dropout=best_p["dropout"],
    ).to(DEVICE)
    opt    = torch.optim.Adam(final.parameters(), lr=best_p["lr"])
    loss_fn = nn.MSELoss()
    ds = TensorDataset(*_make_tensors(seq_tr, stat_tr, y_tr))
    dl = DataLoader(ds, batch_size=best_p["batch_size"], shuffle=True)

    train_losses = []
    for epoch in range(100):
        final.train()
        epoch_loss = 0.0
        for s, st, yb in dl:
            s, st, yb = s.to(DEVICE), st.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            l = loss_fn(final(s, st), yb)
            l.backward()
            opt.step()
            epoch_loss += l.item()
        train_losses.append(epoch_loss / len(dl))
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/100  loss={train_losses[-1]:.4f}")

    torch.save(final.state_dict(), os.path.join(SAVE_DIR, "lstm.pt"))
    with open(os.path.join(RESULT_DIR, "metrics", "lstm_best_params.json"), "w") as f:
        json.dump(best_p, f, indent=2)
    with open(os.path.join(RESULT_DIR, "loss_history_lstm.json"), "w") as f:
        json.dump({
            "train_loss": train_losses,
            "optuna_trials": [t.value for t in study.trials],
        }, f)

    print("LSTM 모델 저장 완료.")
    return final, scaler_X, scaler_y, test_df


if __name__ == "__main__":
    from features import build_parcel_features
    from disaggregate import (
        load_county_production, fit_county_model,
        compute_parcel_weights, disaggregate_yield,
    )

    pf        = build_parcel_features()
    county_df = load_county_production()
    cw        = pf.groupby("year")[["ta_season_mean", "rn_season_sum"]].mean().reset_index()
    reg       = fit_county_model(county_df, cw)
    pf_w      = compute_parcel_weights(pf, reg["a"], reg["b"], reg["c"])
    parcel_df = disaggregate_yield(pf_w, county_df)
    train_lstm(parcel_df)
