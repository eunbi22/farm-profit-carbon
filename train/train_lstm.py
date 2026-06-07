"""
TensorFlow/Keras LSTM 필지별 생산량 예측 (Optuna + Adam).

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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import optuna

from config import RESULT_DIR, OPTUNA_TRIALS, RANDOM_SEED, CV_FOLDS
from dataset import (
    load_dataset, split_train_test, get_cv_splits,
    fit_scalers, prepare_lstm_sequences, TARGET,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
SAVE_DIR = os.path.join(RESULT_DIR, "models")


def build_model(hidden_size: int, num_layers: int, dropout: float) -> keras.Model:
    seq_input    = keras.Input(shape=(5, 2),  name="seq_input")
    static_input = keras.Input(shape=(4,),    name="static_input")

    x = seq_input
    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)
        x = layers.LSTM(hidden_size, return_sequences=return_sequences,
                        dropout=dropout)(x)

    x = layers.Concatenate()([x, static_input])
    x = layers.Dense(hidden_size, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    output = layers.Dense(1)(x)

    return keras.Model(inputs=[seq_input, static_input], outputs=output)


def _train_one_fold(params: dict, seq_tr, stat_tr, y_tr,
                    seq_val, stat_val, y_val, n_epochs=50):
    tf.random.set_seed(RANDOM_SEED)
    model = build_model(params["hidden_size"], params["num_layers"], params["dropout"])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params["lr"]),
                  loss="mse")
    model.fit(
        [seq_tr, stat_tr], y_tr,
        validation_data=([seq_val, stat_val], y_val),
        epochs=n_epochs,
        batch_size=params["batch_size"],
        verbose=0,
    )
    val_loss = model.evaluate([seq_val, stat_val], y_val, verbose=0)
    return val_loss


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
        seq_tr, stat_tr, y_tr   = prepare_lstm_sequences(train_df[tr_mask],  scaler_X, scaler_y)
        seq_val, stat_val, y_val = prepare_lstm_sequences(train_df[val_mask], scaler_X, scaler_y)
        fold_losses.append(
            _train_one_fold(params, seq_tr, stat_tr, y_tr, seq_val, stat_val, y_val)
        )
    return float(np.mean(fold_losses))


def train_lstm(parcel_df):
    tf.random.set_seed(RANDOM_SEED)
    df = load_dataset(parcel_df)
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

    seq_tr, stat_tr, y_tr = prepare_lstm_sequences(train_df, scaler_X, scaler_y)
    final = build_model(best_p["hidden_size"], best_p["num_layers"], best_p["dropout"])
    final.compile(optimizer=keras.optimizers.Adam(learning_rate=best_p["lr"]), loss="mse")

    history = final.fit(
        [seq_tr, stat_tr], y_tr,
        epochs=100,
        batch_size=best_p["batch_size"],
        verbose=0,
        callbacks=[keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs:
                print(f"  Epoch {epoch+1:3d}/100  loss={logs['loss']:.4f}")
                if (epoch + 1) % 10 == 0 else None
        )],
    )

    final.save(os.path.join(SAVE_DIR, "lstm.keras"))
    with open(os.path.join(RESULT_DIR, "metrics", "lstm_best_params.json"), "w") as f:
        json.dump(best_p, f, indent=2)
    with open(os.path.join(RESULT_DIR, "loss_history_lstm.json"), "w") as f:
        json.dump({
            "train_loss":   history.history["loss"],
            "optuna_trials": [t.value for t in study.trials],
        }, f)

    print("LSTM 모델 저장 완료.")
    return final, scaler_X, scaler_y, test_df


if __name__ == "__main__":
    import numpy as np
    from features import build_parcel_features

    pf = build_parcel_features()
    # 임시 yield: 나중에 실제 생산량 데이터로 교체
    pf["yield_per_10a"] = (
        pf["ta_season_mean"] * 5.0 + pf["rn_season_sum"] * 0.05 + 400
    ).clip(lower=100)
    train_lstm(pf)
