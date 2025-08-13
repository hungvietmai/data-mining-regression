#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nnr_regression.py — Neural Network Regression (Keras + SciKeras) cho dự báo thu nhập nông nghiệp

Tính năng:
- 2 chế độ:
    * time: Train theo năm (hỗ trợ train nhiều năm: --train_years 2010 2014) → Test theo năm
    * cv  : GroupKFold 10 trên toàn bộ dữ liệu (nhóm theo hộ để tránh leakage giữa các năm)
- Tiền xử lý an toàn (fit trên TRAIN): impute, (tùy chọn) winsorization, one-hot categorical, scale numeric (mặc định bật)
- Mục tiêu: AgrInc hoặc CropInc; hỗ trợ --log_target log1p và hiệu chỉnh Duan smearing khi đảo log
- Mạng: MLP nhiều tầng ẩn, ReLU (mặc định), BatchNorm (tùy chọn), Dropout, L2
- Huấn luyện: Adam, EarlyStopping, validation_split nội bộ cho TRAIN
- Xuất: metrics (log & raw), group MAE theo Kinh/tinh

Cách chạy ví dụ:
# Time-split: train 2010+2014 -> test 2016 (log1p, không smearing, one-hot huyện)
python nnr_regression.py --data_dir ./data --run time --target AgrInc \
  --train_years 2010 2014 --test_year 2016 --log_target log1p \
  --exclude_related_income --winsor_lower 0.02 --winsor_upper 0.98 \
  --onehot_geo district --drop_first --epochs 300 --batch_size 128 --patience 20

# Cross-validation 10-fold (GroupKFold), log1p
python nnr_regression.py --data_dir ./data --run cv --cv 10 --target AgrInc \
  --log_target log1p --exclude_related_income --onehot_geo province --drop_first \
  --epochs 200 --batch_size 128 --patience 15
"""

import os
import argparse
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn import __version__ as sklearn_version
from packaging import version

from tensorflow import keras
import tensorflow as tf
from scikeras.wrappers import KerasRegressor

# -----------------
# Utility
# -----------------
def safe_expm1(x, cap=20.0):
    """Stable expm1: clip logits to [-cap, cap] before expm1 to avoid inf/overflow."""
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -cap, cap)
    return np.expm1(x)

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def onehot_compat(drop='first'):
    # Dense output cho NN; tương thích các bản sklearn
    if version.parse(sklearn_version) >= version.parse("1.2"):
        return OneHotEncoder(handle_unknown='ignore', drop=drop, sparse_output=False)
    return OneHotEncoder(handle_unknown='ignore', drop=drop, sparse=False)

class QuantileClipper:
    """Winsorize numeric theo phân vị (dùng trong Pipeline)."""
    def __init__(self, lower=0.0, upper=1.0):
        self.lower = lower
        self.upper = upper
        self.lo_ = None
        self.hi_ = None
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: X = X.reshape(-1,1)
        if self.lower <= 0.0 and self.upper >= 1.0:
            self.lo_ = None; self.hi_ = None
            return self
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        self.hi_ = np.maximum(self.hi_, self.lo_ + 1e-12)
        return self
    def transform(self, X):
        if self.lo_ is None: return X
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: X = X.reshape(-1,1)
        return np.clip(X, self.lo_, self.hi_)

# -----------------
# Data loading
# -----------------
def load_year(data_dir: str, year: int) -> pd.DataFrame:
    """Ưu tiên CSV sạch nếu có, rồi CSV thường, sau cùng .dta"""
    candidates = [f"{year}_clean.csv", f"{year}.csv", f"{year}.dta"]
    for name in candidates:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            print(f"[INFO] Loading {p} ...")
            if name.endswith(".csv"):
                return pd.read_csv(p)
            else:
                return pd.read_stata(p)
    raise FileNotFoundError(f"Không thấy file cho năm {year} trong {data_dir} (tìm: {candidates})")

# -----------------
# Preprocessing
# -----------------
def detect_cols(df: pd.DataFrame, geo_level: str, year_as_category: bool) -> Tuple[List[str], List[str]]:
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    # Geo
    add = []
    if geo_level == 'province': add = ['tinh']
    elif geo_level == 'district': add = ['tinh','quan']
    elif geo_level in ('commune','all'): add = ['tinh','quan','xa']
    for c in add:
        if c in df.columns and c not in cat_cols:
            cat_cols.append(c)
    # Year
    if year_as_category and 'Year' in df.columns and 'Year' not in cat_cols:
        cat_cols.append('Year')
    # Numeric
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in cat_cols]
    return num_cols, cat_cols

def build_preprocessor(df: pd.DataFrame, exclude_cols: List[str], geo_level: str,
                       year_as_category: bool, winsor_lower: float, winsor_upper: float,
                       scale_numeric: bool, drop_first: bool) -> Tuple[ColumnTransformer, List[str], List[str]]:
    cols = [c for c in df.columns if c not in exclude_cols]
    tmp = df[cols].copy()
    num_cols, cat_cols = detect_cols(tmp, geo_level, year_as_category)

    num_steps = [('imp', SimpleImputer(strategy='median'))]
    if winsor_lower is not None and winsor_upper is not None and (winsor_lower>0.0 or winsor_upper<1.0):
        num_steps.append(('winsor', QuantileClipper(winsor_lower, winsor_upper)))
    if scale_numeric:
        num_steps.append(('sc', StandardScaler()))  # NN rất cần scale

    cat_steps = [('imp', SimpleImputer(strategy='most_frequent')),
                 ('ohe', onehot_compat(drop='first' if drop_first else None))]

    preproc = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=num_steps), num_cols),
            ('cat', Pipeline(steps=cat_steps), cat_cols)
        ]
    )
    return preproc, num_cols, cat_cols

def get_feature_names(preproc: ColumnTransformer) -> List[str]:
    names = []
    for name, trans, cols in preproc.transformers_:
        if name == 'num':
            names.extend(cols)
        elif name == 'cat':
            ohe = trans.named_steps['ohe']
            names.extend(list(ohe.get_feature_names_out(cols)))
    return names

# -----------------
# Targets & Metrics
# -----------------
def make_y(series: pd.Series, mode: str) -> Tuple[np.ndarray, Dict]:
    """mode: 'raw' | 'log1p' (clip âm về 0 để log an toàn)"""
    s = pd.to_numeric(series, errors='coerce').values.astype(float)
    meta = {'mode': mode}
    if mode == 'raw':
        y = s
        meta['invert'] = lambda arr: arr
    else:
        s = np.where(np.isfinite(s), s, np.nan)
        s = np.nan_to_num(s, nan=0.0)
        s = np.maximum(s, 0.0)
        y = np.log1p(s).astype(float)
        meta['invert'] = lambda arr: safe_expm1(arr)
    return y, meta

def duan_smearing(y_true_log: np.ndarray, y_pred_log: np.ndarray, clip_range=(0.5, 1.5)) -> float:
    resid = y_true_log - y_pred_log
    s = float(np.mean(np.exp(resid)))
    if clip_range:
        s = float(np.clip(s, clip_range[0], clip_range[1]))
    return s

def evaluate(y_true, y_pred) -> Dict[str, float]:
    return {'MAE': float(mean_absolute_error(y_true, y_pred)),
            'RMSE': rmse(y_true, y_pred),
            'R2': float(r2_score(y_true, y_pred))}

# -----------------
# Build Keras model
# -----------------
def build_mlp(meta,
              hidden_layers=(256,128,64),
              activation='relu',
              dropout=0.1,
              l2=1e-4,
              batchnorm=True,
              lr=1e-3,
              loss='mse',
              random_state=None,
              **kwargs):
    # Seed
    if random_state is not None:
        tf.keras.utils.set_random_seed(random_state)

    n_features_in_ = int(meta.get('n_features_in_', meta['X_shape_'][1]))
    inputs = keras.Input(shape=(n_features_in_,), name='inputs')
    x = inputs
    for i, units in enumerate(hidden_layers):
        x = keras.layers.Dense(units, kernel_regularizer=keras.regularizers.l2(l2))(x)
        if batchnorm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation)(x)
        if dropout and dropout > 0:
            x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(1, activation='linear', name='y')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss, metrics=[keras.metrics.MeanAbsoluteError(name='mae')])
    return model

def make_nnr(args):
    # SciKeras KerasRegressor, truyền hyperparams qua model__*
    est = KerasRegressor(
        model=build_mlp,
        # loss/optimizer & fit params
        loss=args.loss,
        optimizer='adam',
        optimizer__learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
        random_state=args.random_state,
        # model-specific params
        model__hidden_layers=tuple(int(x) for x in args.hidden.split(',')),
        model__activation=args.activation,
        model__dropout=args.dropout,
        model__l2=args.l2,
        model__batchnorm=(not args.no_batchnorm),
        model__lr=args.lr,
    )
    # EarlyStopping callback (monitor val_loss nếu có validation_split>0)
    cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)
    est.set_params(fit__validation_split=args.val_split, fit__callbacks=[cb], fit__shuffle=True)
    return est

# -----------------
# Runs
# -----------------
def run_time_split(train_dfs: List[pd.DataFrame], df_test: pd.DataFrame, target: str, args) -> pd.DataFrame:
    # Exclusions để tránh leak
    exclude_cols = ['ma_ho']
    if args.exclude_related_income:
        if target == 'AgrInc':
            exclude_cols += ['TotalInc','Wage','CropInc','CropValue']
        elif target == 'CropInc':
            exclude_cols += ['TotalInc','Wage','AgrInc']

    df_train = pd.concat(train_dfs, ignore_index=True, sort=False)

    y_tr, ymeta = make_y(df_train[target], args.log_target)
    y_te, _ = make_y(df_test[target], args.log_target)
    mask_tr = ~np.isnan(y_tr); mask_te = ~np.isnan(y_te)
    X_tr = df_train.loc[mask_tr, :].drop(columns=[target]); y_tr = y_tr[mask_tr]
    X_te = df_test.loc[mask_te, :].drop(columns=[target]); y_te = y_te[mask_te]

    preproc, _, _ = build_preprocessor(
        X_tr, exclude_cols=exclude_cols, geo_level=args.onehot_geo, year_as_category=args.year_as_category,
        winsor_lower=args.winsor_lower, winsor_upper=args.winsor_upper,
        scale_numeric=True, drop_first=args.drop_first   # NN luôn scale numeric
    )

    nnr = make_nnr(args)
    pipe = Pipeline([('prep', preproc), ('est', nnr)])
    pipe.fit(X_tr, y_tr)

    yhat_te = pipe.predict(X_te)
    rep = evaluate(y_te, yhat_te)

    # Back-transform
    if args.log_target == 'log1p':
        if args.smearing:
            yhat_tr = pipe.predict(X_tr)
            s = duan_smearing(y_tr, yhat_tr, clip_range=(0.5, 1.5))
        else:
            s = 1.0
        inv = ymeta['invert']
        y_true_raw = inv(y_te)
        y_pred_raw = s * inv(yhat_te)
        y_pred_raw = np.maximum(y_pred_raw, 0.0)
        rep.update({
            'MAE_raw': float(mean_absolute_error(y_true_raw, y_pred_raw)),
            'RMSE_raw': rmse(y_true_raw, y_pred_raw),
            'R2_raw': float(r2_score(y_true_raw, y_pred_raw)),
            'smearing': s
        })

    rep.update({'model': 'NNR', 'run': 'time',
                'train_years': ','.join(str(y) for y in args.train_years) if args.train_years else str(args.train_year),
                'test_year': args.test_year})

    # Group MAE (raw nếu có)
    try:
        df_eval = X_te.copy()
        if args.log_target == 'log1p':
            inv = ymeta['invert']; s = rep.get('smearing', 1.0)
            df_eval['y_true'] = inv(y_te)
            df_eval['y_pred'] = s * inv(yhat_te)
        else:
            df_eval['y_true'] = y_te; df_eval['y_pred'] = yhat_te
        df_eval['abs_err'] = np.abs(df_eval['y_true'] - df_eval['y_pred'])
        for gcol in ['Kinh','tinh']:
            if gcol in df_eval.columns:
                grp = df_eval.groupby(gcol, observed=False)['abs_err'].mean().reset_index().rename(columns={'abs_err':'MAE_group'})
                grp.to_csv(os.path.join(args.results_dir, f"NNR_by_{gcol}_time_{rep['train_years']}_{args.test_year}.csv"), index=False)
    except Exception as e:
        print(f"[WARN] Không xuất được group error: {e}")

    return pd.DataFrame([rep])

def run_cv(df_all: pd.DataFrame, target: str, args) -> pd.DataFrame:
    exclude_cols = ['ma_ho']
    if args.exclude_related_income:
        if target == 'AgrInc':
            exclude_cols += ['TotalInc','Wage','CropInc','CropValue']
        elif target == 'CropInc':
            exclude_cols += ['TotalInc','Wage','AgrInc']

    y_all, ymeta = make_y(df_all[target], args.log_target)
    mask = ~np.isnan(y_all)
    X_all = df_all.loc[mask, :].drop(columns=[target])
    y_all = y_all[mask]

    # Splitter
    if set(['tinh','quan','xa','ma_ho']).issubset(df_all.columns):
        groups = df_all.loc[mask, ['tinh','quan','xa','ma_ho']].astype(str).agg('-'.join, axis=1).values
        splitter = GroupKFold(n_splits=args.cv).split(X_all, y_all, groups=groups)
    else:
        splitter = KFold(n_splits=args.cv, shuffle=True, random_state=42).split(X_all, y_all)

    mae_list, rmse_list, r2_list = [], [], []
    mae_raw_list, rmse_raw_list, r2_raw_list = [], [], []

    for tr_idx, te_idx in splitter:
        X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

        preproc, _, _ = build_preprocessor(
            X_tr, exclude_cols=exclude_cols, geo_level=args.onehot_geo, year_as_category=args.year_as_category,
            winsor_lower=args.winsor_lower, winsor_upper=args.winsor_upper,
            scale_numeric=True, drop_first=args.drop_first
        )

        nnr = make_nnr(args)
        pipe = Pipeline([('prep', preproc), ('est', nnr)])
        # Trong CV, vẫn dùng validation_split nội bộ từ args.val_split
        pipe.fit(X_tr, y_tr)
        yhat = pipe.predict(X_te)
        rep = evaluate(y_te, yhat)
        mae_list.append(rep['MAE']); rmse_list.append(rep['RMSE']); r2_list.append(rep['R2'])

        if args.log_target == 'log1p':
            s = duan_smearing(y_tr, pipe.predict(X_tr), clip_range=(0.5, 1.5)) if args.smearing else 1.0
            inv = ymeta['invert']
            y_true_raw = inv(y_te); y_pred_raw = s * inv(yhat)
            y_pred_raw = np.maximum(y_pred_raw, 0.0)
            mae_raw_list.append(float(mean_absolute_error(y_true_raw, y_pred_raw)))
            rmse_raw_list.append(rmse(y_true_raw, y_pred_raw))
            r2_raw_list.append(float(r2_score(y_true_raw, y_pred_raw)))

    res = {
        'model': 'NNR', 'run': 'cv', 'cv': args.cv,
        'MAE_mean': float(np.mean(mae_list)), 'RMSE_mean': float(np.mean(rmse_list)), 'R2_mean': float(np.mean(r2_list)),
        'MAE_std': float(np.std(mae_list)), 'RMSE_std': float(np.std(rmse_list)), 'R2_std': float(np.std(r2_list))
    }
    if args.log_target == 'log1p':
        res.update({
            'MAE_raw_mean': float(np.mean(mae_raw_list)), 'RMSE_raw_mean': float(np.mean(rmse_raw_list)), 'R2_raw_mean': float(np.mean(r2_raw_list)),
            'MAE_raw_std': float(np.std(mae_raw_list)), 'RMSE_raw_std': float(np.std(rmse_raw_list)), 'R2_raw_std': float(np.std(r2_raw_list)),
        })
    return pd.DataFrame([res]).sort_values('RMSE_mean')

# -----------------
# Main
# -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--run', type=str, choices=['time','cv'], default='time')
    parser.add_argument('--target', type=str, choices=['AgrInc','CropInc'], default='AgrInc')
    parser.add_argument('--cv', type=int, default=10)
    parser.add_argument('--train_year', type=int, default=2014, help='Dùng nếu không cung cấp --train_years')
    parser.add_argument('--train_years', type=int, nargs='*', help='Ví dụ: --train_years 2010 2014')
    parser.add_argument('--test_year', type=int, default=2016)
    parser.add_argument('--log_target', type=str, choices=['raw','log1p'], default='log1p')
    parser.add_argument('--smearing', action='store_true', help='Hiệu chỉnh Duan khi đảo log về thang gốc (kẹp 0.5–1.5)')
    parser.add_argument('--exclude_related_income', action='store_true')
    parser.add_argument('--winsor_lower', type=float, default=None)
    parser.add_argument('--winsor_upper', type=float, default=None)
    parser.add_argument('--onehot_geo', type=str, default='province', choices=['none','province','district','commune','all'])
    parser.add_argument('--year_as_category', action='store_true')
    parser.add_argument('--drop_first', action='store_true')
    parser.add_argument('--results_dir', type=str, default='./results_nnr')
    # NN hyperparams
    parser.add_argument('--hidden', type=str, default='256,128,64', help='Ví dụ: "256,128,64"')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--loss', type=str, default='mse', choices=['mse','mae'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--val_split', type=float, default=0.1, help='Tách validation nội bộ từ TRAIN')
    parser.add_argument('--no_batchnorm', action='store_true', help='Tắt BatchNorm trong các hidden layer')
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Load data
    df2010 = load_year(args.data_dir, 2010)
    df2014 = load_year(args.data_dir, 2014)
    df2016 = load_year(args.data_dir, 2016)

    if args.run == 'time':
        if args.train_years:
            train_map = {2010: df2010, 2014: df2014, 2016: df2016}
            train_dfs = [train_map[y] for y in args.train_years]
            train_label = ','.join(str(y) for y in args.train_years)
        else:
            train_dfs = [df2014 if args.train_year==2014 else (df2010 if args.train_year==2010 else df2016)]
            train_label = str(args.train_year)
        test_df = df2016 if args.test_year==2016 else (df2014 if args.test_year==2014 else df2010)

        res = run_time_split(train_dfs, test_df, args.target, args)
        out_csv = os.path.join(args.results_dir, f"NNR_{args.target}_time_{train_label}_{args.test_year}.csv")
        res.to_csv(out_csv, index=False)
        print("=== Time-split NNR Results ===")
        print(res.to_string(index=False))
        print(f"[DONE] Saved: {out_csv}")
    else:
        df_all = pd.concat([df2010, df2014, df2016], ignore_index=True, sort=False)
        res = run_cv(df_all, args.target, args)
        out_csv = os.path.join(args.results_dir, f"NNR_{args.target}_cv_{args.cv}fold.csv")
        res.to_csv(out_csv, index=False)
        print("=== CV NNR Results ===")
        print(res.to_string(index=False))
        print(f"[DONE] Saved: {out_csv}")

if __name__ == '__main__':
    main()
