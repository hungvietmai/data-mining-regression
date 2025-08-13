#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gb_regression.py — Gradient Boosting cho dự báo thu nhập nông nghiệp

Tính năng:
- 2 chế độ:
    * time: Train theo năm (hỗ trợ train nhiều năm: --train_years 2010 2014) → Test theo năm
    * cv  : GroupKFold 10 trên toàn bộ dữ liệu (nhóm theo hộ tránh leakage)
- Engine:
    * --engine xgb  : XGBoost (khuyến nghị nếu đã cài xgboost)
    * --engine skgb : sklearn HistGradientBoostingRegressor (fallback nếu không có xgboost)
- Tiền xử lý an toàn: impute, (tùy chọn) winsorization, one-hot categorical, (tùy chọn) scale numeric
- Target: AgrInc (mặc định) hoặc CropInc; hỗ trợ --log_target log1p và Duan smearing --smearing
- Đánh giá: MAE/RMSE/R² trên thang đang học; nếu log1p → xuất thêm *_raw và hệ số smearing
- Xuất:
    * results_gb/GB_<...>.csv (metrics)
    * GB_feature_importance_*.csv (nếu engine có importance)
    * GB_perm_importance_*.csv (nếu cần dùng permutation importance)
    * GB_by_Kinh_*.csv, GB_by_tinh_*.csv
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
from sklearn.model_selection import GroupKFold, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn import __version__ as sklearn_version
from packaging import version

# Optional XGBoost
_HAS_XGB = True
try:
    from xgboost import XGBRegressor
except Exception:
    _HAS_XGB = False

# -----------------
# Utility
# -----------------
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def onehot_compat(drop='first'):
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
        num_steps.append(('sc', StandardScaler()))

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
        meta['invert'] = lambda arr: np.expm1(arr)
    return y, meta

def duan_smearing(y_true_log: np.ndarray, y_pred_log: np.ndarray, clip_range=(0.5, 1.5)) -> float:
    """Hệ số smearing s = mean(exp(residual)). Có kẹp biên để tránh nổ."""
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
# Model factory
# -----------------
def make_gb(engine: str,
            n_estimators: int,
            learning_rate: float,
            max_depth: int,
            subsample: float,
            colsample_bytree,
            reg_lambda: float,
            reg_alpha: float,
            min_samples_leaf: int,
            random_state: int):
    if engine == 'xgb' and _HAS_XGB:
        return XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            min_child_weight=min_samples_leaf,
            objective='reg:squarederror',
            n_jobs=-1,
            random_state=random_state,
        )
    else:
        # Fallback: HistGradientBoostingRegressor (sklearn)
        # Note: HGBR không có feature_importances_, sẽ dùng permutation importance
        max_depth_hgbr = None if max_depth is None or max_depth <= 0 else max_depth
        return HistGradientBoostingRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth_hgbr,
            max_bins=255,
            l2_regularization=reg_lambda,
            early_stopping=False,
            random_state=random_state
        )

# -----------------
# Runs
# -----------------
def run_time_split(train_dfs: List[pd.DataFrame], df_test: pd.DataFrame, target: str, args) -> pd.DataFrame:
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
        scale_numeric=args.scale_numeric, drop_first=args.drop_first
    )

    gb = make_gb(args.engine, args.n_estimators, args.learning_rate, args.max_depth,
                 args.subsample, args.colsample_bytree, args.reg_lambda, args.reg_alpha,
                 args.min_samples_leaf, args.random_state)

    pipe = Pipeline([('prep', preproc), ('est', gb)])
    if args.tune and args.engine == 'xgb' and _HAS_XGB:
        param_grid = {
            'est__n_estimators': [800, 1500, 2000],
            'est__learning_rate': [0.05, 0.03, 0.01],
            'est__max_depth': [4, 6, 8],
            'est__subsample': [0.8, 0.9, 1.0],
            'est__colsample_bytree': [0.7, 0.9, 1.0],
            'est__reg_lambda': [1.0, 2.0, 5.0],
        }
        gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_root_mean_squared_error', verbose=1)
        gs.fit(X_tr, y_tr)
        pipe = gs.best_estimator_
    else:
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

    rep.update({'model': 'GB('+('xgb' if args.engine=='xgb' and _HAS_XGB else 'skgb')+')',
                'run': 'time',
                'train_years': ','.join(str(y) for y in args.train_years) if args.train_years else str(args.train_year),
                'test_year': args.test_year})

    # Feature importance
    try:
        est = pipe.named_steps['est']
        prep = pipe.named_steps['prep']
        feat_names = get_feature_names(prep)
        if hasattr(est, 'feature_importances_'):
            importances = est.feature_importances_
            if len(importances)==len(feat_names):
                fim = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=False)
                os.makedirs(args.results_dir, exist_ok=True)
                fim.to_csv(os.path.join(args.results_dir, f"GB_feature_importance_time_{rep['train_years']}_{args.test_year}.csv"), index=False)
        else:
            # Permutation importance trên test nếu không có FI sẵn
            pi = permutation_importance(estimator=pipe, X=X_te, y=y_te, n_repeats=5, random_state=args.random_state, n_jobs=-1)
            fim = pd.DataFrame({'feature': feat_names, 'importance_mean': pi.importances_mean, 'importance_std': pi.importances_std}) \
                    .sort_values('importance_mean', ascending=False)
            fim.to_csv(os.path.join(args.results_dir, f"GB_perm_importance_time_{rep['train_years']}_{args.test_year}.csv"), index=False)
    except Exception as e:
        print(f"[WARN] Không xuất được importance: {e}")

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
                grp.to_csv(os.path.join(args.results_dir, f"GB_by_{gcol}_time_{rep['train_years']}_{args.test_year}.csv"), index=False)
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
            scale_numeric=args.scale_numeric, drop_first=args.drop_first
        )

        gb = make_gb(args.engine, args.n_estimators, args.learning_rate, args.max_depth,
                     args.subsample, args.colsample_bytree, args.reg_lambda, args.reg_alpha,
                     args.min_samples_leaf, args.random_state)

        pipe = Pipeline([('prep', preproc), ('est', gb)])
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
        'model': 'GB('+('xgb' if args.engine=='xgb' and _HAS_XGB else 'skgb')+')',
        'run': 'cv', 'cv': args.cv,
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
    parser.add_argument('--engine', type=str, choices=['xgb','skgb'], default='xgb')
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
    parser.add_argument('--scale_numeric', action='store_true')
    parser.add_argument('--onehot_geo', type=str, default='province', choices=['none','province','district','commune','all'])
    parser.add_argument('--year_as_category', action='store_true')
    parser.add_argument('--drop_first', action='store_true')
    parser.add_argument('--results_dir', type=str, default='./results_gb')
    # Hyperparams (áp cho cả engine; xgb dùng đủ, skgb dùng subset)
    parser.add_argument('--n_estimators', type=int, default=2000)
    parser.add_argument('--learning_rate', type=float, default=0.03)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--subsample', type=float, default=0.9)
    parser.add_argument('--colsample_bytree', default=0.9)
    parser.add_argument('--reg_lambda', type=float, default=2.0)
    parser.add_argument('--reg_alpha', type=float, default=0.0)
    parser.add_argument('--min_samples_leaf', type=int, default=1)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--tune', action='store_true', help='GridSearchCV nhỏ (chỉ khi engine=xgb)')
    args = parser.parse_args()

    if args.engine == 'xgb' and not _HAS_XGB:
        print("[WARN] xgboost chưa cài. Tự động chuyển sang --engine skgb (HistGradientBoosting).")
        args.engine = 'skgb'

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
        out_csv = os.path.join(args.results_dir, f"GB_{args.target}_time_{train_label}_{args.test_year}.csv")
        res.to_csv(out_csv, index=False)
        print("=== Time-split GB Results ===")
        print(res.to_string(index=False))
        print(f"[DONE] Saved: {out_csv}")
    else:
        df_all = pd.concat([df2010, df2014, df2016], ignore_index=True, sort=False)
        res = run_cv(df_all, args.target, args)
        out_csv = os.path.join(args.results_dir, f"GB_{args.target}_cv_{args.cv}fold.csv")
        res.to_csv(out_csv, index=False)
        print("=== CV GB Results ===")
        print(res.to_string(index=False))
        print(f"[DONE] Saved: {out_csv}")

if __name__ == '__main__':
    main()
