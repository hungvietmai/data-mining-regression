"""
Baseline Linear Models (schema-driven, time-split)
-------------------------------------------------
Train baseline models for tabular survey data with a time-based split.
Supported models:
  - OLS (statsmodels if available; else sklearn LinearRegression fallback)
  - HuberRegressor (robust to outliers)

Features
- Schema-driven preprocessing from YAML: numeric / categorical_binary / categorical_ordinal
- Safe preprocessing: impute, optional winsorize, optional scaling, one-hot (drop_first optional)
- Target transform: none | log1p (with Duan smearing on back-transform) | asinh
- Leakage control for income-like targets
- Metrics on modeling scale and on raw scale (if transform used)
- Save coefficients/feature importances and group errors (Kinh/tinh) if columns exist
- Save predictions (optional)

Example
-------
python baseline_linear_time_split.py \
  --data_dir ./data --schema_path ./data_schema.yaml \
  --run time --target AgrInc --train_years 2010 2014 --test_year 2016 \
  --log_target log1p --smearing \
  --winsor_lower 0.05 --winsor_upper 0.95 --drop_first --scale_numeric \
  --model both --results_dir ./results_baseline
"""

from __future__ import annotations
import os
import sys
import argparse
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

# YAML schema
try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None

from packaging import version
from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Linear models
from sklearn.linear_model import LinearRegression, HuberRegressor

# Try statsmodels for true OLS (otherwise fallback to sklearn LinearRegression)
try:
    import statsmodels.api as sm
    _HAS_SM = True
except Exception:
    _HAS_SM = False

# -----------------
# Utils
# -----------------

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def onehot_compat(drop='first') -> OneHotEncoder:
    """Compat giữa sklearn <1.2 và >=1.2 (sparse vs sparse_output)."""
    if version.parse(sklearn_version) >= version.parse("1.2"):
        return OneHotEncoder(handle_unknown='ignore', drop=drop, sparse_output=False)
    return OneHotEncoder(handle_unknown='ignore', drop=drop, sparse=False)


class QuantileClipper:
    """Winsorize numeric theo phân vị (dùng trong Pipeline)."""
    def __init__(self, lower: float = 0.0, upper: float = 1.0):
        self.lower = lower
        self.upper = upper
        self.lo_: Optional[np.ndarray] = None
        self.hi_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.lower <= 0.0 and self.upper >= 1.0:
            self.lo_ = None
            self.hi_ = None
            return self
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        self.hi_ = np.maximum(self.hi_, self.lo_ + 1e-12)
        return self

    def transform(self, X):
        if self.lo_ is None:
            return X
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.clip(X, self.lo_, self.hi_)


def duan_smearing(y_true_log: np.ndarray, y_pred_log: np.ndarray, clip_range: Optional[Tuple[float,float]] = None) -> float:
    resid = y_true_log - y_pred_log
    s = float(np.mean(np.exp(resid)))
    if clip_range is not None:
        s = float(np.clip(s, clip_range[0], clip_range[1]))
    return s


def evaluate(y_true, y_pred) -> Dict[str, float]:
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': rmse(y_true, y_pred),
        'R2': float(r2_score(y_true, y_pred))
    }


def get_feature_names(preproc: ColumnTransformer) -> List[str]:
    names: List[str] = []
    for name, transformer, cols in preproc.transformers_:
        if name == 'remainder' and transformer == 'drop':
            continue
        if hasattr(transformer, 'named_steps'):
            last = list(transformer.named_steps.values())[-1]
            if hasattr(last, 'get_feature_names_out'):
                try:
                    feats = last.get_feature_names_out(cols)
                except Exception:
                    feats = cols
            else:
                feats = cols
        else:
            feats = cols
        names.extend(list(feats))
    return names


# -----------------
# Data I/O & Schema
# -----------------

def load_year(data_dir: str, year: int) -> pd.DataFrame:
    candidates = [f"{year}_clean.csv", f"{year}.csv", f"{year}.dta"]
    for name in candidates:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            print(f"[INFO] Loading {p} ...")
            if name.endswith('.csv'):
                return pd.read_csv(p)
            else:
                return pd.read_stata(p)
    raise FileNotFoundError(f"Không thấy file cho năm {year} trong {data_dir} (tìm: {candidates})")


def load_schema(schema_path: str) -> Dict[str, List[str]]:
    if yaml is None:
        print("[ERROR] Thiếu thư viện PyYAML. Cài đặt: pip install pyyaml", file=sys.stderr)
        sys.exit(2)
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = yaml.safe_load(f)
    for k in ['numeric', 'categorical_binary', 'categorical_ordinal']:
        if k not in schema:
            schema[k] = []
    return schema


def enforce_schema(df: pd.DataFrame, schema: Dict[str, List[str]]) -> pd.DataFrame:
    df = df.copy()
    for c in schema.get('numeric', []):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in schema.get('categorical_binary', []) + schema.get('categorical_ordinal', []):
        if c in df.columns:
            df[c] = df[c].astype('category')
    return df


# -----------------
# Preprocessor builder
# -----------------

def build_preprocessor_from_schema(
    X: pd.DataFrame,
    schema: Dict[str, List[str]],
    exclude_cols: List[str],
    winsor_lower: float,
    winsor_upper: float,
    scale_numeric: bool,
    drop_first: bool,
    year_as_category: bool,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    df = X.copy()

    # Treat year as category if requested
    if year_as_category:
        for yc in ['year', 'Year']:
            if yc in df.columns:
                df[yc] = df[yc].astype(str)

    # Select columns by schema
    num_cols = [c for c in schema.get('numeric', []) if c in df.columns and c not in exclude_cols]
    cat_cols = [c for c in (schema.get('categorical_binary', []) + schema.get('categorical_ordinal', []))
                if c in df.columns and c not in exclude_cols]

    # Warn missing schema columns
    missing = [c for c in schema.get('numeric', []) + schema.get('categorical_binary', []) + schema.get('categorical_ordinal', [])
               if c not in df.columns and c not in exclude_cols]
    if missing:
        print(f"[WARN] Các cột trong schema không có trong dữ liệu và sẽ bỏ qua: {missing}")

    num_steps = [('imputer', SimpleImputer(strategy='median'))]
    if winsor_lower > 0.0 or winsor_upper < 1.0:
        num_steps.append(('winsor', QuantileClipper(winsor_lower, winsor_upper)))
    if scale_numeric:
        num_steps.append(('scaler', StandardScaler(with_mean=True, with_std=True)))

    cat_steps = [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', onehot_compat(drop='first' if drop_first else None)),
    ]

    preproc = ColumnTransformer(
        transformers=[
            ('num', Pipeline(num_steps), num_cols),
            ('cat', Pipeline(cat_steps), cat_cols),
        ],
        remainder='drop',
    )
    return preproc, num_cols, cat_cols


# -----------------
# Targets & exclusions
# -----------------

def make_y(y: pd.Series, how: str):
    y = y.copy()
    if how == 'none':
        return y.values.astype(float), {'transform': lambda v: v, 'invert': lambda v: v}
    if how == 'log1p':
        y = y.fillna(0.0).clip(lower=0.0)
        return np.log1p(y.values.astype(float)), {'transform': np.log1p, 'invert': np.expm1}
    if how == 'asinh':
        return np.arcsinh(y.values.astype(float)), {'transform': np.arcsinh, 'invert': np.sinh}
    raise ValueError("log_target phải thuộc {none, log1p, asinh}")


def get_exclusions(target: str, exclude_related_income: bool) -> List[str]:
    excl = ['ma_ho']
    if exclude_related_income:
        if target == 'AgrInc':
            excl += ['TotalInc', 'Wage', 'CropInc', 'CropValue']
        elif target == 'CropInc':
            excl += ['TotalInc', 'Wage', 'AgrInc', 'CropValue']
        else:
            excl += ['TotalInc', 'Wage']
    return excl


# -----------------
# Fit helpers
# -----------------

def fit_ols(X_tr_t: np.ndarray, y_tr: np.ndarray):
    if _HAS_SM:
        Xc = sm.add_constant(X_tr_t, has_constant='add')
        model = sm.OLS(y_tr, Xc)
        res = model.fit()
        return res
    else:
        lr = LinearRegression(fit_intercept=True, n_jobs=None)
        lr.fit(X_tr_t, y_tr)
        return lr


def predict_ols(model, X_te_t: np.ndarray) -> np.ndarray:
    if _HAS_SM and hasattr(model, 'predict') and not isinstance(model, LinearRegression):
        Xc_te = sm.add_constant(X_te_t, has_constant='add')
        return np.asarray(model.predict(Xc_te))
    else:
        return np.asarray(model.predict(X_te_t))


# -----------------
# Training (time-split)
# -----------------

def run_time_split(train_dfs: List[pd.DataFrame], test_df: pd.DataFrame, target: str, args, schema: Dict[str, List[str]]) -> pd.DataFrame:
    df_train = pd.concat(train_dfs, ignore_index=True, sort=False)

    # enforce schema types
    df_train = enforce_schema(df_train, schema)
    test_df  = enforce_schema(test_df, schema)

    exclude_cols = get_exclusions(target, args.exclude_related_income)

    # target transform
    y_tr, ymeta = make_y(df_train[target], args.log_target)
    y_te, _     = make_y(test_df[target], args.log_target)

    mask_tr = ~np.isnan(y_tr)
    mask_te = ~np.isnan(y_te)
    X_tr = df_train.loc[mask_tr, :].drop(columns=[target])
    y_tr = y_tr[mask_tr]
    X_te = test_df.loc[mask_te, :].drop(columns=[target])
    y_te = y_te[mask_te]

    preproc, _, _ = build_preprocessor_from_schema(
        X_tr, schema=schema, exclude_cols=exclude_cols,
        winsor_lower=args.winsor_lower, winsor_upper=args.winsor_upper,
        scale_numeric=args.scale_numeric, drop_first=args.drop_first,
        year_as_category=args.year_as_category,
    )

    # Fit preprocessor on TRAIN only, then transform both
    preproc.fit(X_tr, y_tr)
    X_tr_t = preproc.transform(X_tr)
    X_te_t = preproc.transform(X_te)

    # Feature names for coefficient export
    feat_names = []
    try:
        # Try sklearn >=1.0 get_feature_names_out
        feat_names = preproc.get_feature_names_out()
    except Exception:
        feat_names = []

    reports: List[Dict[str, object]] = []

    # ---------------- OLS ----------------
    if args.model in ('ols', 'both'):
        ols_model = fit_ols(X_tr_t, y_tr)
        yhat_te = predict_ols(ols_model, X_te_t)

        rep = evaluate(y_te, yhat_te)
        # Back-transform
        if args.log_target in ('log1p', 'asinh'):
            inv = ymeta['invert']
            if args.log_target == 'log1p':
                # Duan smearing on TRAIN
                yhat_tr = predict_ols(ols_model, X_tr_t)
                s = duan_smearing(y_tr, yhat_tr, clip_range=(0.5, 1.5)) if args.smearing else 1.0
                y_true_raw = inv(y_te)
                y_pred_raw = s * inv(yhat_te)
                y_pred_raw = np.maximum(y_pred_raw, 0.0)
                rep['smearing'] = s
            else:
                y_true_raw = inv(y_te)
                y_pred_raw = inv(yhat_te)
            rep.update({
                'MAE_raw': float(mean_absolute_error(y_true_raw, y_pred_raw)),
                'RMSE_raw': rmse(y_true_raw, y_pred_raw),
                'R2_raw': float(r2_score(y_true_raw, y_pred_raw)),
            })

        rep.update({'model': 'OLS' if _HAS_SM else 'LinearRegression', 'run': 'time',
                    'train_years': ','.join(str(y) for y in args.train_years) if args.train_years else str(args.train_year),
                    'test_year': args.test_year, 'target': target})
        reports.append(rep)

        # Save coefficients
        try:
            os.makedirs(args.results_dir, exist_ok=True)
            if _HAS_SM and hasattr(ols_model, 'params'):
                params = ols_model.params
                if feat_names:
                    # align: statsmodels adds const at index 0
                    coef_df = pd.DataFrame({'feature': ['const'] + list(feat_names), 'coef': params.values})
                else:
                    coef_df = pd.DataFrame({'feature': list(params.index), 'coef': params.values})
            else:
                coefs = getattr(ols_model, 'coef_', None)
                intercept = getattr(ols_model, 'intercept_', 0.0)
                rows = []
                rows.append({'feature': 'intercept', 'coef': float(intercept)})
                if coefs is not None:
                    for i, c in enumerate(coefs):
                        fname = feat_names[i] if i < len(feat_names) else f'f_{i}'
                        rows.append({'feature': fname, 'coef': float(c)})
                coef_df = pd.DataFrame(rows)
            coef_df.to_csv(os.path.join(args.results_dir, f"OLS_coeffs_time_{rep['train_years']}_{args.test_year}.csv"), index=False)
        except Exception as e:
            print(f"[WARN] Không xuất được coefficients OLS: {e}")

        # Group errors & predictions
        try:
            df_eval = X_te.copy()
            if args.log_target in ('log1p', 'asinh'):
                df_eval['y_true'] = y_true_raw
                df_eval['y_pred'] = y_pred_raw
            else:
                df_eval['y_true'] = y_te
                df_eval['y_pred'] = yhat_te
            df_eval['abs_err'] = np.abs(df_eval['y_true'] - df_eval['y_pred'])
            for gcol in ['Kinh', 'tinh']:
                if gcol in df_eval.columns:
                    grp = df_eval.groupby(gcol, observed=False)['abs_err'].mean().reset_index().rename(columns={'abs_err': 'MAE_group'})
                    grp.to_csv(os.path.join(args.results_dir, f"OLS_by_{gcol}_time_{rep['train_years']}_{args.test_year}.csv"), index=False)
            if args.save_predictions:
                out_pred = X_te.copy()
                out_pred[target + '_true'] = df_eval['y_true']
                out_pred[target + '_pred'] = df_eval['y_pred']
                out_pred.to_csv(os.path.join(args.results_dir, f"OLS_preds_time_{rep['train_years']}_{args.test_year}.csv"), index=False)
        except Exception as e:
            print(f"[WARN] Không xuất được group errors / predictions (OLS): {e}")

    # ---------------- Huber ----------------
    if args.model in ('huber', 'both'):
        huber = HuberRegressor(
            epsilon=args.huber_epsilon,
            alpha=args.huber_alpha,
            max_iter=args.huber_max_iter,
            fit_intercept=True,
        )
        huber.fit(X_tr_t, y_tr)
        yhat_te = huber.predict(X_te_t)

        rep = evaluate(y_te, yhat_te)
        # Back-transform
        if args.log_target in ('log1p', 'asinh'):
            inv = ymeta['invert']
            if args.log_target == 'log1p':
                yhat_tr = huber.predict(X_tr_t)
                s = duan_smearing(y_tr, yhat_tr, clip_range=(0.5, 1.5)) if args.smearing else 1.0
                y_true_raw = inv(y_te)
                y_pred_raw = s * inv(yhat_te)
                y_pred_raw = np.maximum(y_pred_raw, 0.0)
                rep['smearing'] = s
            else:
                y_true_raw = inv(y_te)
                y_pred_raw = inv(yhat_te)
            rep.update({
                'MAE_raw': float(mean_absolute_error(y_true_raw, y_pred_raw)),
                'RMSE_raw': rmse(y_true_raw, y_pred_raw),
                'R2_raw': float(r2_score(y_true_raw, y_pred_raw)),
            })

        rep.update({'model': 'Huber', 'run': 'time',
                    'train_years': ','.join(str(y) for y in args.train_years) if args.train_years else str(args.train_year),
                    'test_year': args.test_year, 'target': target})
        reports.append(rep)

        # Save coefficients
        try:
            os.makedirs(args.results_dir, exist_ok=True)
            coefs = getattr(huber, 'coef_', None)
            intercept = getattr(huber, 'intercept_', 0.0)
            rows = []
            rows.append({'feature': 'intercept', 'coef': float(intercept)})
            if coefs is not None:
                for i, c in enumerate(coefs):
                    fname = feat_names[i] if i < len(feat_names) else f'f_{i}'
                    rows.append({'feature': fname, 'coef': float(c)})
            coef_df = pd.DataFrame(rows)
            coef_df.to_csv(os.path.join(args.results_dir, f"Huber_coeffs_time_{rep['train_years']}_{args.test_year}.csv"), index=False)
        except Exception as e:
            print(f"[WARN] Không xuất được coefficients Huber: {e}")

        # Group errors & predictions
        try:
            df_eval = X_te.copy()
            if args.log_target in ('log1p', 'asinh'):
                df_eval['y_true'] = y_true_raw
                df_eval['y_pred'] = y_pred_raw
            else:
                df_eval['y_true'] = y_te
                df_eval['y_pred'] = yhat_te
            df_eval['abs_err'] = np.abs(df_eval['y_true'] - df_eval['y_pred'])
            for gcol in ['Kinh', 'tinh']:
                if gcol in df_eval.columns:
                    grp = df_eval.groupby(gcol, observed=False)['abs_err'].mean().reset_index().rename(columns={'abs_err': 'MAE_group'})
                    grp.to_csv(os.path.join(args.results_dir, f"Huber_by_{gcol}_time_{rep['train_years']}_{args.test_year}.csv"), index=False)
            if args.save_predictions:
                out_pred = X_te.copy()
                out_pred[target + '_true'] = df_eval['y_true']
                out_pred[target + '_pred'] = df_eval['y_pred']
                out_pred.to_csv(os.path.join(args.results_dir, f"Huber_preds_time_{rep['train_years']}_{args.test_year}.csv"), index=False)
        except Exception as e:
            print(f"[WARN] Không xuất được group errors / predictions (Huber): {e}")

    # return combined report
    return pd.DataFrame(reports)


# -----------------
# Main
# -----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--schema_path', type=str, required=True)
    p.add_argument('--run', choices=['time'], default='time', help='Chỉ hỗ trợ time-split')
    p.add_argument('--target', type=str, default='AgrInc')

    # Chọn năm
    p.add_argument('--train_year', type=int, default=2014)
    p.add_argument('--train_years', nargs='*', type=int, default=None)
    p.add_argument('--test_year', type=int, default=2016)

    # Preprocess
    p.add_argument('--winsor_lower', type=float, default=0.0)
    p.add_argument('--winsor_upper', type=float, default=1.0)
    p.add_argument('--scale_numeric', action='store_true', help='Highly recommended for Huber')
    p.add_argument('--drop_first', action='store_true')
    p.add_argument('--year_as_category', action='store_true')
    p.add_argument('--exclude_related_income', action='store_true')

    # Target transform
    p.add_argument('--log_target', choices=['none','log1p','asinh'], default='log1p')
    p.add_argument('--smearing', action='store_true', help='Only for log1p back-transform')

    # Model selection
    p.add_argument('--model', choices=['ols','huber','both'], default='both')

    # Huber params
    p.add_argument('--huber_epsilon', type=float, default=1.35, help='Outlier threshold (larger = less robust)')
    p.add_argument('--huber_alpha', type=float, default=0.0001, help='L2 regularization strength')
    p.add_argument('--huber_max_iter', type=int, default=1000)

    # I/O
    p.add_argument('--results_dir', type=str, default='./results_baseline')
    p.add_argument('--save_predictions', action='store_true')

    args = p.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Load data
    df2010 = load_year(args.data_dir, 2010)
    df2014 = load_year(args.data_dir, 2014)
    df2016 = load_year(args.data_dir, 2016)

    # Load schema
    schema = load_schema(args.schema_path)

    if args.run != 'time':
        print('[ERROR] Chỉ hỗ trợ --run time trong bản này.'); sys.exit(2)

    # Chọn train/test theo năm
    if args.train_years:
        train_map = {2010: df2010, 2014: df2014, 2016: df2016}
        try:
            train_dfs = [train_map[y] for y in args.train_years]
        except KeyError as e:
            print(f"[ERROR] Năm train không hợp lệ: {e}"); sys.exit(2)
        train_label = ','.join(str(y) for y in args.train_years)
    else:
        train_dfs = [df2014 if args.train_year == 2014 else (df2010 if args.train_year == 2010 else df2016)]
        train_label = str(args.train_year)

    test_df = df2016 if args.test_year == 2016 else (df2014 if args.test_year == 2014 else df2010)

    # Run
    res = run_time_split(train_dfs, test_df, args.target, args, schema)

    # Save results
    out_csv = os.path.join(args.results_dir, f"BASELINE_{args.model}_{args.target}_time_{train_label}_{args.test_year}.csv")
    res.to_csv(out_csv, index=False)
    print("=== Time-split Baseline Results ===")
    print(res.to_string(index=False))
    print(f"[DONE] Saved: {out_csv}")


if __name__ == '__main__':
    main()
