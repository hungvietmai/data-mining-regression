"""
XGB Regression (schema-driven)
---------------------------------
Huấn luyện mô hình XGBoost để dự báo biến mục tiêu (mặc định: AgrInc)
trên dữ liệu khảo sát nhiều năm. Bản này đọc cấu hình biến từ data_schema.yaml.

Điểm nổi bật:
- Đọc schema YAML để ép kiểu & chọn cột: numeric / categorical_binary / categorical_ordinal
- Tiền xử lý: impute, winsorize (tuỳ chọn), One-Hot Encoder (drop_first tuỳ chọn)
- Biến đổi mục tiêu: none | log1p (kèm Duan smearing) | asinh (xử lý âm/0/dương)
- XGBoost có L1/L2 (reg_alpha/reg_lambda), early stopping với validation split
- Exclude các biến dễ gây rò rỉ khi dự báo AgrInc/CropInc
- Xuất metrics (MAE/RMSE/R2) cả trên thang học và thang gốc (nếu dùng transform)
- Lưu feature importance và lỗi theo nhóm (Kinh/tinh) nếu cột tồn tại
"""

import os
import sys
import argparse
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

# YAML schema
try:
    import yaml  # pip install pyyaml
except Exception as e:
    yaml = None

from packaging import version
from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# XGBoost
_HAS_XGB = True
try:
    from xgboost import XGBRegressor
except Exception:
    _HAS_XGB = False
    from sklearn.ensemble import HistGradientBoostingRegressor  # fallback

# -----------------
# Helpers
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
    def __init__(self, lower=0.0, upper=1.0):
        self.lower = lower
        self.upper = upper
        self.lo_ = None
        self.hi_ = None

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
    """Smearing factor s = mean(exp(residual)) trên train-log.
    Tuỳ chọn cắt s trong [lo, hi] để ổn định.
    """
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
    names = []
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
    # đảm bảo các khoá tồn tại
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
# Preprocess builder
# -----------------

def build_preprocessor_from_schema(X: pd.DataFrame,
                                   schema: Dict[str, List[str]],
                                   exclude_cols: List[str],
                                   winsor_lower: float,
                                   winsor_upper: float,
                                   scale_numeric: bool,
                                   drop_first: bool,
                                   year_as_category: bool) -> Tuple[ColumnTransformer, List[str], List[str]]:
    df = X.copy()
    # Optionally treat 'year' as category
    if year_as_category and 'year' in df.columns:
        df['year'] = df['year'].astype(str)

    # Intersect schema lists with existing columns (tránh thiếu cột)
    num_cols = [c for c in schema.get('numeric', []) if c in df.columns and c not in exclude_cols]
    cat_cols = [c for c in (schema.get('categorical_binary', []) + schema.get('categorical_ordinal', []))
                if c in df.columns and c not in exclude_cols]

    # Cảnh báo nếu schema chỉ ra cột không tồn tại
    missing = [c for c in schema.get('numeric', []) + schema.get('categorical_binary', []) + schema.get('categorical_ordinal', [])
               if c not in df.columns and c not in exclude_cols]
    if missing:
        print(f"[WARN] Các cột trong schema không thấy trong dữ liệu hiện tại và sẽ bỏ qua: {missing}")

    num_steps = [('imputer', SimpleImputer(strategy='median'))]
    if winsor_lower > 0.0 or winsor_upper < 1.0:
        num_steps.append(('winsor', QuantileClipper(winsor_lower, winsor_upper)))
    if scale_numeric:
        num_steps.append(('scaler', StandardScaler(with_mean=True, with_std=True)))

    cat_steps = [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', onehot_compat(drop='first' if drop_first else None))
    ]

    preproc = ColumnTransformer(
        transformers=[
            ('num', Pipeline(num_steps), num_cols),
            ('cat', Pipeline(cat_steps), cat_cols)
        ],
        remainder='drop'
    )
    return preproc, num_cols, cat_cols


# -----------------
# Target transforms
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
# Train/Eval (time-split only)
# -----------------

def run_time_split(train_dfs: List[pd.DataFrame], test_df: pd.DataFrame, target: str, args, schema: Dict[str, List[str]]) -> pd.DataFrame:
    df_train = pd.concat(train_dfs, ignore_index=True, sort=False)

    # ép kiểu theo schema (an toàn)
    df_train = enforce_schema(df_train, schema)
    test_df  = enforce_schema(test_df, schema)

    exclude_cols = get_exclusions(target, args.exclude_related_income)

    # y
    y_tr_raw = df_train[target]
    y_te_raw = test_df[target]
    y_tr, ymeta = make_y(y_tr_raw, args.log_target)
    y_te, _     = make_y(y_te_raw, args.log_target)

    mask_tr = ~np.isnan(y_tr)
    mask_te = ~np.isnan(y_te)
    X_tr = df_train.loc[mask_tr, :].drop(columns=[target])
    y_tr = y_tr[mask_tr]
    X_te = test_df.loc[mask_te, :].drop(columns=[target])
    y_te = y_te[mask_te]

    preproc, num_cols, cat_cols = build_preprocessor_from_schema(
        X_tr, schema=schema, exclude_cols=exclude_cols,
        winsor_lower=args.winsor_lower, winsor_upper=args.winsor_upper,
        scale_numeric=args.scale_numeric, drop_first=args.drop_first,
        year_as_category=args.year_as_category
    )

    # Estimator
    if _HAS_XGB:
        est = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda,
            reg_alpha=args.reg_alpha,
            min_child_weight=args.min_child_weight,
            gamma=args.gamma,
            n_jobs=-1,
            random_state=args.random_state
        )
    else:
        print("[WARN] xgboost chưa cài; dùng HistGradientBoostingRegressor làm fallback (không có L1).")
        est = HistGradientBoostingRegressor(
            max_iter=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth if args.max_depth and args.max_depth>0 else None,
            l2_regularization=args.reg_lambda,
            random_state=args.random_state
        )

    pipe = Pipeline([('prep', preproc), ('est', est)])

    # Early stopping (chỉ với XGB). Tách một phần train làm validation.
    if _HAS_XGB and args.early_stopping_rounds > 0 and 0.0 < args.val_size < 0.5:
        X_tr2, X_val, y_tr2, y_val = train_test_split(X_tr, y_tr, test_size=args.val_size, random_state=args.random_state)
        # fit preproc riêng rồi transform -> tránh leak
        pipe.named_steps['prep'].fit(X_tr2, y_tr2)
        X_tr2t = pipe.named_steps['prep'].transform(X_tr2)
        X_valt = pipe.named_steps['prep'].transform(X_val)
        est = pipe.named_steps['est']
        est.set_params(early_stopping_rounds=args.early_stopping_rounds)
        est.fit(X_tr2t, y_tr2, eval_set=[(X_valt, y_val)], verbose=False)
    else:
        pipe.fit(X_tr, y_tr)

    # Predict & evaluate (trên scale học)
    yhat_te = pipe.predict(X_te)
    rep = evaluate(y_te, yhat_te)

    # Back-transform nếu cần
    if args.log_target in ('log1p', 'asinh'):
        inv = ymeta['invert']
        if args.log_target == 'log1p':
            if args.smearing:
                # smearing tính trên train (log)
                yhat_tr = pipe.predict(X_tr)
                s = duan_smearing(y_tr, yhat_tr, clip_range=(0.5, 1.5))
            else:
                s = 1.0
            y_true_raw = inv(y_te)
            y_pred_raw = s * inv(yhat_te)
            y_pred_raw = np.maximum(y_pred_raw, 0.0)
            rep['smearing'] = s
        else:  # asinh
            y_true_raw = inv(y_te)
            y_pred_raw = inv(yhat_te)
        rep.update({
            'MAE_raw': float(mean_absolute_error(y_true_raw, y_pred_raw)),
            'RMSE_raw': rmse(y_true_raw, y_pred_raw),
            'R2_raw': float(r2_score(y_true_raw, y_pred_raw))
        })

    # Thông tin run
    rep.update({
        'model': 'XGB' if _HAS_XGB else 'skHGB',
        'run': 'time',
        'train_years': ','.join(str(y) for y in args.train_years) if args.train_years else str(args.train_year),
        'test_year': args.test_year,
        'target': target
    })

    # Lưu feature importance nếu có
    try:
        est = pipe.named_steps['est']
        prep = pipe.named_steps['prep']
        feat_names = get_feature_names(prep)
        if hasattr(est, 'feature_importances_') and len(feat_names) == len(est.feature_importances_):
            fim = pd.DataFrame({'feature': feat_names, 'importance': est.feature_importances_}) \
                    .sort_values('importance', ascending=False)
            os.makedirs(args.results_dir, exist_ok=True)
            fim.to_csv(os.path.join(args.results_dir, f"XGB_feature_importance_time_{rep['train_years']}_{args.test_year}.csv"), index=False)
    except Exception as e:
        print(f"[WARN] Không xuất được feature importance: {e}")

    # Lỗi theo nhóm (Kinh/tinh) trên thang gốc (nếu có transform), ngược lại dùng scale học
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
                grp.to_csv(os.path.join(args.results_dir, f"XGB_by_{gcol}_time_{rep['train_years']}_{args.test_year}.csv"), index=False)
    except Exception as e:
        print(f"[WARN] Không xuất được group error: {e}")

    # Lưu dự đoán nếu yêu cầu
    try:
        if args.save_predictions:
            out_pred = X_te.copy()
            out_pred[target + '_true'] = df_eval['y_true'] if 'y_true' in df_eval.columns else y_te
            out_pred[target + '_pred'] = df_eval['y_pred'] if 'y_pred' in df_eval.columns else yhat_te
            out_pred.to_csv(os.path.join(args.results_dir, f"XGB_preds_time_{rep['train_years']}_{args.test_year}.csv"), index=False)
    except Exception as e:
        print(f"[WARN] Không lưu được predictions: {e}")

    return pd.DataFrame([rep])


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
    p.add_argument('--scale_numeric', action='store_true')
    p.add_argument('--drop_first', action='store_true')
    p.add_argument('--year_as_category', action='store_true')
    p.add_argument('--exclude_related_income', action='store_true')

    # Target transform
    p.add_argument('--log_target', choices=['none','log1p','asinh'], default='log1p')
    p.add_argument('--smearing', action='store_true', help='Chỉ dùng cho log1p khi back-transform')

    # XGB hyperparams
    p.add_argument('--n_estimators', type=int, default=1500)
    p.add_argument('--learning_rate', type=float, default=0.03)
    p.add_argument('--max_depth', type=int, default=6)
    p.add_argument('--subsample', type=float, default=0.9)
    p.add_argument('--colsample_bytree', type=float, default=0.9)
    p.add_argument('--reg_lambda', type=float, default=2.0)
    p.add_argument('--reg_alpha', type=float, default=0.0)
    p.add_argument('--min_child_weight', type=float, default=1.0)
    p.add_argument('--gamma', type=float, default=0.0)
    p.add_argument('--random_state', type=int, default=42)

    # Early stopping
    p.add_argument('--early_stopping_rounds', type=int, default=0, help='>0 để bật; dùng val_size để tách validation')
    p.add_argument('--val_size', type=float, default=0.0, help='Tỷ lệ validation từ train (0..0.5)')

    # I/O
    p.add_argument('--results_dir', type=str, default='./results_xgb')
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

    # Chạy
    res = run_time_split(train_dfs, test_df, args.target, args, schema)

    # Lưu kết quả
    out_csv = os.path.join(args.results_dir, f"XGB_{args.target}_time_{train_label}_{args.test_year}.csv")
    res.to_csv(out_csv, index=False)
    print("=== Time-split XGB Results ===")
    print(res.to_string(index=False))
    print(f"[DONE] Saved: {out_csv}")


if __name__ == '__main__':
    main()
