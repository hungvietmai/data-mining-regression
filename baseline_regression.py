"""
baseline_regression.py — OLS & Huber baseline cho dự báo thu nhập nông nghiệp

Chức năng:
- Chạy 2 chế độ:
    * time: Train 2014 -> Test 2016 (temporal generalization)
    * cv:   GroupKFold 10 trên toàn bộ dữ liệu (IID generalization, nhóm theo hộ)
- Target: AgrInc (mặc định) hoặc CropInc
- Tuỳ chọn log-transform mục tiêu (--log_target log1p) để ổn định phương sai
- Tiền xử lý: impute, (tùy chọn) winsorization, one-hot categorical, (tùy chọn) scale numeric
- Mô hình: OLS (LinearRegression) & HuberRegressor
- Đánh giá: MAE, RMSE, R2. Với log-target, báo cáo thêm MAE/RMSE/R2 trên thang gốc.
- Xuất: results_baseline/*.csv (kết quả tổng hợp, hệ số OLS/Huber, lỗi theo nhóm Kinh/tinh)
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
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn import __version__ as sklearn_version
from packaging import version

# -----------------
# Utility
# -----------------
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def onehot_compat(drop='first'):
    # Luôn trả dense output; tương thích sklearn>=1.2 và cũ hơn
    if version.parse(sklearn_version) >= version.parse("1.2"):
        return OneHotEncoder(handle_unknown='ignore', drop=drop, sparse_output=False)
    return OneHotEncoder(handle_unknown='ignore', drop=drop, sparse=False)

class QuantileClipper:
    """Transformer đơn giản để winsorize numeric theo phân vị (dùng trong Pipeline)."""
    def __init__(self, lower=0.0, upper=1.0):
        self.lower = lower
        self.upper = upper
        self.lo_ = None
        self.hi_ = None
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: X = X.reshape(-1,1)
        if self.lower <= 0.0 and self.upper >= 1.0:
            # no-op
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
    p = os.path.join(data_dir, f"{year}.dta")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Không thấy file {p}")
    return pd.read_stata(p)

# -----------------
# Preprocessing
# -----------------
BINARY_LIKE = [
    "Kinh","Poor","Moneysupp","Electricity","Male","Married",
    "Treat","Contract","Rice","Veg","Coffee","Tea","natShock","ecoShock","socShock"
]

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
    # Numeric = số trừ các cột categorical
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
    # winsorization optional
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
    """mode: 'raw' | 'log1p'
    - Với 'log1p': clip giá trị âm lên 0 để tránh cảnh báo/NaN khi log1p.
    """
    s = pd.to_numeric(series, errors='coerce').values.astype(float)
    meta = {'mode': mode}
    if mode == 'raw':
        y = s
        meta['invert'] = lambda arr: arr
    else:
        s = np.where(np.isfinite(s), s, np.nan)
        s = np.nan_to_num(s, nan=0.0)   # thay NaN bằng 0
        s = np.maximum(s, 0.0)          # clip âm lên 0
        y = np.log1p(s).astype(float)
        meta['invert'] = lambda arr: np.expm1(arr)
    return y, meta

def evaluate(y_true, y_pred) -> Dict[str, float]:
    return {'MAE': float(mean_absolute_error(y_true, y_pred)),
            'RMSE': rmse(y_true, y_pred),
            'R2': float(r2_score(y_true, y_pred))}

# -----------------
# Runs
# -----------------
def run_time_split(df_train: pd.DataFrame, df_test: pd.DataFrame, target: str, args) -> pd.DataFrame:
    # Exclusions to avoid leakage
    exclude_cols = ['ma_ho']
    if args.exclude_related_income:
        if target == 'AgrInc':
            exclude_cols += ['TotalInc','Wage','CropInc','CropValue']
        elif target == 'CropInc':
            exclude_cols += ['TotalInc','Wage','AgrInc']

    # Build X/y
    y_tr, ymeta = make_y(df_train[target], args.log_target)
    y_te, _ = make_y(df_test[target], args.log_target)

    # Drop rows where y is NaN
    mask_tr = ~np.isnan(y_tr)
    mask_te = ~np.isnan(y_te)
    X_tr = df_train.loc[mask_tr, :].drop(columns=[target])
    X_te = df_test.loc[mask_te, :].drop(columns=[target])
    y_tr = y_tr[mask_tr]
    y_te = y_te[mask_te]

    # Preprocessor: FIT TRÊN TRAIN CHỈ (tránh leakage)
    preproc, _, _ = build_preprocessor(
        X_tr,
        exclude_cols=exclude_cols, geo_level=args.onehot_geo, year_as_category=args.year_as_category,
        winsor_lower=args.winsor_lower, winsor_upper=args.winsor_upper,
        scale_numeric=args.scale_numeric, drop_first=args.drop_first
    )

    models = {
        'OLS': LinearRegression(),
        'HUBER': HuberRegressor(epsilon=1.5, alpha=1e-2, max_iter=2000)
    }

    rows = []
    for name, model in models.items():
        pipe = Pipeline([('prep', preproc), ('est', model)])
        pipe.fit(X_tr, y_tr)
        yhat = pipe.predict(X_te)
        rep = evaluate(y_te, yhat)

        # If log-target, also compute metrics on raw scale
        if args.log_target == 'log1p':
            inv = ymeta['invert']
            rep.update({
                'MAE_raw': float(mean_absolute_error(inv(y_te), inv(yhat))),
                'RMSE_raw': rmse(inv(y_te), inv(yhat)),
                'R2_raw': float(r2_score(inv(y_te), inv(yhat)))
            })
        rep.update({'model': name, 'run': 'time', 'train_year': args.train_year, 'test_year': args.test_year})
        rows.append(rep)

        # Save coefficients for linear models (trên TRAIN)
        try:
            prep = pipe.named_steps['prep']
            est = pipe.named_steps['est']
            feat_names = get_feature_names(prep)
            coef = getattr(est, 'coef_', None)
            if coef is not None and len(coef)==len(feat_names):
                coefs = pd.DataFrame({'feature': feat_names, 'coef': coef})
                out_path = os.path.join(args.results_dir, f"{name}_coeffs_time_{args.train_year}_{args.test_year}.csv")
                coefs.to_csv(out_path, index=False)
        except Exception as e:
            print(f"[WARN] Không xuất được coefficients cho {name}: {e}")

        # Group errors by Kinh and tinh on raw scale if possible
        try:
            df_eval = X_te.copy()
            df_eval['y_true'] = ymeta['invert'](y_te) if args.log_target=='log1p' else y_te
            df_eval['y_pred'] = ymeta['invert'](yhat) if args.log_target=='log1p' else yhat
            df_eval['abs_err'] = np.abs(df_eval['y_true'] - df_eval['y_pred'])
            for gcol in ['Kinh','tinh']:
                if gcol in df_eval.columns:
                    grp = df_eval.groupby(gcol, observed=False)['abs_err'].mean().reset_index().rename(columns={'abs_err':'MAE_group'})
                    outp = os.path.join(args.results_dir, f"{name}_by_{gcol}_time_{args.train_year}_{args.test_year}.csv")
                    grp.to_csv(outp, index=False)
        except Exception as e:
            print(f"[WARN] Không xuất được group error cho {name}: {e}")

    return pd.DataFrame(rows)

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

    # group by household để tránh leakage qua các năm
    if set(['tinh','quan','xa','ma_ho']).issubset(df_all.columns):
        groups = df_all.loc[mask, ['tinh','quan','xa','ma_ho']].astype(str).agg('-'.join, axis=1).values
        splitter = GroupKFold(n_splits=args.cv).split(X_all, y_all, groups=groups)
    else:
        splitter = KFold(n_splits=args.cv, shuffle=True, random_state=42).split(X_all, y_all)

    models = {'OLS': LinearRegression(), 'HUBER': HuberRegressor()}

    rows = []
    for name, model in models.items():
        mae_list, rmse_list, r2_list = [], [], []
        mae_raw_list, rmse_raw_list, r2_raw_list = [], [], []
        for tr_idx, te_idx in splitter:
            X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
            y_tr, y_te = y_all[tr_idx], y_all[te_idx]

            # Preprocessor FIT TRÊN TRAIN của fold (không leakage)
            preproc, _, _ = build_preprocessor(
                X_tr, exclude_cols=exclude_cols, geo_level=args.onehot_geo, year_as_category=args.year_as_category,
                winsor_lower=args.winsor_lower, winsor_upper=args.winsor_upper,
                scale_numeric=args.scale_numeric, drop_first=args.drop_first
            )

            pipe = Pipeline([('prep', preproc), ('est', model)])
            pipe.fit(X_tr, y_tr)
            yhat = pipe.predict(X_te)
            rep = evaluate(y_te, yhat)
            mae_list.append(rep['MAE']); rmse_list.append(rep['RMSE']); r2_list.append(rep['R2'])

            if args.log_target == 'log1p':
                inv = ymeta['invert']
                mae_raw_list.append(float(mean_absolute_error(inv(y_te), inv(yhat))))
                rmse_raw_list.append(rmse(inv(y_te), inv(yhat)))
                r2_raw_list.append(float(r2_score(inv(y_te), inv(yhat))))

        res = {
            'model': name, 'run': 'cv', 'cv': args.cv,
            'MAE_mean': float(np.mean(mae_list)), 'RMSE_mean': float(np.mean(rmse_list)), 'R2_mean': float(np.mean(r2_list)),
            'MAE_std': float(np.std(mae_list)), 'RMSE_std': float(np.std(rmse_list)), 'R2_std': float(np.std(r2_list))
        }
        if args.log_target == 'log1p':
            res.update({
                'MAE_raw_mean': float(np.mean(mae_raw_list)), 'RMSE_raw_mean': float(np.mean(rmse_raw_list)), 'R2_raw_mean': float(np.mean(r2_raw_list)),
                'MAE_raw_std': float(np.std(mae_raw_list)), 'RMSE_raw_std': float(np.std(rmse_raw_list)), 'R2_raw_std': float(np.std(r2_raw_list))
            })
        rows.append(res)

    return pd.DataFrame(rows).sort_values('RMSE_mean')

# -----------------
# Main
# -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--run', type=str, choices=['time','cv'], default='time')
    parser.add_argument('--target', type=str, choices=['AgrInc','CropInc'], default='AgrInc')
    parser.add_argument('--cv', type=int, default=10)
    parser.add_argument('--train_year', type=int, default=2014)
    parser.add_argument('--test_year', type=int, default=2016)
    parser.add_argument('--log_target', type=str, choices=['raw','log1p'], default='log1p')
    parser.add_argument('--exclude_related_income', action='store_true',
                        help='Loại các biến có thể làm rò rỉ mục tiêu (TotalInc, Wage, CropInc/CropValue ...)')
    parser.add_argument('--winsor_lower', type=float, default=None)
    parser.add_argument('--winsor_upper', type=float, default=None)
    parser.add_argument('--scale_numeric', action='store_true')
    parser.add_argument('--onehot_geo', type=str, default='province', choices=['none','province','district','commune','all'])
    parser.add_argument('--year_as_category', action='store_true')
    parser.add_argument('--drop_first', action='store_true')
    parser.add_argument('--results_dir', type=str, default='./results_baseline')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Load data
    df2010 = load_year(args.data_dir, 2010)
    df2014 = load_year(args.data_dir, 2014)
    df2016 = load_year(args.data_dir, 2016)

    if args.run == 'time':
        train_map = {2010: df2010, 2014: df2014, 2016: df2016}
        if args.train_year not in train_map or args.test_year not in train_map:
            raise SystemExit("train_year/test_year không hợp lệ (phải là 2010/2014/2016)")
        res = run_time_split(train_map[args.train_year], train_map[args.test_year], args.target, args)
        out_csv = os.path.join(args.results_dir, f"baseline_{args.target}_time_{args.train_year}_{args.test_year}.csv")
        res.to_csv(out_csv, index=False)
        print("=== Time-split Results ===")
        print(res.to_string(index=False))
        print(f"[DONE] Saved: {out_csv}")
    else:
        # Merge all
        df_all = pd.concat([df2010, df2014, df2016], ignore_index=True, sort=False)
        res = run_cv(df_all, args.target, args)
        out_csv = os.path.join(args.results_dir, f"baseline_{args.target}_cv_{args.cv}fold.csv")
        res.to_csv(out_csv, index=False)
        print("=== CV Results ===")
        print(res.to_string(index=False))
        print(f"[DONE] Saved: {out_csv}")

if __name__ == '__main__':
    main()
