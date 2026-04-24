from __future__ import annotations
from itertools import product
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Literal, Any, Callable
import warnings
import sys
import numpy as np
import pandas as pd
from .plotting import plot_hd_bars_signed, plot_irf_single, \
    plot_irf_grid, plot_cf_policy_space_single, plot_irf_single_var_grid

try:
    from scipy.linalg import cholesky, qr
    from scipy.optimize import minimize, differential_evolution
except Exception:  # allow importing without scipy; raise when used
    cholesky = None
    qr = None
    minimize = None
    differential_evolution = None

TimeAxis = Literal["columns", "index"]          # где время в DataFrame
Layout = Literal["KL_KxT", "TxK"]               # KL: K x T (vars x time), TxK: time x vars
TimeOrder = Literal["KL_reverse", "chronological"]  # KL_reverse: col0 newest, colLast oldest

def _is_datetime_like_index(idx: pd.Index) -> bool:
    return isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex)) or pd.api.types.is_datetime64_any_dtype(idx)

def _as_2d_float(a: Any) -> np.ndarray:
    x = np.asarray(a)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    return x.astype(float, copy=False)

def _matrix_rank(A: np.ndarray, tol: Optional[float] = None) -> int:
    # Robust rank via SVD
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    if tol is None:
        tol = np.max(A.shape) * np.finfo(float).eps * (s[0] if s.size else 1.0)
    return int(np.sum(s > tol))

def _random_orthogonal(K: int, rng: np.random.Generator) -> np.ndarray:
    if qr is None:
        raise ImportError("scipy is required for QR-based random orthogonal draws.")
    A = rng.standard_normal((K, K))
    Q, R = qr(A) # type: ignore
    # Fix sign ambiguity for reproducibility-ish
    d = np.sign(np.diag(R))
    d[d == 0] = 1.0
    return Q * d # type: ignore

def _build_companion(A_endo_no_const: np.ndarray, K: int, p: int) -> np.ndarray:
    """
    A_endo_no_const: (K, Kp) reduced-form coefficients excluding constant, ordered [A1|A2|...|Ap]
    companion size: (Kp, Kp)
    """
    if A_endo_no_const.shape != (K, K * p):
        raise ValueError(f"A_endo_no_const must be (K, Kp) = ({K},{K*p}), got {A_endo_no_const.shape}")
    A_comp = np.zeros((K * p, K * p))
    A_comp[:K, :K * p] = A_endo_no_const
    if p > 1:
        A_comp[K:, :-K] = np.eye(K * (p - 1))
    return A_comp

def _irf_companion(A_endo_no_const: np.ndarray, B0inv: np.ndarray, horizon: int) -> np.ndarray:
    """
    Returns irfs: (horizon+1, K, K) where irfs[h,:,j] = response at h to shock j.
    """
    K = B0inv.shape[0]
    if B0inv.shape != (K, K):
        raise ValueError("B0inv must be square (K,K).")
    p = A_endo_no_const.shape[1] // K
    A_comp = _build_companion(A_endo_no_const, K=K, p=p)
    J = np.zeros((K * p, K))
    J[:K, :K] = np.eye(K)

    irfs = np.zeros((horizon + 1, K, K))
    A_pow = np.eye(K * p)
    for h in range(horizon + 1):
        irfs[h] = (J.T @ A_pow @ J) @ B0inv
        A_pow = A_pow @ A_comp
    return irfs

def _long_run_matrix(A_endo_no_const: np.ndarray, B0inv: np.ndarray) -> np.ndarray:
    """
    Long-run multiplier: C(∞) = (I - A1 - ... - Ap)^(-1) B0inv
    """
    K = B0inv.shape[0]
    p = A_endo_no_const.shape[1] // K
    A_sum = np.zeros((K, K))
    for i in range(p):
        A_sum += A_endo_no_const[:, i * K:(i + 1) * K]
    M = np.eye(K) - A_sum
    # Check for singularity
    if np.linalg.cond(M) > 1e12:
        warnings.warn("Matrix (I - A_sum) is near singular. Long-run effects may be unstable.")
    return np.linalg.solve(M, B0inv)

def _givens_Q_from_params(params: np.ndarray, K: int) -> np.ndarray:
    """
    Build orthogonal Q via sequence of Givens rotations. params length = K*(K-1)/2.
    """
    Q = np.eye(K)
    idx = 0
    for i in range(K - 1):
        for j in range(i + 1, K):
            theta = params[idx]
            idx += 1
            G = np.eye(K)
            c, s = np.cos(theta), np.sin(theta)
            G[i, i] = c
            G[j, j] = c
            G[i, j] = -s
            G[j, i] = s
            Q = Q @ G
    return Q

def _initial_and_exog_hd(X,
                         A_hat,
                         p,
                         K,
                         T,
                         const: bool = True,
                         exog_data = None):
    """
    Функция строит накопленные шоки от начальный значений, константы и
    экзогенных переменных
    X -- KxT матрица эндогенных переменных
    A_hat -- KxKp матрица коэфициентов из приведенной формы
    p -- лаги эндогенных
    K -- кол-во эндогенных переменных
    T -- длинна временного ряда
    const -- булевое значение на наличие константы
    exog_data -- K-exog x T матрица экзогенных переменных
    скрипт во многом следует Cesa-Bianchi кроме части с
    влиянием экзогенных переменных.
    """
    data_init = X[:, :p].T.flatten()
    nobs = T - p

    # const dependencies for the A_exog and for the const vector
    if const:
        A_endo_no_const = A_hat[:, 1:K*p + 1].copy()
        vec_const = np.vstack((A_hat[:, 0].reshape(K, 1), np.zeros((K*(p-1), 1))))
        n_exog = A_hat.shape[1]-K*p-1
    else:
        A_endo_no_const = A_hat[:, :K*p].copy()
        n_exog = A_hat.shape[1] -K*p

    if exog_data is not None:
        A_exog = np.zeros((K*p, n_exog))
        A_exog[:K, :] = A_hat[:, -n_exog:].copy()
    
    # check whether the dimensions are correct
    if A_endo_no_const.shape != (K, K * p):
        raise ValueError(f"A_endo_no_const must be (K, K*p) = "+
                     f"({K},{K*p}), got {A_endo_no_const.shape}")

    A_comp = np.zeros((K * p, K * p))
    A_comp[:K, :K * p] = A_endo_no_const
    if p > 1:
        A_comp[K:, :-K] = np.eye(K * (p - 1))
    
    J_select = np.zeros((K, K*p))
    J_select[:, :K] = np.eye(K)  # ✅ Верно: берем только верхний блок

    # const
    if const:
        HD_const = np.zeros((K*p, nobs+1))
        for t in range(1, nobs + 1):
            HD_const[:, t] = np.squeeze(vec_const +
                                   (A_comp @ HD_const[:, t-1]).reshape(K*p, 1))
    else:
        HD_const = np.zeros((K*p, nobs+1))
        
    # init
    HD_init = np.zeros((K*p, nobs+1))
    HD_init[:, 0] = data_init
    for t in range(1, nobs + 1):
        HD_init[:, t] = A_comp @ HD_init[:, t-1]

    # exog
    if exog_data is not None:
        HD_exog = np.zeros((K*p, nobs+1))
        for t in range(1, nobs + 1):
            HD_exog[:, t] = A_exog @ exog_data[:, t] + A_comp @ HD_exog[:, t-1]
    else:
        HD_exog = np.zeros((K*p, nobs+1))

    HD_init = (J_select @ HD_init)[:, 1:]
    HD_const = (J_select @ HD_const)[:, 1:]
    HD_exog = (J_select @ HD_exog)[:, 1:] 
    HD_nonendo = HD_init + HD_const + HD_exog

    return HD_init, HD_exog, HD_const, HD_nonendo  

def _pacf_to_ar(pacf: np.ndarray) -> np.ndarray:
    """
    Преобразует частичные автокорреляции (PACF) в коэффициенты AR
    через рекурсию Дурбина-Левинсона.
    """
    p = len(pacf)
    ar_coef = np.zeros(p)
    for k in range(p):
        phi_kk = pacf[k]
        if k == 0:
            ar_coef[k] = phi_kk
        else:
            ar_coef[:k] = ar_coef[:k] - phi_kk * ar_coef[:k][::-1]
            ar_coef[k] = phi_kk
    return ar_coef

def _generate_fixed_innovations(T: int, burn_in: int = 100, seed: int = 42) -> np.ndarray:
    """Генерирует фиксированные инновации для оптимизации"""
    rng = np.random.default_rng(seed)
    eta = rng.normal(0, 1, T + burn_in)
    return eta[burn_in:]

def _generate_ar_series_fixed(rho: np.ndarray, eta_fixed: np.ndarray) -> np.ndarray:
    """Генерирует шок по AR процессу с фиксированными инновациями"""
    T = len(eta_fixed)
    p = len(rho)
    shock = np.zeros(T)
    for t in range(p, T):
        shock[t] = np.sum(rho * shock[t-p:t][::-1]) + eta_fixed[t]
    return shock

def _orthogonalize_and_normalize_shock(candidate_shock: np.ndarray,
                                       other_shocks: np.ndarray) -> np.ndarray:
    """
    Ортогонализует, центрирует и нормирует шок.
    Гарантирует: (1) не коррелирует с other_shocks, (2) mean=0, (3) std=1.
    """
    # Ортогонализация
    proj_coef, _, _, _ = np.linalg.lstsq(other_shocks, candidate_shock, rcond=None)
    projection = other_shocks @ proj_coef
    orthogonal = candidate_shock - projection
    # Центрирование
    orthogonal = orthogonal - np.mean(orthogonal)

    # Нормировка
    if np.std(orthogonal) < 1e-6:
        warnings.warn("Shock std is near zero after orthogonalization")
        return orthogonal
    normalized = orthogonal / np.std(orthogonal)

    return normalized

def get_independent_seeds(master_seed, row_index):
    """
    Создает уникальный положительный seed для конкретного ряда.
    """
    return abs(master_seed * 1000 + row_index)

def orthogonalize_shock_block(candidate_shocks_block, U_other):
    """
    Ортогонализирует блок шоков политики относительно U_other и друг друга.
    candidate_shocks_block: (n_policy, T) - сырые кандидаты
    U_other: (K-n_policy, T) - остальные шоки
    """
    n_policy, T = candidate_shocks_block.shape
    # 1. Сначала ортогонализуем весь блок к U_other
    cleaned_block = []
    for i in range(n_policy):
        raw = candidate_shocks_block[i, :]
        ortho = _orthogonalize_and_normalize_shock(raw, U_other)
        cleaned_block.append(ortho)

    cleaned_block = np.array(cleaned_block)  # (n_policy, T)

    # 2. Теперь обеспечиваем ортогональность внутри блока (QR-разложение)
    # T должно быть >= n_policy, что обычно верно
    Q, R = np.linalg.qr(cleaned_block.T)  # Q: (T, n_policy)

    # Q содержит ортогональные векторы-столбцы. 
    # Нам нужно вернуть их как строки (n_policy, T)
    # Важно: QR может изменить знак, но сохранит ортогональность
    orthogonal_block = Q.T 

    # 3. Нормируем каждый шок к единичной дисперсии (QR нормирует к норме 1, но не к std=1)
    for i in range(n_policy):
        orthogonal_block[i, :] /= np.std(orthogonal_block[i, :])
        
    return orthogonal_block

@dataclass
class RestrictionResult:
    Q: np.ndarray
    B0inv: np.ndarray
    details: Dict[str, Any]

@dataclass
class CounterfactualResult:
    """
    Результаты контрфактического анализа политики.
    """
    original_model: SVAR_KL
    counterfactual_model: SVAR_KL
    optimal_pacf: np.ndarray
    optimal_seed: int
    loss_value: float
    loss_history: list = field(default_factory=list)
    shock_original: Optional[np.ndarray] = None
    shock_counterfactual: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class SVAR_KL:
    """
    SVAR/VAR in Killian–Lütkepohl matrix layout.
    Expected main input layout (KL):
      - variables along rows (K)
      - time along columns (T)
      - columns ordered "reverse": col0 newest (t), colLast oldest (0)
    """

    # --------- init / data ----------
    def __init__(
        self,
        data: pd.DataFrame,
        p: int,
        exog: Optional[pd.DataFrame] = None,
        layout: Optional[Layout] = None,
        u_dict: list = None,
        y_dict: list = None,
        time_order: TimeOrder = "KL_reverse",
        add_const: bool = True,
        check_binary_collinearity: bool = True,
        log_vars: List[str] = None,
        base_levels: Optional[Dict[str, float]] = None,
        name: str = "SVAR_KL",
        verbose: bool = True
    ):
        self.name = name
        self.verbose = verbose
        # 1. Валидация p
        if not isinstance(p, int) or p < 1:
            raise ValueError("p must be a positive integer >= 1")
        self.p = p

        self.add_const = bool(add_const)
        
        # 2. Валидация time_order
        if time_order not in ["chronological", "KL_reverse"]:
            raise ValueError(f"Unknown time_order: {time_order}. Must be 'chronological' or 'KL_reverse'")
        self.time_order = time_order

        # 3. Валидация данных
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("data DataFrame is empty")
        
        self._raw_data = data.copy()
        self._raw_exog = exog.copy() if exog is not None else None

        # Infer / validate orientation
        self.layout = self._infer_layout(data, layout=layout)
        self.X, self.var_names, self.time_index = self._coerce_to_KT(data, layout=self.layout)
        self.K, self.T = self.X.shape
        self.nobs = self.T - self.p
        
        # 4. Валидация размерностей
        if self.K < 1:
            raise ValueError("No variables found in data")
        if self.T <= self.p:
            raise ValueError(f"Need T > p. Got T={self.T}, p={self.p}. Increase data length or decrease p.")

        # 5. Валидация словарей имен
        if u_dict is None:
            self.u_dict = {k: k for k in self.var_names}
        else:
            if len(u_dict) != self.K:
                raise ValueError(f"u_dict length ({len(u_dict)}) must match number of variables ({self.K})")
            self.u_dict = {k: v for k, v in zip(self.var_names, u_dict)}

        if y_dict is None:
            self.y_dict = {k: k for k in self.var_names}
        else:
            if len(y_dict) != self.K:
                raise ValueError(f"y_dict length ({len(y_dict)}) must match number of variables ({self.K})")
            self.y_dict = {k: v for k, v in zip(self.var_names, y_dict)}

        # 6. Обработка экзогенных переменных
        self.exog = None
        self.exog_names = None
        if exog is not None:
            if not isinstance(exog, pd.DataFrame):
                raise TypeError("exog must be a pandas DataFrame")
            ex_layout = self._infer_layout(exog, layout=layout)  # usually same convention
            E, ex_names, ex_time = self._coerce_to_KT(exog, layout=ex_layout)
            if E.shape[1] != self.T:
                raise ValueError("Exog must have same T (columns/time) as endog in KL layout.")
            self.exog = E
            self.exog_names = ex_names

        # 7. Валидация значений (NaN/Inf)
        self._validate_values(self.X, label="endog")
        if self.exog is not None:
            self._validate_values(self.exog, label="exog")

        # 8. Валидация log_vars и подготовка вектора масштабирования
        if log_vars is not None:
            if not isinstance(log_vars, list):
                raise TypeError("log_vars must be a list of variable names")
            for var in log_vars:
                if var not in self.var_names:
                    raise ValueError(f"Variable '{var}' in log_vars not found in data columns. Available: {self.var_names}")
            self.log_vars = log_vars
        else:
            self.log_vars = []

        # 9. Инициализация вектора масштабирования для процентов
        self._base_levels = base_levels
        self._update_scale_factors(base_levels)

        if check_binary_collinearity:
            self._check_binary_multicollinearity(self._raw_data)

        # --------- OLS outputs (filled by fit_ols) ----------
        self.Y: Optional[np.ndarray] = None         # (K, T-p)
        self.Z: Optional[np.ndarray] = None         # (nreg, T-p)
        self.A_hat: Optional[np.ndarray] = None     # (K, nreg)
        self.E: Optional[np.ndarray] = None         # (K, T-p)
        self.Sigma_u: Optional[np.ndarray] = None   # (K, K)
        self.P: Optional[np.ndarray] = None         # (K, K) Cholesky of Sigma_u (lower)

        # convenience slices
        self.A_hat_endo: Optional[np.ndarray] = None      # (K, 1+Kp) if const else (K, Kp)
        self.A_endo_no_const: Optional[np.ndarray] = None # (K, Kp)

        # --------- identification outputs ----------
        self.Q: Optional[np.ndarray] = None         #  (K, K)
        self.B0inv: Optional[np.ndarray] = None     # (K, K)
        self.Upsilon: Optional[np.ndarray] = None   # (K,K) long-run matrix (Λ/Υ in your notation)


    def _update_scale_factors(self, base_levels: Optional[Dict[str, float]] = None):
        """
        Обновляет вектор масштабирования для конвертации в проценты.
        Вызывается в __init__ или вручную при изменении base_levels.
        """
        scale = np.ones(self.K)
        log_mask = np.array([v in self.log_vars for v in self.var_names], dtype=bool)
        
        for i in range(self.K):
            var_name = self.var_names[i]
            if log_mask[i]:
                # Для логов: умножение на 100
                scale[i] = 100.0
            else:
                # Для уровней: (1 / base) * 100
                if base_levels is not None and var_name in base_levels:
                    base = base_levels[var_name]

                else:
                    base = 1.0
                    if base_levels is None:
                        if self.verbose:
                            print('Уровни не переданны, использовал единичный вектор')
                    else:
                        if self.verbose:
                            print(f'Переменную из словаря базового уровня {var_name} не нашел в списке переменных. Что-то не то, проверь ввод')
                
                if abs(base) < 1e-12:
                    warnings.warn(f"Base level for '{var_name}' is near zero. Using 1.0 to avoid division by zero.")
                    base = 1.0
                
                scale[i] = 100.0 / base
        
        self._scale_factors_irf = scale[np.newaxis, :, np.newaxis]
        self._scale_factors_upsilon = scale

    def set_base_levels(self, base_levels: Dict[str, float]):
        """
        Позволяет обновить базовые уровни после инициализации.
        """
        self._update_scale_factors(base_levels)


    @property
    def Cinf(self) -> Optional[np.ndarray]:
        return getattr(self, "Upsilon", None)

    @Cinf.setter
    def Cinf(self, value: Optional[np.ndarray]) -> None:
        self.Upsilon = value

    # ----------------- data helpers -----------------
    def _infer_layout(self, df: pd.DataFrame, layout: Optional[Layout]) -> Layout:
        if layout is not None:
            return layout

        # Heuristic:
        # - if columns are datetime-like => time in columns => KL_KxT
        # - if index is datetime-like => time in index => TxK
        col_time = _is_datetime_like_index(df.columns)
        idx_time = _is_datetime_like_index(df.index)

        if col_time and not idx_time:
            return "KL_KxT"
        if idx_time and not col_time:
            return "TxK"

        # ambiguous: default to KL_KxT (your convention), but be explicit
        return "KL_KxT"

    def _coerce_to_KT(self, df: pd.DataFrame, layout: Layout) -> Tuple[np.ndarray, List[str], pd.Index]:
        if layout == "KL_KxT":
            X = df.to_numpy(dtype=float)
            # Use index as var names if they are meaningful, else generate
            var_names = [str(i) for i in df.index] if df.index is not None and len(df.index) == X.shape[0] else [f"y{i}" for i in range(X.shape[0])]
            # If index is not unique or not matching rows, fallback to column names if TxK was intended? 
            # But layout is KL_KxT, so rows are vars.
            if len(var_names) != X.shape[0]:
                 var_names = [f"var_{i}" for i in range(X.shape[0])]
            
            time_index = df.columns
            return X, var_names, time_index

        if layout == "TxK":
            # df: time x vars -> transpose into KL
            X = df.to_numpy(dtype=float).T
            var_names = [str(c) for c in df.columns]
            time_index = df.index
            return X, var_names, time_index

        raise ValueError(f"Unknown layout: {layout}")

    def _validate_values(self, X: np.ndarray, label: str):
        if not np.isfinite(X).all():
            bad = np.argwhere(~np.isfinite(X))
            raise ValueError(f"{label} contains NaN/inf at positions like {bad[:5].tolist()} (showing up to 5).")

    def _check_binary_multicollinearity(self, df: pd.DataFrame, tol: float = 1e-12):
        """
        Checks only among binary variables (0/1) in the provided DataFrame.
        In KL_KxT layout, variables are rows; in TxK, variables are columns.
        """
        if self.layout == "KL_KxT":
            # variables in index/rows -> inspect rows
            X = df.to_numpy()
            names = [str(i) for i in df.index] if len(df.index) == X.shape[0] else [f"y{i}" for i in range(X.shape[0])]
            if len(names) != X.shape[0]:
                 names = [f"var_{i}" for i in range(X.shape[0])]

            # detect binary rows
            binary_rows = []
            for k in range(X.shape[0]):
                vals = np.unique(X[k, :])
                vals = vals[~pd.isna(vals)]
                if vals.size > 0 and set(vals.tolist()).issubset({0, 1, 0.0, 1.0}):
                    binary_rows.append(k)
            if len(binary_rows) <= 1:
                return
            B = X[binary_rows, :].astype(float)
            # remove any all-constant rows (still "binary") separately
            nonconst = np.var(B, axis=1) > tol
            B = B[nonconst, :]
            if B.shape[0] <= 1:
                return
            r = _matrix_rank(B)
            if r < B.shape[0]:
                bad_names = [names[binary_rows[i]] for i in range(len(binary_rows)) if (i < len(nonconst) and nonconst[i])]
                raise ValueError(
                    "Binary variables are perfectly multicollinear (rank deficient). "
                    f"Binary set rank={r}, count={B.shape[0]}. Vars: {bad_names}"
                )
        else:
            # TxK: variables in columns
            X = df.to_numpy()
            names = [str(c) for c in df.columns]
            binary_cols = []
            for j in range(X.shape[1]):
                vals = np.unique(X[:, j])
                vals = vals[~pd.isna(vals)]
                if vals.size > 0 and set(vals.tolist()).issubset({0, 1, 0.0, 1.0}):
                    binary_cols.append(j)
            if len(binary_cols) <= 1:
                return
            B = X[:, binary_cols].astype(float)
            nonconst = np.var(B, axis=0) > tol
            B = B[:, nonconst]
            if B.shape[1] <= 1:
                return
            r = _matrix_rank(B)
            if r < B.shape[1]:
                bad_names = [names[binary_cols[i]] for i in range(len(binary_cols)) if (i < len(nonconst) and nonconst[i])]
                raise ValueError(
                    "Binary variables are perfectly multicollinear (rank deficient). "
                    f"Binary set rank={r}, count={B.shape[1]}. Vars: {bad_names}"
                )

    # ----------------- VAR: build Y,Z in KL order -----------------
    def _build_YZ(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build Y and Z matrices for VAR(p) estimation.
        
        Chronological mode (col 0 = oldest, col -1 = newest):
        - Y uses the LAST nobs columns (most recent observations).
        - Lags are shifted to the LEFT (older data).
        """
        X = self.X
        K, T = self.X.shape
        p = self.p
        nobs = T - p

        if self.time_order == "chronological":
            # --- ХРОНОЛОГИЧЕСКИЙ ПОРЯДОК (0=oldest, -1=newest) ---
            
            # Y: берем последние nobs наблюдений (индексы от p до T-1)
            # Пример: T=10, p=3 -> Y = X[:, 3:10] (7 наблюдений)
            Y = X[:, p:]  # (K, T-p)

            Z_blocks = []
            if self.add_const:
                Z_blocks.append(np.ones((1, nobs)))

            # Лаги:  сдвигаем окно ВЛЕВО на i позиций
            # Если Y это [p : T], то лаг i это [p-i : T-i]
            # Пример для i=1: X[:, 2:9] (когда Y_t на индексе 3, лаг на индексе 2)
            for i in range(1, p + 1):
                Z_blocks.append(X[:, p - i : T - i])  # (K, T-p)

            Z = np.vstack(Z_blocks)

            # Exog: выравниваем по тем же индексам, что и Y
            if self.exog is not None:
                E = self.exog[:, -nobs:]
                Z = np.vstack([Z, E])
        else:
            # --- KL_REVERSE ПОРЯДОК (0=newest, -1=oldest) ---
            # (Оставляем старую логику для совместимости, если нужно)
            
            # Y: берем первые nobs наблюдений (они же самые новые в этом порядке)
            Y = X[:, :nobs]  # (K, T-p)

            Z_blocks = []
            if self.add_const:
                Z_blocks.append(np.ones((1, nobs)))

            # Лаги: сдвигаем окно ВПРАВО на i позиций (в сторону старых данных)
            for i in range(1, p + 1):
                Z_blocks.append(X[:, i : i + nobs])  # (K, T-p)

            Z = np.vstack(Z_blocks)

            if self.exog is not None:
                E = self.exog[:, :nobs]
                Z = np.vstack([Z, E])

        return Y, Z

    # ----------------- 2) OLS estimation -----------------
    def fit_ols(self) -> "SVAR_KL":
        """
        Estimates: Y = A_hat Z + E  (KL matrix form)
        Stores attributes similar to your OLS_estimation output.
        """
        Y, Z = self._build_YZ()
        
        # Check for multicollinearity in Z
        if np.linalg.cond(Z @ Z.T) > 1e12:
            warnings.warn("Z'Z is near singular. OLS estimates may be unstable.")

        # A_hat = Y Z' (Z Z')^{-1}
        ZZt = Z @ Z.T
        YZt = Y @ Z.T
        A_hat = YZt @ np.linalg.inv(ZZt)
        E = Y - A_hat @ Z

        # Sigma_u (unbiased-ish): E E' / (T-p - nreg) is common; KL often uses / (T-p)
        # We'll store both pieces later if you want. For now: / (T-p).
        nobs = Y.shape[1]
        Sigma_u = (E @ E.T) / nobs
 
        if cholesky is None:
            raise ImportError("scipy is required for Cholesky decomposition.")
        
        # Check if Sigma_u is positive definite
        if not np.all(np.linalg.eigvals(Sigma_u) > 0):
            raise ValueError("Sigma_u is not positive definite. Check data for collinearity.")

        P = cholesky(Sigma_u, lower=True)

        self.Y, self.Z, self.A_hat, self.E = Y, Z, A_hat, E
        self.Sigma_u, self.P = Sigma_u, P

        # convenience: endog-only coefficient block
        # Z contains [const?; lags; exog?]. We split the endog part = const + K*p.
        endo_reg_count = (1 if self.add_const else 0) + self.K * self.p
        self.A_hat_endo = A_hat[:, :endo_reg_count]
        # A_endo_no_const: remove constant if present
        self.A_endo_no_const = self.A_hat_endo[:, 1:] if self.add_const else self.A_hat_endo.copy()

        return self

    # ----------------- 3) Orthogonal decomposition (baseline) -----------------
    def identify_orthogonal(self, Q: Optional[np.ndarray] = None, seed: Optional[int] = None) -> RestrictionResult:
        """
        Baseline orthogonal identification: B0inv = P @ Q
        If Q None -> identity, i.e., Cholesky ordering as in data.
        """
        if self.P is None or self.A_endo_no_const is None:
            raise RuntimeError("Call fit_ols() first.")

        K = self.K
        if Q is None:
            Q = np.eye(K)
        Q = _as_2d_float(Q)
        if Q.shape != (K, K):
            raise ValueError("Q must be (K,K).")

        B0inv = self.P @ Q
        Upsilon = _long_run_matrix(self.A_endo_no_const, B0inv)

        self.Q, self.B0inv, self.Upsilon = Q, B0inv, Upsilon
        return RestrictionResult(Q=Q, B0inv=B0inv, details={"type": "orthogonal", "seed": seed})

    # ----------------- 4) Sign restrictions (accept-reject on IRFs) -----------------
    def identify_sign_restrictions(
        self,
        restrictions: Sequence[Tuple[int, int, int, int]],
        horizon: int,
        n_draws: int = 5000,
        seed: Optional[int] = None,
        require_all: bool = True,
    ) -> RestrictionResult:
        """
        restrictions: list of (row, col, h, sign)
          row: response variable index
          col: shock index
          h: horizon (0..horizon)
          sign: +1 or -1

        Draw random orthogonal Q, compute IRFs, accept if all sign restrictions hold.
        """
        if self.P is None or self.A_endo_no_const is None:
            raise RuntimeError("Call fit_ols() first.")

        rng = np.random.default_rng(seed)
        K = self.K

        def ok(irfs: np.ndarray) -> bool:
            checks = []
            for (r, c, h, sgn) in restrictions:
                val = irfs[h, r, c]
                checks.append((sgn * val) > 0)
            return all(checks) if require_all else any(checks)

        for d in range(n_draws):
            Q = _random_orthogonal(K, rng)
            B0inv = self.P @ Q
            irfs = _irf_companion(self.A_endo_no_const, B0inv, horizon=horizon)
            if ok(irfs):
                Upsilon = _long_run_matrix(self.A_endo_no_const, B0inv)
                self.Q, self.B0inv, self.Upsilon = Q, B0inv, Upsilon
                return RestrictionResult(Q=Q, B0inv=B0inv, details={"type": "sign", "draw": d + 1, "seed": seed})

        raise RuntimeError(f"No acceptable Q found in {n_draws} draws for given sign restrictions.")


    # ----------------- 6) Combined identification via minimization -----------------
    def identify_combined(
        self,
        short_run_anchor: Sequence[Tuple[float, int, int, int, float]] = [],
        long_run_anchor: Sequence[Tuple[float, int, int, float]] = [],
        short_run_signs: Sequence[Tuple[float, int, int, bool, int, float]] = [],
        long_run_signs: Sequence[Tuple[float, int, int, bool, float]] = [],
        horizon_for_short_sign: Optional[int] = None,  # used only for sanity check
        n_starts: int = 50,
        seed: Optional[int] = None,
        method: str = "BFGS",
    ) -> RestrictionResult:
        """
        Minimizes a penalty to satisfy:
          - short_run_anchor on B0inv (impact)
          - long_run_anchor on Cinf
          - short_sign_signs IRF(h) narrative assymetric restrictions
          - long_sign_signs on Cinf narrative assymetric restrictions

        short_run_anchor: (target, row, col, horizon, weight)
        long_run_anchor: (target, row, col, weight)
        short_run_signs: (target, row, col, sign(True if the irf should be greater that the target, False otherwise), horizon, weight)
        long_run_signs: (target, row, col, sign(True if the irf should be greater that the target, False otherwise), weight)
        """
        if minimize is None:
            raise ImportError("scipy is required for optimization-based identification.")
        if self.P is None or self.A_endo_no_const is None:
            raise RuntimeError("Call fit_ols() first.")

        rng = np.random.default_rng(seed)
        K = self.K
        n_params = K * (K - 1) // 2

        # Small validation (optional)
        if horizon_for_short_sign is not None:
            for (_, _, _, _, h, _) in short_run_signs:
                if h > horizon_for_short_sign:
                    raise ValueError("A short-run sign restriction has h > horizon_for_short_sign.")

        def penalty(params: np.ndarray) -> float:
            Q = _givens_Q_from_params(params, K)
            B0inv = self.P @ Q
            
            # Compute Upsilon
            Upsilon = _long_run_matrix(self.A_endo_no_const, B0inv)
            
            # Convert to percent for penalty calculation (FAST)
            Upsilon_pct = Upsilon * self._scale_factors_upsilon

            irfs_pct = None
            if short_run_anchor or short_run_signs:
                combined = list(short_run_anchor) + list(short_run_signs)
                max_h = max(h[-2] for h in combined) 
                irfs = _irf_companion(self.A_endo_no_const, B0inv, horizon=max_h)
                irfs_pct = irfs * self._scale_factors_irf

            loss = 0.0

            def safe_exp_pen(val: float, target: float, sgn: bool, weight: float) -> float:
                diff = val - target
                # sgn=True  => штрафуем, когда val < target => exp(-diff)
                # sgn=False => штрафуем, когда val > target => exp(diff)
                exp_arg = -diff if sgn else diff
                # Clip защищает от overflow: exp(50) ≈ 5e21, exp(710) → inf
                exp_arg = float(np.clip(exp_arg, -50.0, 50.0))
                return weight * float(np.exp(exp_arg) - 1.0)

            # short-run target (quadratic)
            if short_run_anchor:
                for (target, row, col, h, whght) in short_run_anchor:
                    v = irfs_pct[h, row, col]
                    loss += whght * float((target - v) ** 2)

            # long-run target (quadratic)
            if long_run_anchor:
                for (target, row, col, whght) in long_run_anchor:
                    v = Upsilon_pct[row, col]
                    loss += whght * float((target - v) ** 2)

            # short-run narrative restrictions
            if short_run_signs:        
                for (target, row, col, sgn, h, whght) in short_run_signs:
                    v = irfs_pct[h, row, col]
                    loss += safe_exp_pen(v, target, sgn, whght)

            # long-run sign restrictions
            if long_run_signs:
                for (target, row, col, sgn, whght) in long_run_signs:
                    v = Upsilon_pct[row, col]
                    loss += safe_exp_pen(v, target, sgn, whght)

            return loss

        best = {"fun": np.inf, "x": None, "res": None}
        drops = {}
        for s in range(n_starts):
            x0 = rng.uniform(0.0, 2 * np.pi, size=n_params)
            res = minimize(penalty, x0=x0, method=method)
            if res.fun < best["fun"]:
                best = {"fun": float(res.fun), "x": res.x.copy(), "res": res}
            else:
                drops[s] = {"fun": float(res.fun), "x": res.x.copy(), "res": res, 
                            "Q": _givens_Q_from_params(res.x.copy(), K)}

        if best["x"] is None:
            raise RuntimeError("Optimization failed for all starts.")

        Q = _givens_Q_from_params(best["x"], K)
        B0inv = self.P @ Q
        Upsilon = _long_run_matrix(self.A_endo_no_const, B0inv)

        self.Q, self.B0inv, self.Upsilon = Q, B0inv, Upsilon
        return RestrictionResult(
            Q=Q,
            B0inv=B0inv,
            details={"type": "combined",
                     "best_fun": best["fun"],
                     "method": method, 
                     "n_starts": n_starts, 
                     "seed": seed, 
                    'drops': drops},
        )

    # ----------------- 7) IRF + bootstrap variants -----------------
    def irf(self, horizon: int, B0inv: Optional[np.ndarray] = None) -> np.ndarray:
        if self.A_endo_no_const is None:
            raise RuntimeError("Call fit_ols() first.")
        if B0inv is None:
            if self.B0inv is None:
                raise RuntimeError("No identification stored. Call identify_* first or pass B0inv.")
            B0inv = self.B0inv
        return _irf_companion(self.A_endo_no_const, B0inv, horizon=horizon)


    def irf_bootstrap(
        self,
        horizon: int,
        n_boot: int = 500,
        seed: Optional[int] = None,
        scheme: Literal[
            "fixed_Q",            # keep Q from original identification, re-estimate VAR each draw, then B0inv=P_b @ Q_fixed
            "redo_cholesky",      # each draw uses its own Cholesky (Q=I)
            "redo_sign",          # each draw re-runs sign restrictions (slow)
            "redo_combined",      # each draw re-runs combined minimization (slow)
        ] = "fixed_Q",
        sign_restrictions: Sequence[Tuple[int, int, int, int]] = (),
        combined_kwargs: Optional[dict] = None,
        n_draws_sign: int = 5000,
    ) -> np.ndarray:
        """
        Returns IRF draws: (n_boot, horizon+1, K, K)
        Residual bootstrap (basic): resample reduced-form residuals columns with replacement,
        simulate series using fitted VAR, refit OLS.
        """
        if self.A_hat_endo is None or self.E is None or self.A_endo_no_const is None:
            raise RuntimeError("Call fit_ols() first.")
        if scheme == "fixed_Q" and self.Q is None:
            raise RuntimeError("scheme='fixed_Q' requires stored Q from identify_*. ")

        rng = np.random.default_rng(seed)
        K, T = self.K, self.T
        p = self.p
        nobs = T - p

        # Extract reduced-form A's and const
        if self.add_const:
            c = self.A_hat_endo[:, [0]]  # (K,1)
            A_no_const = self.A_endo_no_const  # (K,Kp)
        else:
            c = None
            A_no_const = self.A_endo_no_const

        # residuals aligned with  Y (K, nobs)
        U = self.E

        def simulate_from_bootstrap() -> np.ndarray:
            # Resample residual columns (u_t) with replacement over nobs
            idx = rng.integers(0, nobs, size=nobs)
            U_b = U[:, idx]  # (K,nobs)

            # Build X_b in KL order (K,T): we need p "initial" columns at the right end
            X_b = np.zeros((K, T))
            # keep initial history from original X for stability (you can change later)
            X_b[:, -p:] = self.X[:, -p:]   # oldest p columns

            # simulate forward in KL-reverse indexing:
            # we fill columns from right-to-left? Actually newest is col0, oldest is colT-1.
            # We'll generate usable part col (T-p-1 down to 0) using already-filled columns to the right.
            for j in range(T - p - 1, -1, -1):
                # current time column j depends on lags at  j+1..j+p (to the right)
                z_lags = []
                for i in range(1, p + 1):
                    z_lags.append(X_b[:, j + i])  # (K,)
                z = np.concatenate(z_lags, axis=0)  # (Kp,)

                x = A_no_const @ z  # (K,)
                if c is not None:
                    x = x + c[:, 0]
                # map bootstrap residual for this observation:
                # j=0 corresponds to "newest" among usable obs, aligns with U_b[:,0] in our Y ordering
                x = x + U_b[:, j]
                X_b[:, j] = x
            return X_b
        
        def simulate_from_bootstrap_chrono() -> np.ndarray:
            """
            Симуляция VAR для bootstrap в ХРОНОЛОГИЧЕСКОМ порядке:
            col 0 = oldest (t=0), col -1 = newest (t=T-1)
            """
            # 1. Resample residuals (без изменений)
            idx = rng.integers(0, nobs, size=nobs)
            U_b = U[:, idx]  # (K, nobs)

            # 2. Build X_b in chronological order (K, T)
            X_b = np.zeros((K, T))
            
            # Инициализация: начальные лаги теперь в ЛЕВОЙ части (самые старые)
            # X_b[:, :p] = [Y_0, Y_{-1}, ..., Y_{-p+1}]
            X_b[:, :p] = self.X[:, :p]  # oldest p columns

            # 3. Симуляция вперед: от прошлого (p) к будущему (T-1)
            # j - это индекс текущего прогнозируемого периода
            for j in range(p, T):
                # Извлекаем лаги: они находятся ЛЕВЕЕ текущего индекса j
                z_lags = []
                for i in range(1, p + 1):
                    # Лаг 1:  j-1, Лаг 2: j-2, ..., Лаг p: j-p
                    z_lags.append(X_b[:, j - i])  # (K,)
                
                z = np.concatenate(z_lags, axis=0)  # (Kp,) вектор [Y_{t-1}; Y_{t-2}; ...]

                # Предсказание по модели
                x = A_no_const @ z  # (K,)
                if c is not None:
                    x = x + c[:, 0]
                
                # Добавляем шок
                # Важно: U_b индексируется от 0 до nobs-1
                # Период j=p соответствует первому наблюдению в U_b (индекс 0)
                x = x + U_b[:, j - p]
                
                X_b[:, j] = x
                
            return X_b

        irf_draws = np.zeros((n_boot, horizon + 1, K, K))

        for b in range(n_boot):
            if self.time_order == "KL_reverse":
                X_b = simulate_from_bootstrap()
            else:
                X_b = simulate_from_bootstrap_chrono()

            # Refit quickly using same routines
            tmp = SVAR_KL(
                data=pd.DataFrame(X_b, index=self.var_names, columns=self.time_index),
                p=self.p,
                exog=pd.DataFrame(self.exog) if self.exog is not None else None,
                layout="KL_KxT", 
                time_order=self.time_order,
                add_const=self.add_const,
                check_binary_collinearity=False,
                name=f"{self.name}_boot{b}",
                log_vars = self.log_vars,
                base_levels=self._base_levels, # Передаем log_vars
                u_dict=list(self.u_dict.values()),
                y_dict=list(self.y_dict.values()),
                verbose=False
            ).fit_ols()

            if scheme == "redo_cholesky":
                Q_b = np.eye(K)
                B0inv_b = tmp.P @ Q_b # pyright: ignore[reportOptionalOperand]

            elif scheme == "fixed_Q":
                B0inv_b = tmp.P @ self.Q # pyright: ignore[reportOptionalOperand]

            elif scheme == "redo_sign":
                rr = tmp.identify_sign_restrictions(
                    restrictions=sign_restrictions,
                    horizon=horizon,
                    n_draws=n_draws_sign,
                    seed=None if seed is None else (seed + 10_000 + b),
                )
                B0inv_b = rr.B0inv

            elif scheme == "redo_combined":
                kwargs = combined_kwargs or {}
                rr = tmp.identify_combined(seed=None if seed is None else (seed + 20_000 + b), **kwargs)
                B0inv_b = rr.B0inv
            elif scheme == 'innovations':
                B0inv_b = np.eye(K)

            else:
                raise ValueError(f"Unknown scheme: {scheme}")

            irf_draws[b] = tmp.irf(horizon=horizon, B0inv=B0inv_b)

        return irf_draws
    
    def print_upsilon(self):
        df = pd.DataFrame(self.Upsilon).round(3)\
                        .set_axis(list(self.y_dict.values()), axis = 0)\
                        .set_axis(list(self.u_dict.values()), axis = 1)
        return df

    # ----------------- 8) Historical decomposition -----------------
    def historical_decomposition(self, 
                                 B0inv: Optional[np.ndarray] = None, 
                                 plot_hd = False,
                                 init_clean = True,
                                 exog_clean = True,
                                 const_clean = True,
                                 shocks_contrib: bool = True) -> list:
        """
        Returns contributions: (K, nobs, K) where contributions[:,t,j] is the contribution
        of shock j to variables at time t (aligned with Y columns, KL order).
        """
        if self.E is None or self.A_endo_no_const is None:
            raise RuntimeError("Call fit_ols() first.")
        if B0inv is None:
            if self.B0inv is None:
                raise RuntimeError("No identification stored. Call identify_* first or pass B0inv.")
            B0inv = self.B0inv

        K = self.K
        nobs = self.E.shape[1]

        # structural shocks: u_t = B0inv * eps_t => eps_t = B0 * u_t
        B0 = np.linalg.inv(B0inv)
        Us = B0 @ self.E  # (K, nobs)

        # MA representation coefficients Phi_h = J' A_comp^h J (KxK)
        # contributions at time t: sum_{h=0..t} Phi_h B0inv e_{t-h} (with decomposition by shock)
        # We'll compute IRF on impact matrix B0inv, then apply eps shock-by-shock.
        # irfs[h] already equals Phi_h * B0inv.
        irfs = _irf_companion(self.A_endo_no_const, B0inv, horizon=nobs - 1)  # (nobs, K, K)

        contrib = np.zeros((K, nobs, K))
        for t in range(nobs):
            # y_t = sum_{h=0..t} irfs[h] @ eps_{t-h}
            # to decompose by shock j, take column j of irfs[h] times eps_j
            for h in range(t + 1):
                u = Us[:, t - h]  # (K,)
                # add per shock
                for j in range(K):
                    contrib[:, t, j] += irfs[h][:, j] * u[j]

        dec_list = []
        for v in range(self.K):
            dec_list.append(pd.DataFrame(np.vstack([self.Y[v],  # pyright: ignore[reportOptionalSubscript]
                             contrib[v, :, :].T]),
             index=[self.y_dict[self.var_names[v]]] + [self.u_dict[i] for i in self.var_names],
             columns=self.time_index[self.p:]))
        
        # get the other hd influence
        HD_init, HD_exog, HD_const, HD_nonendo = _initial_and_exog_hd(self.X,
                    self.A_hat,
                    self.p,
                    self.K,
                    self.T,
                    const=self.add_const,
                    exog_data=self.exog)
        additional_contrib = ['Ошибка модели']
        if not init_clean:
            additional_contrib.append('Вклад НУ')
        if not self.add_const and not const_clean:
            additional_contrib.append('Вклад константы')
        if self.exog is not None and not exog_clean:
            additional_contrib.append('Вклад экзопеременных')
        for var_idx in range(self.K):
            # === Вклад начального условия ===
            if not init_clean:
                # Показываем как отдельный компонент в графике
                dec_list[var_idx].loc['Вклад Y_0'] = HD_init[var_idx, :]
                
            else:
                # Скрываем компонент, но вычитаем его эффект из базового ряда
                dec_list[var_idx].iloc[0] -= HD_init[var_idx, :]
            
            # === Вклад константы ===
            if self.add_const:
                if not const_clean:
                    # Показываем вклад константы
                    dec_list[var_idx].loc['Вклад константы'] = HD_const[var_idx, :]
                    
                else:
                    # Вычитаем константу
                    dec_list[var_idx].iloc[0] -= HD_const[var_idx, :]
            
            # === Вклад экзогенных переменных ===
            if self.exog is not None:
                if not exog_clean:
                    # Показываем как отдельный компонент
                    dec_list[var_idx].loc['Вклад экзопеременных'] = HD_exog[var_idx, :]
                    
                else:
                    # Скрываем, но корректируем базовый ряд
                    dec_list[var_idx].iloc[0] -= HD_exog[var_idx, :]

        # get the plot     
        if plot_hd:
            for v in range(self.K):
                plot_hd_bars_signed(dec_list[v],
                                    cumm=True,
                                    K=self.K) # pyright: ignore[reportCallIssue]
        if shocks_contrib == True:
            share_hd_full = pd.DataFrame(columns=list(self.y_dict.values()))
            for i in range(self.K):
                share_hd = dec_list[i].copy()
                share_hd.loc['Необъяснено шоками'] = share_hd.iloc[0] - share_hd.iloc[1:].sum(axis = 0)
                share_hd = share_hd.abs().copy()
                share_hd.iloc[0] = share_hd.iloc[1:].sum(axis = 0)
                share_array = share_hd.sum(axis=1)
                share_hd_full.iloc[:,i] = share_array.iloc[1:]/share_array.iloc[0]
            share_hd_full = (share_hd_full*100).astype(float).round(1)
            share_hd_full = share_hd_full.set_axis(list(self.u_dict.values()) + additional_contrib, axis = 0)
            share_hd_full = share_hd_full.set_axis(list(self.y_dict.values()), axis = 1)

            
        return dec_list, share_hd_full

    def plot_irfs_grid(self,
                       irf_sims: list, # follow that rule index = 0 is the original irf, the others are sims
                       horizon_plot: int, 
                       main_color: str = "cadetblue",
                       sign_draws: bool = False,
                       percent: bool = True,
                       innovations: bool = False,
                       figsize: tuple = (9, 15)):
        # check of the horizon
        if horizon_plot > irf_sims[0].shape[0] - 1:
            horizon_plot = irf_sims[0].shape[0] - 1
            print('The horizon_plot is greater than it was simulated. Max horizon was obtained from sims')
        
        if percent:
            irf_sims = [sim * self._scale_factors_irf for sim in irf_sims]

        plot_irf_grid(irf_sims=irf_sims,
                        y_labels=list(self.y_dict.values()),
                        u_labels=list(self.y_dict.values()) if innovations else list(self.u_dict.values()),
                        horizon=horizon_plot,
                        ci_color=main_color,
                        sign_draws=sign_draws,
                        figsize = figsize)

    def plot_single_irf(self,
                        irf_sims: np.ndarray,
                        variable: str,
                        shock: str,
                        ci_color: str,
                        plot_simulations: bool,
                        horizon: int,
                        cumm: bool = False,
                        sign_draws: bool = False,
                        percent: bool = True,
                        figsize: tuple = (9, 5)):
        if horizon > np.asarray(irf_sims).shape[1]-1:
            horizon = np.asarray(irf_sims).shape[1]-1
            print('The horizon_plot is greater than it was simulated. Max horizon was obtained from sims')
        if variable not in self.var_names:
            raise ValueError(f"Unknown variable: {variable}. Check model.var_names")
        if shock not in list(self.u_dict.values()):
            raise ValueError(f"Unknown shock: {shock}. Check model.u_dict values")
        
        i, j = self.var_names.index(variable), list(self.u_dict.values()).index(shock)
        
        if percent:
            irf_sims = [sim * self._scale_factors_irf for sim in irf_sims]

        plot_irf_single(irf_sims, i, j, 
                        horizon,
                        self.u_dict,
                        self.y_dict,
                        plot_simulations,
                        ci_color=ci_color,
                        cumm=cumm,
                        figsize = figsize)
    
    def counterfactual_policy(
                        self,
                        policy_shock_index: list,
                        target_series_index: list,
                        loss_function: Callable = None,
                        ar_order: int = None,
                        n_simulations: int = 100,
                        initial_pacf: Optional[np.ndarray] = None,
                        pacf_bounds: Tuple[float, float] = (-0.95, 0.95),
                        verbose: bool = True,
                        **kwargs
                        ) -> CounterfactualResult:
        """
        Контрфактический анализ через поиск новых шоков.
        Обязательно передать шоки политики и функцию потерь и ВСЕ аргументы к ней. Все аргументы вводить через keywords
        Также нужно передать либо стартовые PACF для шоков либо задать лаг p для каждого процесса генерации AR(p)
        Посмотрите kwargs этой функции и не передавайте функцию потерь, в которой есть эти аргументы
        Parameters
        ----------
        policy_shock_index : list
            Индексы шоков политики, если шок один, то можно передать [шок_политики]
        target_series_index : list
            Целевые переменные для функции потерь
        loss_function : Callable, optional
            Функция потерь по целевым переменным
            Сигнатура: loss_function(target_series,**kwargs) -> float
        ar_order : int
            Порядок AR процесса для генерации шока (по умолчанию соответствует количеству лагов в модели).
        n_simulations : int
            Количество попыток оптимизации с разными seed.
        initial_pacf : np.ndarray, optional
            Начальные значения PACF. Если None, используется нулевой вектор.
        pacf_bounds : tuple
            Границы для PACF (гарантируют стационарность).
        verbose : bool
            Выводить ли прогресс оптимизации.
        
        Returns
        -------
        CounterfactualResult
            Объект с результатами контрфактического анализа.
        """

        if differential_evolution is None:
            raise ImportError("scipy.optimize.differential_evolution required for counterfactual analysis")
        
        if self.E is None or self.B0inv is None:
            raise RuntimeError("Call fit_ols() and identify_*() first")
        
        if loss_function is None:
            raise ValueError("Функцию потерь дай мне, а, чертила? Смотри докстринг")
        
        if ar_order is None:
            ar_order = self.p

        # === Подготовка данных ===
        K, T = self.K, self.T
        p = self.p
        nobs = T - p
        
        # Структурные шоки
        B0 = np.linalg.inv(self.B0inv)
        Us = B0 @ self.E  # (K, nobs)
        
        # Разделяем шоки политики и остальные
        if len(policy_shock_index) > 1:
            policy_shocks = Us[policy_shock_index, :].copy()
        else:
            policy_shocks = Us[policy_shock_index, :].copy()[np.newaxis, :]
        U_other = np.delete(Us, policy_shock_index, axis=0)  # (K-len(policy_shock_index), nobs)
        
        # IRF для всех комбинаций переменных и шоков
        irfs = self.irf(horizon=nobs)  # (horizon+1, K, K)
        contrib_list = list(product(target_series_index, policy_shock_index))
        irf_for_clean = [(target_series_index.index(ts), 
                          policy_shock_index.index(ps), 
                          irfs[:, ts, ps],
                          ts,
                          ps) for ts, ps in contrib_list]
        
        # Целевые переменные без вклада исторических шоков политики
        if len(target_series_index) > 1:
            target_series = self.Y[target_series_index, :].copy()  
        else:
            target_series = self.Y[target_series_index, :].copy()[np.newaxis, :]  

        target_series_clean = target_series.copy()
        for irf in irf_for_clean:
            ts_idx, ps_idx, irf_vals = irf[0], irf[1], irf[2]
            
            # Берем шок и обязательно делаем его плоским вектором (1D)
            shock_vector = policy_shocks[ps_idx, :].flatten()
            
            contrib = np.convolve(shock_vector, irf_vals, mode='full')[:nobs]
            target_series_clean[ts_idx, :] -= contrib

        # === Функция потерь для оптимизации ===
        def loss_for_optimization(pacfs, sim_idx, U_other, policy_shocks, 
                          target_series_clean, irf_for_clean, ar_order, 
                          loss_function, kwargs_dict):
            nobs = U_other.shape[1]
            n_policy_shocks = policy_shocks.shape[0]
            new_policy_shocks = policy_shocks.copy()
            
            # Список для хранения уже ортогонализованных шоков (для взаимной ортогональности)
            
            if n_policy_shocks > 1:
                ortho_policy_list = []
                for i in range(n_policy_shocks):
                    seed = get_independent_seeds(sim_idx, i)
                    eta = _generate_fixed_innovations(nobs, burn_in=100, seed=seed)
                    
                    # Динамическая индексация PACF
                    start_idx = i * ar_order
                    end_idx = start_idx + ar_order
                    rho = _pacf_to_ar(pacfs[start_idx:end_idx])
                    
                    xi = _generate_ar_series_fixed(rho, eta)
                    ortho_policy_list.append(xi)
                ortho_candidates_block = np.array(ortho_policy_list)
                new_policy_shocks = orthogonalize_shock_block(ortho_candidates_block, U_other)
            else:
                seed = get_independent_seeds(sim_idx, 0)
                eta = _generate_fixed_innovations(nobs, burn_in=100, seed=seed)
                
                # Динамическая индексация PACF
                start_idx = 0
                end_idx = start_idx + ar_order
                rho = _pacf_to_ar(pacfs[start_idx:end_idx])
                
                xi = _generate_ar_series_fixed(rho, eta)
                new_policy_shocks[0, :] = _orthogonalize_and_normalize_shock(xi, U_other.T).flatten()

            # Считаем новые целевые переменные
            new_target_series = target_series_clean.copy()
            for irf in irf_for_clean:
                contrib = np.convolve(new_policy_shocks[irf[1], :].flatten(), irf[2], mode='full')[:nobs]
                new_target_series[irf[0], :] += contrib
            
            loss = loss_function(new_target_series, **kwargs_dict)
            return loss

        # === Оптимизация === 
        best_loss = np.inf
        best_pacf = None
        best_sim = None
        loss_history = []
        
        n_params = ar_order * len(policy_shock_index)
        if initial_pacf is None:
            initial_pacf = np.zeros(n_params)
        
        # Границы для каждого параметра
        bounds = [pacf_bounds] * n_params
        
        for sim_idx in range(n_simulations):
            seed_de = get_independent_seeds(sim_idx, len(policy_shock_index))
            
            result = differential_evolution(
                loss_for_optimization,
                bounds,
                args=(sim_idx, U_other, policy_shocks, 
                      target_series_clean, irf_for_clean, ar_order, 
                      loss_function, kwargs),
                seed=seed_de,
                maxiter=500,
                popsize=15,
                tol=1e-6,
                polish=True
            )
            
            loss_history.append(result.fun)
            
            if result.fun < best_loss:
                best_loss = result.fun
                best_pacf = result.x.copy()
                best_sim = sim_idx
            
            if verbose:
                sys.stdout.write(f"\rМинимальное значение функции потерь {best_loss:.6f}, "
                            f"сделано симуляций {sim_idx+1}/{n_simulations} "
                            f"({100*(sim_idx+1)/n_simulations:.1f}%)")
                sys.stdout.flush()
        
        if verbose:
            print(f"\nОптимальные PACF: {best_pacf}")
            ar_coefs = [_pacf_to_ar(best_pacf[i*ar_order:(i+1)*ar_order]) 
                    for i in range(len(policy_shock_index))]
            print(f"Оптимальные AR коэффициенты: {ar_coefs}")
            print(f"Минимальные потери: {best_loss:.6f}")

        # === Генерация контрфактических данных ===
        best_policy_shocks = policy_shocks.copy()
        for i in range(len(best_policy_shocks)):
            seed = get_independent_seeds(best_sim, i)
            eta = _generate_fixed_innovations(nobs, burn_in=100, seed=seed)
            rho = _pacf_to_ar(best_pacf[i*ar_order:(i+1)*ar_order])
            xi = _generate_ar_series_fixed(rho, eta)
            best_policy_shocks[i, :] = _orthogonalize_and_normalize_shock(xi, U_other.T)

        # Создаем новые данные (используем numpy для индексации)
        new_data = self._raw_data.copy(deep=True)  
        for ts in range(K):
            for irf in irf_for_clean:
                contrib_new = np.convolve(best_policy_shocks[irf[1], :].flatten(), irfs[:, ts, irf[4]], mode='full')[:nobs]
                contrib_old = np.convolve(policy_shocks[irf[1], :].flatten(), irfs[:, ts, irf[4]], mode='full')[:nobs]
                new_data.iloc[ts, p:] = new_data.iloc[ts, p:] + contrib_new - contrib_old

        # Создаем новую модель
        counterfactual_model = SVAR_KL(
            data=new_data,
            p=self.p,
            exog=self._raw_exog,
            layout=self.layout,
            time_order=self.time_order,
            add_const=self.add_const,
            check_binary_collinearity=False,
            name=f"{self.name}_counterfactual",
            log_vars=self.log_vars, # Передаем log_vars
            base_levels=self._base_levels,
            u_dict=list(self.u_dict.values()),
            y_dict=list(self.y_dict.values()),
            verbose = False
        )

        # Копируем оцененные параметры
        counterfactual_model.Y = self.Y
        counterfactual_model.Z = self.Z
        counterfactual_model.A_hat = self.A_hat
        counterfactual_model.E = None
        counterfactual_model.Sigma_u = self.Sigma_u
        counterfactual_model.P = self.P
        counterfactual_model.A_hat_endo = self.A_hat_endo
        counterfactual_model.A_endo_no_const = self.A_endo_no_const

        # Создаем результат
        result = CounterfactualResult(
            original_model=self,
            counterfactual_model=counterfactual_model,
            optimal_pacf=best_pacf,
            optimal_seed=best_sim,  
            loss_value=best_loss,
            loss_history=loss_history,
            shock_original=policy_shocks,
            shock_counterfactual=best_policy_shocks,
            metadata={
                'ar_order': ar_order,
                'n_simulations': n_simulations,
                'loss_function': loss_function.__name__ if hasattr(loss_function, '__name__') else 'custom',
                'lf_params': kwargs,
                'new_data': new_data
            }
        )

        return result
    
    def cf_policy_space(
            self,
            policy_shock_index: list,
            n_simulations: int = 100,
            ar_order: int = 2,
            pacf_bounds: Tuple[float, float] = (-0.95, 0.95)
    ) -> dict: 
        """
            Создание контрфактических симуляций шоков политики.
    
    Алгоритм:
    1. Очищает целевые ряды от вклада исторических шоков политики
    2. Генерирует случайные стационарные шоки через PACF-параметризацию
    3. Для каждой симуляции считает новый путь целевых переменных
    4. Строит график: оригинал, очищенный ряд, медиана симуляций + ДИ
    
    Parameters
    ----------
    model : SVAR_KL
        Оцененная модель с идентификацией
    policy_shock_index : list
        Индексы целевых переменных для визуализации
    n_simulations : int
        Количество симуляций для построения распределения
    ar_order : int
        Порядок AR процесса для генерации шоков
    pacf_bounds : tuple
        Границы для PACF (гарантируют стационарность)
    ci_levels : tuple
        Уровни доверительных интервалов в процентах
    figsize : tuple
        Размер фигуры (width, height)
    colors : dict, optional
        Словарь цветов для элементов графика
    seed_base : int
        Базовый seed для воспроизводимости
    
    Returns
    -------
    results : dict
        Словарь с симулированными траекториями и статистиками
        """
        
        if self.E is None or self.B0inv is None:
            raise RuntimeError("Call fit_ols() and identify_*() first")
        
        if ar_order is None:
            ar_order = self.p

        # === Подготовка данных (блок как в counterfactual_policy) ===
        K, T = self.K, self.T
        p = self.p
        nobs = T - p
        
        # Структурные шоки
        B0 = np.linalg.inv(self.B0inv)
        Us = B0 @ self.E  # (K, nobs)
        
        # Разделяем шоки политики и остальные
        if len(policy_shock_index) > 1:
            policy_shocks = Us[policy_shock_index, :].copy()
        else:
            policy_shocks = Us[policy_shock_index, :].copy()[np.newaxis, :]
        U_other = np.delete(Us, policy_shock_index, axis=0)  # (K-len(policy_shock_index), nobs)
        
        # IRF для всех комбинаций переменных и шоков
        irfs = self.irf(horizon=nobs)  # (horizon+1, K, K)

        
        # Очищенные переменные без вклада исторических шоков политики
        series = self.Y.copy() 

        series_clean = series.copy()
        for k in range(self.K):
            for ps in policy_shock_index:
                shock_vector = Us[ps, :].flatten()
           
                contrib = np.convolve(shock_vector, irfs[:, k, ps], mode='full')[:nobs]
                series_clean[k, :] -= contrib

        # ====== Симуляции =======
        simulations = {k: np.zeros((n_simulations, nobs)) for k in range(self.K)}
    
        for sim_idx in range(n_simulations):
            # Генерация новых шоков политики для каждой симуляции
            new_policy_shocks = policy_shocks.copy()
            
            for i in range(len(policy_shock_index)):
                seed = get_independent_seeds(sim_idx, i)
                eta = _generate_fixed_innovations(nobs, burn_in=100, seed=seed)
                
                # Случайные PACF в допустимых границах
                pacf = np.random.uniform(pacf_bounds[0], pacf_bounds[1], size=ar_order)
                rho = _pacf_to_ar(pacf)
                
                xi = _generate_ar_series_fixed(rho, eta)
                ps_new = _orthogonalize_and_normalize_shock(xi, U_other.T)
                new_policy_shocks[i, :] = ps_new
            
            # Расчет новых траекторий целевых переменных
            for k in range(self.K):
                new_series = series_clean[k, :].copy()
                for ps in policy_shock_index:

                    shock_local_idx = policy_shock_index.index(ps)
                    contrib = np.convolve(new_policy_shocks[shock_local_idx, :].flatten(), irfs[:, k, ps], mode='full')[:nobs]
                    new_series += contrib
                simulations[k][sim_idx, :] = new_series

        # ====== Словарь с симами ======
        sim_result = {}
        for k in range(self.K):
            dict_serie = {'original': series[k,:],
                          'clean':    series_clean[k,:],
                          'sims':     simulations[k],
                          'policy':   policy_shock_index,
                          'name':     list(self.y_dict.values())[k]}
            var_names_list = list(self.y_dict.keys())
            sim_result[var_names_list[k]] = dict_serie

        sim_result['policy_shocks'] = policy_shock_index

        return sim_result
    def plot_policy_space(
            self,
            cf_results: dict,
            time_index: pd.Index = None,
            ci_levels: Tuple[int, int] = (68, 90),
            figsize: Tuple[int, int] = (14, 10),
            colors: dict = None,
            policy_shock_labels: Optional[list] = None,
            n_simulations_plot: int = 10,
            suptitle: str = "Контрфактические симуляции шоков политики",
            ):
        """
        Графическая визуализация пространства политики
    

    4. Строит график: оригинал, очищенный ряд, медиана симуляций + ДИ
    
    Parameters
    ----------
    sim_result : Результат симуляций с помощью cf_policy_space
    ci_levels : tuple
        Уровни доверительных интервалов в процентах
    figsize : tuple
        Размер фигуры (width, height)
    colors : dict, optional
        Словарь цветов для элементов графика
    seed_base : int
        Базовый seed для воспроизводимости
    
    Returns
    -------
    График
        """
        if policy_shock_labels is None:
            policy_shock_labels = [list(self.u_dict.values())[i] for i in cf_results['policy_shocks']]

        if time_index is None:
            time_index = self.time_index[self.p:]

        plot_cf_policy_space_single(
            cf_results = cf_results,
            time_index = time_index,
            ci_levels = ci_levels,
            figsize = figsize,
            colors = colors,
            policy_shock_labels = policy_shock_labels,
            n_simulations_plot = n_simulations_plot,
            suptitle = suptitle)

    def plot_single_var_grid(
                self,
                irf_sims,
                response_var: int = 0,
                innovations: bool = False,
                horizon: Optional[int] = None,
                n_cols: int = 2,
                figsize: Tuple[int, int] = (12, 10),
                title : Optional[str] = None
    ):
        """
        Строит сетку для одной переменной

        Параметры
    ----------
    irf_sims : array-like
        Либо список массивов формы (horizon+1, K, K) для каждого draw,
        либо массив формы (n_draws, horizon+1, K, K).
    response_var : int, optional
        Индекс переменной, на которую смотрим отклик (по умолчанию 0).
    y_label : str, optional
        Название переменной отклика. Если None, будет использован индекс.
    u_labels : sequence[str], optional
        Названия для шоков (столбцы). Если None, будут использованы индексы.
    horizon : int, optional
        Горизонт (число периодов). Если None, берётся из формы данных.
    n_cols : int, optional
        Количество столбцов в сетке графиков.
    figsize : tuple, optional
        Размер фигуры (width, height).
    title : str, optional
        Заголовок (suptitle).
        """
        if isinstance(response_var, str):
            if response_var not in list(self.y_dict.keys()) or response_var not in list(self.y_dict.values()):
                print('Не могу найти переменную, отклики которой строятся')
            else:
                if response_var not in self.y_dict.keys():
                    y_label = self.y_dict[response_var]
                    response_var = list(self.y_dict.keys()).index(response_var)
                    
                else:
                    y_label = response_var
                    response_var = list(self.y_dict.values()).index(response_var)
        else:
            y_label = list(self.y_dict.values())[response_var]

        
        plot_irf_single_var_grid(
                irf_sims = irf_sims,
                response_var = response_var,
                y_label = y_label,
                u_labels = list(self.y_dict.values()) if innovations else list(self.u_dict.values()),
                horizon = horizon,
                n_cols = n_cols,
                figsize = figsize,
                title = title)        