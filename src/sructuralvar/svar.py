from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Literal, Any

import numpy as np
import pandas as pd

try:
    from scipy.linalg import cholesky, qr
    from scipy.optimize import minimize
except Exception:  # allow importing without scipy; raise when used
    cholesky = None
    qr = None
    minimize = None


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
    Q, R = qr(A)
    # Fix sign ambiguity for reproducibility-ish
    d = np.sign(np.diag(R))
    d[d == 0] = 1.0
    return Q * d


def _build_companion(A_endo_no_const: np.ndarray, K: int, p: int) -> np.ndarray:
    """
    A_endo_no_const: (K, K*p) reduced-form coefficients excluding constant, ordered [A1|A2|...|Ap]
    companion size: (K*p, K*p)
    """
    if A_endo_no_const.shape != (K, K * p):
        raise ValueError(f"A_endo_no_const must be (K, K*p) = ({K},{K*p}), got {A_endo_no_const.shape}")

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


@dataclass
class RestrictionResult:
    Q: np.ndarray
    B0inv: np.ndarray
    details: Dict[str, Any]


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
        time_order: TimeOrder = "KL_reverse",
        add_const: bool = True,
        check_binary_collinearity: bool = True,
        name: str = "SVAR_KL",
    ):
        self.name = name
        self.p = int(p)
        if self.p < 1:
            raise ValueError("p must be >= 1")

        self.add_const = bool(add_const)
        self.time_order = time_order

        self._raw_data = data.copy()
        self._raw_exog = exog.copy() if exog is not None else None

        # infer / validate orientation
        self.layout = self._infer_layout(data, layout=layout)
        self.X, self.var_names, self.time_index = self._coerce_to_KT(data, layout=self.layout)
        self.K, self.T = self.X.shape

        if self.T <= self.p:
            raise ValueError(f"Need T > p. Got T={self.T}, p={self.p}")

        self.exog = None
        self.exog_names = None
        if exog is not None:
            ex_layout = self._infer_layout(exog, layout=layout)  # usually same convention
            E, ex_names, ex_time = self._coerce_to_KT(exog, layout=ex_layout)
            if E.shape[1] != self.T:
                raise ValueError("Exog must have same T (columns/time) as endog in KL layout.")
            self.exog = E
            self.exog_names = ex_names

        if self.time_order == "chronological":
            # превращаем chrono (oldest->newest) в KL_reverse (newest->oldest)
            self.X = self.X[:, ::-1]
            if self.exog is not None:
                self.exog = self.exog[:, ::-1]
        elif self.time_order != "KL_reverse":
            raise ValueError(f"Unknown time_order: {self.time_order}")
        
        self._validate_values(self.X, label="endog")
        if self.exog is not None:
            self._validate_values(self.exog, label="exog")

        if check_binary_collinearity:
            self._check_binary_multicollinearity(self._raw_data)

        # --------- OLS outputs (filled by fit_ols) ----------
        self.Y: Optional[np.ndarray] = None         # (K, T-p)
        self.Z: Optional[np.ndarray] = None         # (nreg, T-p)
        self.B_hat: Optional[np.ndarray] = None     # (K, nreg)
        self.E: Optional[np.ndarray] = None         # (K, T-p)
        self.Sigma_u: Optional[np.ndarray] = None   # (K, K)
        self.P: Optional[np.ndarray] = None         # (K, K) Cholesky of Sigma_u (lower)

        # convenience slices
        self.B_hat_endo: Optional[np.ndarray] = None      # (K, 1+Kp) if const else (K, Kp)
        self.A_endo_no_const: Optional[np.ndarray] = None # (K, Kp)

        # --------- identification outputs ----------
        self.Q: Optional[np.ndarray] = None         # (K, K)
        self.B0inv: Optional[np.ndarray] = None     # (K, K)
        self.Upsilon: Optional[np.ndarray] = None   # (K,K) long-run matrix (Λ/Υ in your notation)

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
            var_names = [str(i) for i in df.index] if df.index is not None else [f"y{i}" for i in range(X.shape[0])]
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
            names = [str(i) for i in df.index]
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
        KL_reverse (default): X[:,0] newest, X[:,-1] oldest.
        For VAR(p), usable obs = T-p (columns 0..T-p-1).
        """
        X = self.X
        K, T = X.shape
        p = self.p
        nobs = T - p

        # Y = [X_t, X_{t-1}, ..., X_{p+1}]  -> first T-p columns
        Y = X[:, :nobs]  # (K, T-p)

        Z_blocks = []
        if self.add_const:
            Z_blocks.append(np.ones((1, nobs)))

        # lag i: shift right by i
        for i in range(1, p + 1):
            Z_blocks.append(X[:, i:i + nobs])  # (K, T-p)

        Z = np.vstack(Z_blocks)  # (1+Kp, T-p) or (Kp, T-p)

        # Optional: append exog contemporaneous (aligned with Y) if provided
        # NOTE: you can extend later with lagged exog; пока — минимально.
        if self.exog is not None:
            E = self.exog[:, :nobs]  # align same columns as Y
            Z = np.vstack([Z, E])

        return Y, Z

    # ----------------- 2) OLS estimation -----------------
    def fit_ols(self) -> "SVAR_KL":
        """
        Estimates: Y = B_hat Z + E  (KL matrix form)
        Stores attributes similar to your OLS_estimation output.
        """
        Y, Z = self._build_YZ()
        # B_hat = Y Z' (Z Z')^{-1}
        ZZt = Z @ Z.T
        YZt = Y @ Z.T
        B_hat = YZt @ np.linalg.inv(ZZt)
        E = Y - B_hat @ Z

        # Sigma_u (unbiased-ish): E E' / (T-p - nreg) is common; KL often uses / (T-p)
        # We'll store both pieces later if you want. For now: / (T-p).
        nobs = Y.shape[1]
        Sigma_u = (E @ E.T) / nobs

        if cholesky is None:
            raise ImportError("scipy is required for Cholesky decomposition.")
        P = cholesky(Sigma_u, lower=True)

        self.Y, self.Z, self.B_hat, self.E = Y, Z, B_hat, E
        self.Sigma_u, self.P = Sigma_u, P

        # convenience: endog-only coefficient block
        # Z contains [const?; lags; exog?]. We split the endog part = const + K*p.
        endo_reg_count = (1 if self.add_const else 0) + self.K * self.p
        self.B_hat_endo = B_hat[:, :endo_reg_count]
        # A_endo_no_const: remove constant if present
        self.A_endo_no_const = self.B_hat_endo[:, 1:] if self.add_const else self.B_hat_endo.copy()

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

    # ----------------- 5) Long-run restrictions (zeros/signs) -----------------
    # We'll implement via combined minimization (item 6) as the core engine,
    # because long-run zeros typically require optimization unless you use
    # a direct parameterization.

    # ----------------- 6) Combined identification via minimization -----------------
    def identify_combined(
        self,
        short_run_zeros: Sequence[Tuple[int, int]] = (),
        long_run_zeros: Sequence[Tuple[int, int]] = (),
        short_sign_restrictions: Sequence[Tuple[int, int, int, int]] = (),
        long_sign_restrictions: Sequence[Tuple[int, int, int]] = (),
        horizon_for_short_sign: Optional[int] = None,  # used only for sanity check
        n_starts: int = 50,
        seed: Optional[int] = None,
        method: str = "BFGS",
    ) -> RestrictionResult:
        """
        Minimizes a penalty to satisfy:
          - short_run_zeros on B0inv (impact)
          - long_run_zeros on Cinf
          - short_sign_restrictions on IRF(h)
          - long_sign_restrictions on Cinf

        short_sign_restrictions: (row, col, h, sign)
        long_sign_restrictions: (row, col, sign)
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
            for (_, _, h, _) in short_sign_restrictions:
                if h > horizon_for_short_sign:
                    raise ValueError("A short-run sign restriction has h > horizon_for_short_sign.")

        def penalty(params: np.ndarray) -> float:
            Q = _givens_Q_from_params(params, K)
            B0inv = self.P @ Q
            Upsilon = _long_run_matrix(self.A_endo_no_const, B0inv)

            loss = 0.0

            # short-run zeros (quadratic)
            for (i, j) in short_run_zeros:
                loss += float(B0inv[i, j] ** 2)

            # long-run zeros (quartic, stronger)
            for (i, j) in long_run_zeros:
                loss += float(Upsilon[i, j] ** 4)

            # short-run sign restrictions (via IRF)
            if short_sign_restrictions:
                max_h = max(h for (_, _, h, _) in short_sign_restrictions)
                irfs = _irf_companion(self.A_endo_no_const, B0inv, horizon=max_h)
                for (r, c, h, sgn) in short_sign_restrictions:
                    v = irfs[h, r, c]
                    # smooth asymmetric penalty: exp(-sgn*v)-1
                    loss += float(np.exp(-sgn * v) - 1.0)

            # long-run sign restrictions on Cinf
            for (r, c, sgn) in long_sign_restrictions:
                v = Upsilon[r, c]
                loss += float(np.exp(-sgn * v) - 1.0)

            return loss

        best = {"fun": np.inf, "x": None, "res": None}

        for s in range(n_starts):
            x0 = rng.uniform(0.0, 2 * np.pi, size=n_params)
            res = minimize(penalty, x0=x0, method=method)
            if res.fun < best["fun"]:
                best = {"fun": float(res.fun), "x": res.x.copy(), "res": res}

        if best["x"] is None:
            raise RuntimeError("Optimization failed for all starts.")

        Q = _givens_Q_from_params(best["x"], K)
        B0inv = self.P @ Q
        Upsilon = _long_run_matrix(self.A_endo_no_const, B0inv)

        self.Q, self.B0inv, self.Upsilon = Q, B0inv, Upsilon
        return RestrictionResult(
            Q=Q,
            B0inv=B0inv,
            details={"type": "combined", "best_fun": best["fun"], "method": method, "n_starts": n_starts, "seed": seed},
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
        if self.B_hat_endo is None or self.E is None or self.A_endo_no_const is None:
            raise RuntimeError("Call fit_ols() first.")
        if scheme == "fixed_Q" and self.Q is None:
            raise RuntimeError("scheme='fixed_Q' requires stored Q from identify_*.")

        rng = np.random.default_rng(seed)
        K, T = self.K, self.T
        p = self.p
        nobs = T - p

        # Extract reduced-form A's and const
        if self.add_const:
            c = self.B_hat_endo[:, [0]]  # (K,1)
            A_no_const = self.A_endo_no_const  # (K,Kp)
        else:
            c = None
            A_no_const = self.A_endo_no_const

        # residuals aligned with Y (K, nobs)
        U = self.E

        def simulate_from_bootstrap() -> np.ndarray:
            # Resample residual columns (u_t) with replacement over nobs
            idx = rng.integers(0, nobs, size=nobs)
            U_b = U[:, idx]  # (K,nobs)

            # Build X_b in KL order (K,T): we need p "initial" columns at the right end
            X_b = np.zeros((K, T))
            # keep initial history from original X for stability (you can change later)
            X_b[:, -p:] = self.X[:, -p:]  # oldest p columns

            # simulate forward in KL-reverse indexing:
            # we fill columns from right-to-left? Actually newest is col0, oldest is colT-1.
            # We'll generate usable part col (T-p-1 down to 0) using already-filled columns to the right.
            for j in range(T - p - 1, -1, -1):
                # current time column j depends on lags at j+1..j+p (to the right)
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

        irf_draws = np.zeros((n_boot, horizon + 1, K, K))

        for b in range(n_boot):
            X_b = simulate_from_bootstrap()

            # Refit quickly using same routines
            tmp = SVAR_KL(
                data=pd.DataFrame(X_b, index=self.var_names, columns=self.time_index),
                p=self.p,
                layout="KL_KxT",
                time_order=self.time_order,
                add_const=self.add_const,
                check_binary_collinearity=False,
                name=f"{self.name}_boot{b}",
            ).fit_ols()

            if scheme == "redo_cholesky":
                Q_b = np.eye(K)
                B0inv_b = tmp.P @ Q_b

            elif scheme == "fixed_Q":
                B0inv_b = tmp.P @ self.Q

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

            else:
                raise ValueError(f"Unknown scheme: {scheme}")

            irf_draws[b] = tmp.irf(horizon=horizon, B0inv=B0inv_b)

        return irf_draws

    # ----------------- 8) Historical decomposition -----------------
    def historical_decomposition(self, B0inv: Optional[np.ndarray] = None) -> np.ndarray:
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

        # structural shocks: u_t = B0inv * eps_t  => eps_t = B0 * u_t
        B0 = np.linalg.inv(B0inv)
        eps = B0 @ self.E  # (K, nobs)

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
                e = eps[:, t - h]  # (K,)
                # add per shock
                for j in range(K):
                    contrib[:, t, j] += irfs[h][:, j] * e[j]
        return contrib