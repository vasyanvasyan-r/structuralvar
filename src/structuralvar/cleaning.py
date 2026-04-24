import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm

from scipy.stats import chi2
import inspect

import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning

warnings.simplefilter("ignore", InterpolationWarning)

from typing import Optional, Tuple, Union

class Clean:
        
    doc_en = """    
    A package for selecting an appropriate Vector Autoregression (VAR) model
    and estimating impulse response functions, developed by K and K.
    ...
    """
    doc_ru = """
    Пакет для подбора корректной модели векторной авторегрессии и оценки импульсных откликов, сделанный К и К
    ...
    """

    def __init__(self, 
                 timeseries: Union[np.ndarray, pd.DataFrame],
                 list_with_names: Optional[list] = [], 
                 i_criteria: Optional[str] = 'aic',
                 p_max: Optional[int] = 10) -> None:
        
        if isinstance(timeseries, pd.DataFrame):
            print("Это pandas DataFrame!")
            self.df = timeseries.copy()
            timeseries = timeseries.sort_index(ascending=False).to_numpy().T
            self.var_names = [f'var_{i}' for i in range(1, timeseries.shape[0] + 1)] if self.df.columns.to_list() == [] else self.df.columns.to_list()

        elif isinstance(timeseries, np.ndarray):
            print("Это NumPy массив!")
            self.var_names = [f'var_{i}' for i in range(1, timeseries.shape[0] + 1)] if list_with_names == [] else list_with_names
        else:
            raise ValueError("timeseries должен быть либо DataFrame, либо ndarray")
        
        self.X: np.ndarray = timeseries
        self.p_max: int = p_max
        self.i_criteria: Optional[str] = i_criteria
        self.K, self.T = timeseries.shape
        self.clean_params = {}  # === хранилище коэффициентов очистки ===

    def call_with_kwargs(self, func, **kwargs):
        sig = inspect.signature(func)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return filtered

    def OLS_estimation(self, Nseries, lag, from_class : bool = True, add_const : bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
        if from_class:
            X = self.X; p = self.p_max; T = self.T
        else:
            X = Nseries; p = lag; T = X.shape[1]

        Z = np.concatenate(
            [np.concatenate([np.array([1]) if add_const else None, X[:, i:i+p].T.flatten()]).reshape(-1, 1)
             for i in range(T - p)],
            axis=1
        )[:, 1:]

        Y = X[:, :Z.shape[1]]
        A_hat = Y @ Z.T @ np.linalg.inv(Z @ Z.T)
        return Z, Y, A_hat, Y - A_hat @ Z, X.shape[0], T
    
    def remove_na(self, series : pd.Series = None, var_name : str = None, from_class : bool = True):
        if from_class:
            if var_name in self.var_names:
                series = self.df.loc[:, var_name]
            else:
                print("Нет такой переменной в списке переменных")
                return None, None
        else:
            if series is not None and not series.empty:
                print("Для очистки от сезонности и трендов использую переданный мне временной ряд")
            else:
                print("Проверьте, что вы передали мне хотя бы что-то одно: или название переменной или временной ряд")
                return None, None

        series = series.reset_index(drop = True).copy()
        not_na_list = series.loc[series.notna()].index.to_list()
        start_index, stop_index = not_na_list[0], not_na_list[-1]
        series = series.loc[start_index:stop_index].copy()
        na_list = series.loc[series.isna()].index.to_list()
        not_na_list = series.loc[series.notna()].index.to_list()

        if len(not_na_list) > 1:
            gaps = [(i, j, [k for k in range(j+1, i)]) for i, j in zip(not_na_list[1:], not_na_list[:-1]) if i-j > 1]
            fill_na = []
            for gap in gaps:
                difference = (series.loc[gap[0]] - series.loc[gap[1]], gap[0] - gap[1])
                if difference[1] > 2:
                    print(f"Обрати внимание, что для переменной var_name есть пробел в данных\nмежду {gap[0]} и {gap[1]} наблюдениями длинной больше 2")
                fill_na += [difference[0]*i/difference[1] for i in range(1, difference[1])]
            series.loc[na_list] = fill_na
        else:
            print("Ошибка, всего одно наблюдение, это за ряд-то такой")

        return series, series.iloc[[0,-1]].index.to_list()
    
    def deseason_and_detrend(self, series: pd.Series = None, var_name: str = None, period: int = 12,
                             set_hp_filter = False, use_trend=True, use_const=True, use_harmonics=True,
                             exp_aprox = False, harmonic_orders=(1,2), use_dummies=True,
                             ridge_alpha=1e-6, cond_threshold=1e12, verbose=False, from_class = True, make_plot = True):
        if from_class:
            if var_name in self.var_names:
                series, starting_i = self.remove_na(var_name = var_name)
            else:
                print("Нет такой переменной в списке переменных")
                return None, None
        else:
            if series is not None and not series.empty:
                print("Для очистки от сезонности и трендов использую переданный мне временной ряд")
                series, starting_i = self.remove_na(series, from_class=False)
            else:
                print("Проверьте, что вы передали мне хотя бы что-то одно: или название переменной или временной ряд")
                return None, None
            
        if exp_aprox:
            y = np.log(np.asarray(series).astype(float).ravel())
        else:
            y = np.asarray(series).astype(float).ravel()

        if set_hp_filter:
            cycle, trend = sm.tsa.filters.hpfilter(y, lamb = (6.25*period/4)**4)
            best = {
                    'best_i': 0, 'best_r2': np.var(trend)/np.var(y), 'residuals': cycle,
                    'yhat': trend, 'A_hat': 6.25/period**4, 'Z_best': None, 'names': ['hp_filter'], 'diagnostics': None
            }
            if make_plot:
                plt.figure(figsize=(15, 8))
                plt.plot(self.df.index.to_list()[-series.shape[0]:], y, c = 'blue')
                plt.plot(self.df.index.to_list()[-series.shape[0]:], best['yhat'], c = 'g')
                plt.plot(self.df.index.to_list()[-series.shape[0]:], best['residuals'], c = 'r')
                plt.legend(['Временной ряд', 'Тренд', 'Циклическая компонента'])
                plt.axhline(y = 0, color = 'black', linestyle=':', linewidth=1)
                plt.title(var_name)
                plt.show()
            return best, starting_i

        n = y.shape[0]
        t = np.arange(n)
        best = {'best_i': None, 'best_r2': -np.inf}
        diagnostics = []

        for i in range(period):
            cols = []
            names = []
            if use_trend: cols.append(t); names.append('t')
            if use_harmonics:
                for h in harmonic_orders:
                    cols.append(np.sin(2*np.pi*(h*t + i)/period)); names.append(f'sin_{h}')
                    cols.append(np.cos(2*np.pi*(h*t + i)/period)); names.append(f'cos_{h}')
            if use_const: cols.append(np.ones(n)); names.append('const')
            if use_dummies:
                for j in range(period-1):
                    cols.append((t % period == j).astype(float)); names.append(f'dummy_{j+1}')

            Z = np.vstack(cols) if len(cols) > 0 else np.empty((0, n))
            if Z.size == 0: cond = np.inf; eigmin = 0.0
            else:
                try:
                    s = np.linalg.svd(Z.T, compute_uv=False)
                    cond = s[0] / (s[-1] if s[-1] > 0 else 1e-30)
                    eigmin = s[-1]
                except np.linalg.LinAlgError:
                    cond = np.inf; eigmin = 0.0

            if Z.size == 0:
                beta = np.zeros((0,)); yhat = np.zeros_like(y)
            else:
                try:
                    beta, residuals, rank, svals = np.linalg.lstsq(Z.T, y, rcond=None)
                except Exception:
                    beta = np.linalg.pinv(Z.T) @ y
                if cond > cond_threshold:
                    if verbose: print(f"i={i}: high condition {cond:.2e} -> applying ridge alpha={ridge_alpha}")
                    K = Z @ Z.T; kdim = K.shape[0]
                    K_reg = K + ridge_alpha * np.eye(kdim)
                    beta = np.linalg.solve(K_reg, Z @ y)
                yhat = (beta @ Z).ravel()

            error = y - yhat
            var_y = np.var(y); var_err = np.var(error)
            if var_y > 0:
                r2 = 1.0 - (var_err / var_y)
                diagnostics.append({'i': i, 'cond': cond, 'eigmin': eigmin, 'r2': r2, 'k': Z.shape[0]})
                if r2 > best['best_r2']:
                    best.update({
                        'best_i': i, 'best_r2': r2, 'residuals': np.exp(error) if exp_aprox else error,
                        'yhat': yhat, 'A_hat': beta, 'Z_best': Z.copy(), 'names': names, 'diagnostics': diagnostics.copy()
                    })
            else:
                print("Ряд состоит из одной константы, возвращаю его как он есть")
                best.update({'best_i': 0, 'best_r2': 1, 'residuals': series, 'yhat': series,
                             'A_hat': np.array([series.iloc[0].item() if nm == 'const' else 0 for nm in names]),
                             'Z_best': Z.copy(), 'names': names, 'diagnostics': diagnostics.copy()})
                break     

        if make_plot:
            plt.figure(figsize=(15, 8))
            plt.plot(self.df.index.to_list()[-series.shape[0]:], y, c = 'blue')
            plt.plot(self.df.index.to_list()[-series.shape[0]:], best['residuals'], c = 'r')
            plt.legend(['Временной ряд', 'Чистый ВР']); plt.axhline(y = 0, color = 'black', linestyle=':', linewidth=1)
            plt.title(var_name); plt.show()
        return best, starting_i

    def flatten(self, xss):
        return [x for xs in xss for x in xs]
    
    # === восстановление регрессоров по индексу t ===
    def _reconstruct_Z(self, t, params):
        """Восстанавливает матрицу регрессоров Z для заданных индексов времени t."""
        p = params['Z_params']
        t = np.asarray(t, dtype=float)
        cols = []
        if p['use_trend']: cols.append(t)
        if p['use_harmonics']:
            for h in p['harmonic_orders']:
                cols.append(np.sin(2*np.pi*(h*t + p['best_i'])/p['period']))
                cols.append(np.cos(2*np.pi*(h*t + p['best_i'])/p['period']))
        if p['use_const']: cols.append(np.ones_like(t))
        if p['use_dummies']:
            for j in range(p['period']-1):
                cols.append((t % p['period'] == j).astype(float))
        return np.vstack(cols) if cols else np.empty((0, t.size))

    def transform_to_clean(self, value, var_name, t_index):
        """Переводит реальное значение (уровень/темп) в очищенное пространство."""
        params = self.clean_params.get(var_name)
        if params is None:
            raise ValueError(f"Параметры очистки для {var_name} не найдены. Сначала запустите search_for_stationarity.")

        v = np.asarray(value, dtype=float)
        t = np.asarray(t_index, dtype=int)

        if params['exp_aprox']: v = np.log(v)

        if params['set_hp_filter']:
            trend_at_t = params['hp_trend'][t] if hasattr(params['hp_trend'], '__len__') else params['hp_trend']
            v_clean = v - trend_at_t
        else:
            Z_t = self._reconstruct_Z(t, params)
            if Z_t.shape[0] > 0:
                v_clean = v - (params['beta'] @ Z_t)
            else:
                v_clean = v

        if params['integration'] > 0:
            print(f"⚠️ {var_name}: применено дифференцирование d={params['integration']}. Преобразование уровня без опорных точек ряда носит приблизительный характер.")

        return v_clean if v_clean.size > 1 else v_clean.item()

    def transform_to_real(self, clean_value, var_name, t_index):
        """Переводит очищенное значение обратно в реальный мир."""
        params = self.clean_params.get(var_name)
        if params is None:
            raise ValueError(f"Параметры очистки для {var_name} не найдены.")

        v = np.asarray(clean_value, dtype=float)
        t = np.asarray(t_index, dtype=int)

        if params['integration'] > 0:
            print(f"⚠️ {var_name}: обратное преобразование при d={params['integration']} требует начальных значений ряда для точного восстановления.")

        if params['set_hp_filter']:
            trend_at_t = params['hp_trend'][t] if hasattr(params['hp_trend'], '__len__') else params['hp_trend']
            v_real = v + trend_at_t
        else:
            Z_t = self._reconstruct_Z(t, params)
            if Z_t.shape[0] > 0:
                v_real = v + (params['beta'] @ Z_t)
            else:
                v_real = v

        if params['exp_aprox']: v_real = np.exp(v_real)
        return v_real if v_real.size > 1 else v_real.item()
    # ==========================================

    def search_for_stationarity(self, series : pd.Series = None, verbose = False, var_name : str = None, 
                               only_KPSS : bool = True, from_class = True, **kwargs) -> Tuple[pd.Series, int]:
        deseason_and_detrend_dict = self.call_with_kwargs(self.deseason_and_detrend, **kwargs)
        if from_class:
            if var_name in self.var_names:
                result, starting_i = self.deseason_and_detrend(var_name = var_name, **deseason_and_detrend_dict)
            else:
                print("Нет такой переменной в списке переменных")
                return None
        else:
            if series:
                print("Для очистки от сезонности и трендов использую переданный мне временной ряд")
                result, starting_i = self.deseason_and_detrend(series, from_class=False, **deseason_and_detrend_dict)
            else:
                print("Проверьте, что вы передали мне хотя бы что-то одно: или название переменной или временной ряд")
                return None
                
        clean_series = result['residuals'].copy()
        x_axis = self.df.index.to_list()[-clean_series.shape[0]:]
        
        if only_KPSS:
            kpss_result = kpss(clean_series, regression='c', nlags="auto")
            criteria = kpss_result[1] < .05
            if verbose:
                print(f"{'='*50}\nВременной ряд {var_name}\n")
                print("KPSS Test:"); print(f"  p-value = {kpss_result[1]:.4f}")

            integration = 0
            while criteria and integration < 1:
                integration += 1
                clean_series_n = np.diff(clean_series) # Исправлено: вычисление до использования
                kpss_result = kpss(clean_series_n, regression='c', nlags="auto")
                criteria = kpss_result[1] < .05
                if verbose: print(f"  KPSS p-value (diff {integration}) = {kpss_result[1]:.4f}")
                clean_series = clean_series_n.copy()
        else:
            adf_result = adfuller(clean_series, autolag='AIC')
            kpss_result = kpss(clean_series, regression='c', nlags="auto")
            criteria = (adf_result[1] > .05) or (kpss_result[1] < .05)
            integration = 0
            while criteria and integration < 1:
                integration += 1
                clean_series_n = np.diff(clean_series)
                adf_result = adfuller(clean_series_n, autolag='AIC')
                kpss_result = kpss(clean_series_n, regression='c', nlags="auto")
                criteria = (adf_result[1] > .05) or (kpss_result[1] < .05)
                clean_series = clean_series_n.copy()

        if integration == 0:
            print(f"Исходный ряд {var_name} оказался стационарным")

        # === СОХРАНЯЕМ КОЭФФИЦИЕНТЫ ОЧИСТКИ ===
        self.clean_params[var_name] = {
            'exp_aprox': kwargs.get('exp_aprox', False),
            'integration': integration,
            'set_hp_filter': kwargs.get('set_hp_filter', False),
            'hp_trend': result.get('yhat') if kwargs.get('set_hp_filter', False) else None,
            'beta': result['A_hat'],
            'Z_params': {
                'period': kwargs.get('period', 12),
                'best_i': result['best_i'],
                'use_trend': kwargs.get('use_trend', True),
                'use_harmonics': kwargs.get('use_harmonics', True),
                'use_const': kwargs.get('use_const', True),
                'use_dummies': kwargs.get('use_dummies', True),
                'harmonic_orders': kwargs.get('harmonic_orders', (1, 2))
            },
            'series_length': len(clean_series),
            'starting_i': starting_i
        }
        # ======================================
        return clean_series, integration, starting_i
    
    def LOS(self, period: int = 12, verbose: bool = False, lag_max: int = 10, cond_threshold=1e12, 
            ridge_alpha=1e-6, hp_filter_map: dict = None, **kwargs):
        list_of_series = []
        list_of_start_i = []
        list_of_stop_i = []

        for var_name in self.var_names:
            local_kwargs = kwargs.copy()
            if hp_filter_map and var_name in hp_filter_map:
                local_kwargs["set_hp_filter"] = hp_filter_map[var_name]

            error, integration, starting_i = self.search_for_stationarity(var_name=var_name, **local_kwargs)
            list_of_start_i.append(starting_i[0] + integration)
            list_of_stop_i.append(starting_i[1])
            list_of_series.append(error)

        data = []
        max_start = max(list_of_start_i)
        list_of_start_i = [max_start-i for i in list_of_start_i]
        min_stop = min(list_of_stop_i)
        list_of_stop_i = [min_stop-i for i in list_of_stop_i]
        for i in range(len(list_of_series)):
            series = list_of_series[i]
            if list_of_start_i[i] != 0: series = series[list_of_start_i[i]:]
            if list_of_stop_i[i] != 0: series = series[:list_of_stop_i[i]]
            data.append(series)
        try:
            df = pd.DataFrame(data).set_axis(self.var_names, axis=0).set_axis(self.df.index[max_start:min_stop+1], axis = 1)
            X = df.values
            results = []
            for lag in range(1, lag_max):
                Z, Y, A_hat, E, K, T = self.OLS_estimation(Nseries=X, lag = lag, from_class=False)
                u = (E @ E.T)
                aic = np.log(np.linalg.det(u)) + 2 / T * (lag * K**2 + K)
                hqc = np.log(np.linalg.det(u)) + 2 * np.log(np.log(T)) / T * (lag * K**2 + K)
                sic = np.log(np.linalg.det(u)) + np.log(T) / T * (lag * K**2 + K)
                p = period
                El = [np.hstack([np.zeros((K, i)), E[:, :-i]]) for i in range(1, p + 1)]
                El = np.vstack(El)
                Z_aux = np.vstack([Z, El])
                try:
                    s = np.linalg.svd(Z_aux.T, compute_uv=False)
                    cond = s[0] / (s[-1] if s[-1] > 0 else 1e-30)
                except np.linalg.LinAlgError:
                    cond = np.inf

                if cond > cond_threshold:
                    if verbose: print(f"i={i}: high condition {cond:.2e} -> applying ridge alpha={ridge_alpha}")
                    qform_Z_aux = Z_aux @ Z_aux.T + ridge_alpha * np.eye(Z_aux.shape[0])
                    B_aux = E @ Z_aux.T @ np.linalg.inv(qform_Z_aux)
                else:
                    B_aux = E @ Z_aux.T @ np.linalg.inv(Z_aux @ Z_aux.T)
                E_aux = E - B_aux @ Z_aux
                e = (E_aux @ E_aux.T)
                Q_LM = (T-p)*(1 - np.linalg.trace(e)/np.linalg.trace(u))
                p_val = 1 - chi2.cdf(Q_LM, p * K**2)
                results.append({"lag": lag, "AIC": aic, "HQC": hqc, "SIC": sic, "LM p-value": p_val})

            return pd.DataFrame(results), df
        except:
            print("Что-то не получилось, возвращаю вам хотя бы очищенные данные")
            return (max_start, min_stop), df