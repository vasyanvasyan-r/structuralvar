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

    The implementation is based on methods from Lutkepohl and Kilian (2017)
    (Kilian, L., & Lütkepohl, H. (2017). *Structural Vector Autoregressive Analysis*.
    Cambridge University Press), as well as the earlier academic textbook:
    Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. 
    Springer.

    The main requirement for using the package is to pass time series data
    in a format the model can interpret. There are two supported options:

    1. A NumPy `ndarray` and a list of variable names.
    In this case, the ndarray should be KxT, hence the rows are the variables and the columns are lags 
    2. A `pandas.DataFrame` and a list of column names.  
    In this case, the DataFrame’s index must reflect the time structure,
    meaning that sorting it in ascending order should yield the proper
    chronological order from the first to the last observation.

    """
    doc_ru = """
    
    Пакет для подбора корректной модели векторной авторегрессии и оценки импульсных откликов, сделанный К и К
    За основу были взяты методы из Lutkepohl и Killian 2017 
    (Kilian, L., & Lütkepohl, H. (2017). Structural Vector Autoregressive Analysis. Cambridge University Press) 
    и более ранний академический учебник Lutkepohl H. 
    (Helmut Lütkepohl, 2005. "New Introduction to Multiple Time Series Analysis," Springer Books, Springer, number 978-3-540-27752-1, December.)

    Основным требованием для пакета, является передача ему временных рядов, так чтобы он понял. 
    Есть два варианта: 
    1. numpy ndarray и список имен переменных
    2. передать pandas DataFrame и список столбцов-переменных; обязательное условие, 
    что индексы в таблицы должны быть отражать временную структуру так,
    что сортировка по возрастанию по ним давала от первого наблюдения до последнего

    Если будет переданно, 
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
            self.var_names = [f'var_{i}' for i in range(1, self.K + 1)] if self.df.columns.to_list() == [] else self.df.columns.to_list()

        elif isinstance(timeseries, np.ndarray):
            print("Это NumPy массив!")
            self.var_names = [f'var_{i}' for i in range(1, self.K + 1)] if list_with_names == [] else list_with_names
        else:
            raise ValueError("timeseries должен быть либо DataFrame, либо ndarray")
        
        self.X: np.ndarray = timeseries
        self.p_max: int = p_max # type: ignore
        self.i_criteria: Optional[str] = i_criteria
        self.K, self.T = timeseries.shape     

    def call_with_kwargs(self, 
                        func,
                        **kwargs):
        # получаем сигнатуру функции
        sig = inspect.signature(func)
        # оставляем только те параметры, что есть у func
        filtered = {
            k: v for k, v in kwargs.items()
            if k in sig.parameters
        }
        return filtered

    def OLS_estimation(self,
                       Nseries,
                       lag,
                       from_class : bool = True,
                       add_const : bool = True) -> Tuple[
                                                         np.ndarray, # Z
                                                         np.ndarray, # Y
                                                         np.ndarray, # B_hat
                                                         np.ndarray, # E
                                                         int, int # K and T
                                                         ]:
        """
        Performs OLS estimation for the VAR(p) model.

        Returns
        -------
        Z : np.ndarray
            Regressor matrix with constant and lags.
        Y : np.ndarray
            Matrix of dependent variables.
        B_hat : np.ndarray
            Estimated coefficient matrix.
        """
        if from_class:

            X = self.X
            p = self.p_max
            T = self.T
        else:
            X = Nseries
            p = lag
            T = X.shape[1]


        Z = np.concatenate(
            [np.concatenate([np.array([1]) if add_const else None, X[:, i:i+p].T.flatten()]).reshape(-1, 1) # type: ignore
             for i in range(T - p)],
            axis=1
        )[:, 1:]

        Y = X[:, :Z.shape[1]]

        B_hat = Y @ Z.T @ np.linalg.inv(Z @ Z.T)

        return Z, Y, B_hat, Y - B_hat @ Z, X.shape[0], T
    
    def remove_na(self,
                  series : pd.Series = None, # type: ignore
                  var_name : int = None, # type: ignore
                  from_class : bool = True):
        """
        Функция, которая возвращает ряд без пропусков, пропуски заполняет линейной
        апроксимацией, а если пропуск больше 2-х наблюдений подряд, то он вас об этом уведомит.
        Функция понимает, что если в начале ряда (прямо с первого значения) идут пропуски, то их 
        она заполнять не будет, а вместо этого вернет ряд без них и индекс,
        на котором они заканчиваются. Функция всегда возвращает кортеж из ряда и индекса.
        """
        
        if from_class:
            #check wheather var_name in class
            if var_name in self.var_names: # type: ignore
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
        na_list = series.loc[series.isna()].index.to_list() # type: ignore
        not_na_list = series.loc[series.notna()].index.to_list()


        if len(not_na_list) > 1:
            #найдем дырки без значений

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

        return series, series.iloc[[0,-1]].index.to_list() # type: ignore
    
    def deseason_and_detrend(self,
                            series: pd.Series = None,# type: ignore
                            var_name: str = None, # type: ignore
                            period: int = 12,
                            set_hp_filter = False,
                            use_trend=True,
                            use_const=True,
                            use_harmonics=True,
                            exp_aprox = False,
                            harmonic_orders=(1,2),
                            use_dummies=True,
                            ridge_alpha=1e-6,
                            cond_threshold=1e12,
                            verbose=False,
                            from_class = True,
                            make_plot = True):
        """
        series: pd.Series or 1d-array (n,)
        возвращает dict с лучшими результатами по перебору фаз i=0..period-1:
        {
            'best_i': int,
            'best_r2': float,
            'residuals': np.array(n,),
            'yhat': np.array(n,),
            'B_hat': np.array((k,)),
            'Z_best': np.array((k,n)),
            'diagnostics': {...}
        }
        Параметры:
        use_dummies: если True — добавляет period-1 дамми (предполагается, что константа включена).
        harmonic_orders: кортеж порядков гармоник (1,2) по умолчанию.
        """

        if from_class:
            #check wheather var_name in class
            if var_name in self.var_names: # type: ignore
                series, starting_i = self.remove_na(var_name = var_name) # type: ignore
            else:
                print("Нет такой переменной в списке переменных")
                return None, None
        else:
            if series is not None and not series.empty:
                print("Для очистки от сезонности и трендов использую переданный мне временной ряд")
                series, starting_i = self.remove_na(series, from_class=False) # type: ignore
            else:
                print("Проверьте, что вы передали мне хотя бы что-то одно: или название переменной или временной ряд")
                return None, None
            
        if exp_aprox:
            y = np.log(np.asarray(series).astype(float).ravel())
        else:
            y = np.asarray(series).astype(float).ravel()

        if set_hp_filter:
            cycle, trend = sm.tsa.filters.hpfilter(y, lamb = (6.25*period/4)**4) # type: ignore
            
            best = {
                    'best_i': 0,
                    'best_r2': np.var(trend)/np.var(y),
                    'residuals': cycle,
                    'yhat': trend,
                    'B_hat': 6.25/period**4,
                    'Z_best': None,
                    'names': ['hp_filter'],
                    'diagnostics': None
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

            if use_trend:
                cols.append(t)
                names.append('t')

            if use_harmonics:
                for h in harmonic_orders:
                    cols.append(np.sin(2*np.pi*(h*t + i)/period))
                    names.append(f'sin_{h}')
                    cols.append(np.cos(2*np.pi*(h*t + i)/period))
                    names.append(f'cos_{h}')

            if use_const:
                cols.append(np.ones(n))
                names.append('const')

            if use_dummies:
                # создаём period-1 дамми (при условии, что const=True)
                for j in range(period-1):
                    cols.append((t % period == j).astype(float))
                    names.append(f'dummy_{j+1}')

            # Z shape: (k, n) to match your original orientation; but for lstsq we'll use Z.T (n,k)
            Z = np.vstack(cols) if len(cols) > 0 else np.empty((0, n))

            # Diagnostics: condition number of Z (or Z @ Z.T)
            # compute condition number of design matrix Z.T (n x k) via SVD:
            if Z.size == 0:
                # degenerate
                cond = np.inf
                eigmin = 0.0
            else:
                # use SVD on Z (k x n) -> but more natural on Z.T (n x k)
                # condition number of Z.T:
                try:
                    s = np.linalg.svd(Z.T, compute_uv=False)
                    cond = s[0] / (s[-1] if s[-1] > 0 else 1e-30)
                    eigmin = s[-1]
                except np.linalg.LinAlgError:
                    cond = np.inf
                    eigmin = 0.0

            # Solve least squares (more stable than direct inversion)
            # Z.T @ beta = y  with shape (n,k) @ (k,) = (n,)
            # we want beta shape (k,)
            if Z.size == 0:
                beta = np.zeros((0,))
                yhat = np.zeros_like(y)
            else:
                # use np.linalg.lstsq on Z.T (n,k)
                try:
                    beta, residuals, rank, svals = np.linalg.lstsq(Z.T, y, rcond=None)
                except Exception:
                    # fallback to pseudo-inverse
                    beta = np.linalg.pinv(Z.T) @ y

                # if design is extremely ill-conditioned, try ridge via normal equations (regularized)
                if cond > cond_threshold:
                    if verbose:
                        print(f"i={i}: high condition {cond:.2e} -> applying ridge alpha={ridge_alpha}")
                    # normal eqns with ridge on (k x k) matrix
                    K = Z @ Z.T  # (k,k)
                    kdim = K.shape[0]
                    K_reg = K + ridge_alpha * np.eye(kdim)
                    # compute B = Y Z.T (use shapes consistent)
                    # we have y (n,) and Z (k,n): want B (1,k) -> beta = (Z @ Z.T)^{-1} @ Z @ y
                    beta = np.linalg.solve(K_reg, Z @ y)
                yhat = (beta @ Z).ravel()

            error = y - yhat
            # r2 computed same as ты: 1 - var(error)/var(y)
            # use unbiased variance? keep population var (ddof=0) similar to np.var default
            var_y = np.var(y)
            var_err = np.var(error)
            if var_y > 0:
                r2 = 1.0 - (var_err / var_y)

                diagnostics.append({'i': i, 'cond': cond, 'eigmin': eigmin, 'r2': r2, 'k': Z.shape[0]})

                if r2 > best['best_r2']:
                    best.update({
                        'best_i': i,
                        'best_r2': r2,
                        'residuals': np.exp(error) if exp_aprox else error,
                        'yhat': yhat,
                        'B_hat': beta,
                        'Z_best': Z.copy(),
                        'names': names,
                        'diagnostics': diagnostics.copy()
                    })

                if verbose:
                    print("Top diagnostics (first 5):", diagnostics[:5])
            else:
                print("Ряд состоит из одной константы, возвращаю его как он есть")
                
                best.update({
                    'best_i': 0,
                    'best_r2': 1,
                    'residuals': series,
                    'yhat': series,
                    'B_hat': np.array([series.iloc[0].item() if n == 'const' else 0 for n in names]),
                    'Z_best': Z.copy(),
                    'names': names,
                    'diagnostics': diagnostics.copy()
                })
                break     
        if make_plot:
            plt.figure(figsize=(15, 8))
            plt.plot(self.df.index.to_list()[-series.shape[0]:], y, c = 'blue')
            plt.plot(self.df.index.to_list()[-series.shape[0]:], best['residuals'], c = 'r')
            plt.legend(['Временной ряд', 'Чистый ВР'])
            plt.axhline(y = 0, color = 'black', linestyle=':', linewidth=1)
            plt.title(var_name)
            plt.show()
        return best, starting_i

# function to flatten a list of list
    def flatten(self,
                xss):
        return [x for xs in xss for x in xs]
    
    def search_for_stationarity(self,
                               series : pd.Series = None, # type: ignore
                               verbose = False,
                               var_name : str = None, # type: ignore
                               only_KPSS : bool = True,
                               from_class = True,
                               **kwargs
                            ) -> Tuple[pd.Series, int]:
        """
        Функция выбирает по очереди порядки интеграции, пока не получит
        стационарные ряды
        Возвращает ряд и порядок интеграции 
        """

        # аргументы к функции deseason_and_detrend
        deseason_and_detrend_dict = self.call_with_kwargs(
            self.deseason_and_detrend,
            **kwargs
        )
        if from_class:
            #check wheather var_name in class
            if var_name in self.var_names: # type: ignore
                result, starting_i = self.deseason_and_detrend(var_name = var_name, **deseason_and_detrend_dict) # type: ignore
            
            else:
                print("Нет такой переменной в списке переменных")
                return None # type: ignore
        else:
            if series: # type: ignore
                print("Для очистки от сезонности и трендов использую переданный мне временной ряд")
                result, starting_i = self.deseason_and_detrend(series, from_class=False, **deseason_and_detrend_dict) # type: ignore

            else:
                print("Проверьте, что вы передали мне хотя бы что-то одно: или название переменной или временной ряд")
                return None # type: ignore
        clean_series = result['residuals'].copy() # type: ignore

        x_axis = self.df.index.to_list()[-clean_series.shape[0]:]
        if only_KPSS:
            kpss_result = kpss(clean_series, regression='c', nlags="auto")
            criteria = kpss_result[1] < .05
            if verbose:
                print(f"{'='*50}\nВременной ряд {var_name}\n")
                print("KPSS Test:")
                print(f"  KPSS statistic = {kpss_result[0]:.4f}")
                print(f"  p-value = {kpss_result[1]:.4f}")
                print("  Critical values:", kpss_result[3])

            integration = 0

            while criteria and integration < 1:
                integration += 1
                clean_series = np.diff(clean_series)
                adf_result = adfuller(clean_series, autolag='AIC')
                kpss_result = kpss(clean_series, regression='c', nlags="auto")

                criteria = kpss_result[1] < .05

                if verbose:
                    print(f"{'='*50}\nВременной ряд {var_name}\n")
                    print("KPSS Test:")
                    print(f"  KPSS statistic = {kpss_result[0]:.4f}")
                    print(f"  p-value = {kpss_result[1]:.4f}")
                    print("  Critical values:", kpss_result[3])
                plt.figure(figsize=(15, 8))
                plt.plot(x_axis[integration:], clean_series[1:], c = 'blue')
                plt.plot(x_axis[integration:], clean_series_n, c = 'r') # type: ignore
                plt.legend(['Исходные ряд', f'Интегрированный {integration} раз'])
                plt.axhline(y = 0, color = 'black', linestyle=':', linewidth=1)

                plt.title(f"{var_name} уровень интеграции {integration}")
                plt.show()

                clean_series = clean_series_n.copy() # type: ignore
        else:
            adf_result = adfuller(clean_series, autolag='AIC')
            kpss_result = kpss(clean_series, regression='c', nlags="auto")
            criteria = (adf_result[1] > .05) or (kpss_result[1] < .05)
            if verbose:
                print(f"{'='*50}\nВременной ряд {var_name}\n")
                print("ADF Test:")
                print(f"  ADF statistic = {adf_result[0]:.4f}")
                print(f"  p-value = {adf_result[1]:.4f}")
                print("  Critical values:", adf_result[4]) # type: ignore
                print("-"*40)
                print("KPSS Test:")
                print(f"  KPSS statistic = {kpss_result[0]:.4f}")
                print(f"  p-value = {kpss_result[1]:.4f}")
                print("  Critical values:", kpss_result[3])

            integration = 0
            while criteria and integration < 1:
                integration += 1
                clean_series_n = np.diff(clean_series)
                adf_result = adfuller(clean_series, autolag='AIC')
                kpss_result = kpss(clean_series, regression='c', nlags="auto")

                criteria = (adf_result[1] > .05) or (kpss_result[1] < .05)

                if verbose:
                    print(f"{'='*50}\nВременной ряд {var_name}\n")
                    print("ADF Test:")
                    print(f"  ADF statistic = {adf_result[0]:.4f}")
                    print(f"  p-value = {adf_result[1]:.4f}")
                    print("  Critical values:", adf_result[4]) # type: ignore
                    print("-"*40)
                    print("KPSS Test:")
                    print(f"  KPSS statistic = {kpss_result[0]:.4f}")
                    print(f"  p-value = {kpss_result[1]:.4f}")
                    print("  Critical values:", kpss_result[3])
                plt.figure(figsize=(15, 8))
                plt.plot(x_axis[integration:], clean_series[1:], c = 'blue')
                plt.plot(x_axis[integration:], clean_series_n, c = 'r')
                plt.legend(['Исходные ряд', f'Интегрированный {integration} раз'])
                plt.axhline(y = 0, color = 'black', linestyle=':', linewidth=1)
                plt.title(f"{var_name} уровень интеграции {integration}")
                plt.show()

                clean_series = clean_series_n.copy()
        if integration == 0:
            print(f"Исходный ряд {var_name} оказался стационарным")

        return clean_series, integration, starting_i # type: ignore
    
    def LOS(self,
            period: int = 12,
            verbose: bool = False,
            lag_max: int = 10,
            cond_threshold=1e12,
            ridge_alpha=1e-6,
            hp_filter_map: dict = None,   # <-- словарь {var_name: True/False} # type: ignore
            **kwargs 
            ):
        list_of_series = []
        list_of_start_i = []
        list_of_stop_i = []

        for var_name in self.var_names: # type: ignore
            # определяем, использовать ли hp фильтр для этой переменной
            local_kwargs = kwargs.copy()
            if hp_filter_map and var_name in hp_filter_map:
                local_kwargs["set_hp_filter"] = hp_filter_map[var_name]

            error, integration, starting_i = self.search_for_stationarity( # type: ignore
                var_name=var_name,
                **local_kwargs
            )
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
            if list_of_start_i[i] != 0:
                series = series[list_of_start_i[i]:]
            if list_of_stop_i[i] != 0:
                series = series[:list_of_stop_i[i]]
            data.append(series)
        try:
            # Готовим дата фрейм. 
            df = pd.DataFrame(data).set_axis(self.var_names, axis=0).set_axis(self.df.index[max_start:min_stop+1], axis = 1)
            X = df.values
            results = []
            data = []
            for lag in range(1, lag_max):


                Z, Y, B_hat, E, K, T = self.OLS_estimation(Nseries=X, lag = lag, from_class=False)

                u = (E @ E.T)

                aic = np.log(np.linalg.det(u)) + 2 / T * (lag * K**2 + K)


                hqc = np.log(np.linalg.det(u)) + 2 * np.log(np.log(T)) / T * (lag * K**2 + K)


                sic = np.log(np.linalg.det(u)) + np.log(T) / T * (lag * K**2 + K)

                p = period

                El = []
                for i in range(1, p + 1): # создаём лаги остатков до p-го лага, хотя если хотим выявить до h-го лага, то меняем 
                    lagdata = np.hstack([np.zeros((K, i)), E[:, :-i]])
                    El.append(lagdata)
                El = np.vstack(El)
                Z_aux = np.vstack([Z, El])

                # Проверяем насколько Z_aux близка к вырожденной

                try:
                    s = np.linalg.svd(Z_aux.T, compute_uv=False)
                    cond = s[0] / (s[-1] if s[-1] > 0 else 1e-30)
                    eigmin = s[-1]
                except np.linalg.LinAlgError:
                    cond = np.inf
                    eigmin = 0.0

                # если среди собственных значений есть ооочень маленькие, то немного сдвигаем матрицу для получения болле эффективных оценок
                if cond > cond_threshold:
                    if verbose:
                        print(f"i={i}: high condition {cond:.2e} -> applying ridge alpha={ridge_alpha}")
                    # normal eqns with ridge on (k x k) matrix
                    qform_Z_aux = Z_aux @ Z_aux.T  # (k,k)
                    kdim = qform_Z_aux.shape[0]
                    qform_Z_aux = qform_Z_aux.copy() + ridge_alpha * np.eye(kdim)

                    B_aux = E @ Z_aux.T @ np.linalg.inv(qform_Z_aux)
                else:
                    B_aux = E @ Z_aux.T @ np.linalg.inv(Z_aux @ Z_aux.T) # говорит, что если будет просто np.linalg.inv(Z_aux @ Z_aux.T), то матрица будет вырожденная. Поэтому pinv
                E_aux = E - B_aux @ Z_aux
                e = (E_aux @ E_aux.T)

                Q_LM = (T-p)*(1 - np.linalg.trace(e)/np.linalg.trace(u))
                dof = p * K**2
                p_val = 1 - chi2.cdf(Q_LM, dof)

                results.append({"lag": lag,
                "AIC": aic,
                "HQC": hqc,
                "SIC": sic,
                "LM p-value": p_val,
                })
                data.append([Z, Y])

            print_df = pd.DataFrame(results)
        

            return print_df, df
        except:
            print("Что-то не получилось, возвращаю вам хотя бы очищенные данные")
            return data, (max_start, min_stop)  # type: ignore

    



    
