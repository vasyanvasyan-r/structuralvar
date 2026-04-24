import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from typing import Optional, Sequence, Tuple
import datetime as dt
import warnings

plt.rcParams['font.family'] = 'Times New Roman'


def plot_irf_single(
    irfs,
    i,
    j,
    horizon,
    u_dict,
    y_dict,
    plot_simulations=False,
    ci_levels=(68, 90),
    figsize=(9, 5),
    ci_color = 'cadetblue',
    cumm = False,
    sign_draws = False
):

    irfs = np.asarray(irfs)
    x_axis = np.arange(horizon + 1)
    h_max = irfs.shape[1]
    if horizon + 1> h_max:
        print(f"Слишком большой задан горизонт, у меня нет столько периодов в симмуляциях, максимальный доступный горизонт {h_max}")
        print(f"Новый горизонт {h_max}")
        horizon = h_max.copy()
    # n_draws × (horizon+1)
    responses = irfs[:, :horizon+1, i, j]
    if cumm:
        for h in range(1, horizon+1):
            responses[:, h] += responses[:, h-1]
    # original irf
    original_irf = responses[0, :]

    # медианный отклик (L2-closest)
    med = np.median(responses, axis=0)
    median_index = (
        pd.DataFrame((responses - med) ** 2)
        .sum(axis=1)
        .sort_values()
        .index[0]
    )
    med_res = responses[median_index]

    # точечная медиана
    median = np.percentile(responses, 50, axis=0)

    # доверительные интервалы
    ci = {}
    for level in ci_levels:
        alpha = (100 - level) / 2
        ci[level] = (
            np.percentile(responses, alpha, axis=0),
            np.percentile(responses, 100 - alpha, axis=0),
        )

    # ---------- симуляции ----------
    if plot_simulations:
        plt.figure(figsize=figsize)
        plt.plot(x_axis, responses.T, color="lightgray", alpha=0.15, lw=0.8)
        plt.plot(x_axis, median, color="grey", ls=":", label="Медиана")
        plt.plot(x_axis, med_res, color="grey", ls="--", label="Медианный отклик")
        plt.plot(x_axis, original_irf, color="black", label="Оригинальный отклик")
        plt.axhline(0, color="black", lw=1)
        plt.legend(loc="lower right")
        plt.title(f"{list(y_dict.values())[i]} на шок {list(u_dict.values())[j]} (все симуляции)")
        plt.show()

    # ---------- итоговый график ----------
    plt.figure(figsize=figsize)
    if sign_draws:
        plt.plot(x_axis, median, color="black", ls=":", label="Медиана")
    else:
        plt.plot(x_axis, original_irf, color="black", label="Отклик")
    plt.plot(x_axis, med_res, color="black", ls="--", label="Медианный отклик")

    for level, (lo, hi) in ci.items():
        alpha = 0.3 if level == min(ci_levels) else 0.15
        plt.fill_between(x_axis, lo, hi, alpha=alpha, label=f"{level}% CI", color = ci_color)

    plt.axhline(0, color="black", lw=1)
    plt.legend()
    plt.title(f"{list(y_dict.values())[i]} на шок {list(u_dict.values())[j]}", fontweight='bold' )
    plt.show()

def plot_hd_bars_signed(df_hd: pd.DataFrame,
                        K: int,
                        variable_name: Optional[str] = None, 
                        shocks_labels: Optional[list] = None, 
                        cumm: bool = True):
    """
    Функция построения графика накопленной исторической декомпозиции
    df_hd -- строго pandas DataFrame, по строкам идет так:
            / первая строка -- временной ряд
            / со второй по К+1 строку -- ряды структурных шоков из модели
            / с K+2 строки идут вклады начальных условий, константы и
              экзогенных переменных
    """
    length = df_hd.shape[1]
    if variable_name != 'drop':
        variable_name = str(df_hd.index[0])
    else:
        df_hd = df_hd.rename(index = {df_hd.index[0].item(): variable_name})
    shocks_labels = shocks_labels or [i for i in df_hd.index[1:K+1]]
    trash_labels = [i for i in df_hd.index[K+1:]]
    if not isinstance(df_hd.columns, pd.DatetimeIndex):
        raise ValueError('Time index is not pandas.DatetimeIndex')
    else:
        x = np.arange(df_hd.shape[1])
        x = x - 12 + df_hd.columns.to_list()[0].month - 1 # pyright: ignore[reportAttributeAccessIssue]

    plt.figure(figsize=(9, 6))

    # Основания для положительных и отрицательных значений
    bottom_pos = np.zeros(length)
    bottom_neg = np.zeros(length)

    # Используем палитру заранее, чтобы цвета не повторялись
    
    list_of_colors = [
            'powderblue',      # Голубой
            'moccasin',        # (было) Оранжево-бежевый
            'salmon',          # (было) Лососевый
            'lightgreen',      # (было) Светло-зеленый
            'plum',         # (было) Светло-фиолетовый
            'steelblue',   # (новое) Какашечный, светлый
            'palegoldenrod',   # (новое) Желтоватый
            'mistyrose',       # (новое) Нежно-розовый
            'burlywood',       # (новое) Песочный
            ]
    dict_of_grey_collors = {
        'Вклад экзопеременных':'lightgrey',
        'Вклад константы':'grey',
        'Вклад Y_0':'dimgray'
    }
    #colors
    color_of_variable = 'black'
    colors = list_of_colors[:len(shocks_labels)]
    trash_colors = [dict_of_grey_collors[i] for i in trash_labels]

    #unite shocks and trash labels
    shocks_labels += trash_labels
    colors += trash_colors

    for i, shock in enumerate(shocks_labels):
        values = df_hd.loc[shock].values
        pos = np.where(values > 0, values, 0)
        neg = np.where(values < 0, values, 0)

        color = colors[i]

        # Рисуем обе части одним цветом
        plt.bar(x, pos, bottom=bottom_pos, color=color, alpha=0.8)
        plt.bar(x, neg, bottom=bottom_neg, color=color, alpha=0.8, label=shock)

        # Обновляем основания
        bottom_pos += pos
        bottom_neg += neg
    plt.plot(x, df_hd.loc[variable_name], color=color_of_variable, linewidth=2, label=variable_name)
    plt.xticks(
        ticks=x[x % 12 == 0],
        labels=[i.year for i in df_hd.columns[x % 12 == 0]], # pyright: ignore[reportGeneralTypeIssues]
        rotation=45,
        ha="right"
    )

    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(f"{'Накопленная историческая декомпозиция' if cumm else 'Историческая декомпозиция'}: {variable_name}",
               fontsize = 11, fontweight='bold')
    plt.legend(fontsize=10)
    plt.xlabel("Год", fontsize = 11)
    plt.ylabel("Вклад шоков")
    plt.tight_layout()
    plt.show()


def plot_irf_grid(
    irf_sims,
    y_labels: Optional[Sequence[str]] = None,
    u_labels: Optional[Sequence[str]] = None,
    horizon: Optional[int] = None,
    figsize: Tuple[int, int] = (9, 12),
    title: str = "Функции импульсного отклика",
    ci_color: str = "cadetblue",
    ci_alpha_inner: float = 0.3,
    ci_alpha_outer: float = 0.15,
    median_color: str = "black",
    median_linestyle: str = "--",
    legend_bbox: tuple = (0.5, 0.01),
    legend_loc: str = "upper center",
    legend_ncol: int = 4,
    suptitle_y: float = 0.99,
    sign_draws: bool = False
):
    """
    Построить сетку IRF с общей легендой.

    Параметры
    ----------
    IRF_ortho : array-like
        Либо список массивов формы (horizon+1, K, K) для каждого draw,
        либо массив формы (n_draws, horizon+1, K, K).
    y_labels : sequence[str], optional
        Названия для откликов (строки). Если None, будут использованы индексы.
    u_labels : sequence[str], optional
        Названия для шоков (столбцы). Если None, будут использованы индексы.
    horizon : int, optional
        Горизонт (число периодов). Если None, берётся из формы данных.
    figsize : tuple, optional
        Размер фигуры (width, height).
    title : str, optional
        Заголовок (suptitle).
    ci_color : str, optional
        Цвет для интервалов доверия.
    ci_alpha_inner : float, optional
        Альфа для внутреннего (68%) CI.
    ci_alpha_outer : float, optional
        Альфа для внешнего (90%) CI.
    median_color : str, optional
        Цвет медианы.
    median_linestyle : str, optional
        Стиль линии для медианного конкретного draw (например '--').
    legend_bbox, legend_loc, legend_ncol : параметры легенды
    suptitle_y : float, optional
        Вертикальное положение suptitle.
    sign_draws : bool, optional
        Использует медианный отклик и медиану, иначе берется нулевая
        симуляция, в которой лежит оригинальный отклик

    Возвращает
    -------
    fig, axes : matplotlib.figure.Figure, np.ndarray
        Фигура и массив осей (K x K).
    """
    # Приводим к ndarray формы (n_draws, horizon+1, K, K)
    arr = np.asarray(irf_sims)
    if arr.ndim == 3:
        # один draw: (h+1, K, K) -> (1, h+1, K, K)
        arr = arr[None, ...]
        raise Warning("Find the only one draw, are you sure everything is fine, check the diminsions")
    elif arr.ndim < 3:
        raise ValueError("irf_sims должен быть list/ndarray размерности (h+1,K,K) или (n_draws,h+1,K,K)")

    n_draws, h_plus_1, K1, K2 = arr.shape
    if K1 != K2:
        raise ValueError("Последние два измерения должны быть равны (K x K)")

    K_endo = K1

    if horizon is None:
        horizon = h_plus_1 - 1
    elif horizon + 1 > h_plus_1:
        raise ValueError("Параметр horizon слишком большой для переданных IRF")

    # Метки по умолчанию
    if y_labels is None:
        y_labels = [f"{i}" for i in range(K_endo)]
    if u_labels is None:
        u_labels = [f"{j}" for j in range(K_endo)]
    if len(y_labels) != K_endo or len(u_labels) != K_endo:
        raise ValueError("Длины y_labels и u_labels должны соответствовать числу эндогенных переменных K")

    fig, axes = plt.subplots(K_endo, K_endo, figsize=figsize)
    fig.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)

    # Легенда (прокси-элементы)
    legend_elements = [
        Line2D([0], [0], color=median_color, label="Медиана" if sign_draws else 'Оригинальный отклик'),
        Line2D([0], [0], color=median_color, ls=median_linestyle, label="Медианный отклик"),
        Patch(facecolor=ci_color, alpha=ci_alpha_inner, label="68% ДИ"),
        Patch(facecolor=ci_color, alpha=ci_alpha_outer, label="90% ДИ"),
    ]

    x_axis = np.arange(horizon + 1) # type: ignore

    for i in range(K_endo):
        for j in range(K_endo):
            ax = axes[i, j] if K_endo > 1 else axes  # type: ignore # если K==1 axes может быть объектом, а не массивом

            # responses shape: (n_draws, horizon+1)
            responses = arr[:, : (horizon + 1), i, j] # type: ignore

            # медиана по каждому горизонту
            median = np.median(responses, axis=0)

            # original irf
            original_irf = responses[0,:]

            # находим draw, который наиболее близок к медиане (по сумме квадратов отклонений)
            ssd = np.sum((responses - median) ** 2, axis=1)
            median_index = int(np.argmin(ssd))
            med_res = responses[median_index, :]

            # доверительные интервалы
            lower68 = np.percentile(responses, 16, axis=0)
            upper68 = np.percentile(responses, 84, axis=0)
            lower90 = np.percentile(responses, 5, axis=0)
            upper90 = np.percentile(responses, 95, axis=0)

            # отрисовка
            if sign_draws:
                ax.plot(x_axis, median, color=median_color)
            else:
                ax.plot(x_axis, original_irf, color=median_color)
            ax.plot(x_axis, med_res, color=median_color, ls=median_linestyle)
            ax.fill_between(x_axis, lower68, upper68, color=ci_color, alpha=ci_alpha_inner)
            ax.fill_between(x_axis, lower90, upper90, color=ci_color, alpha=ci_alpha_outer)
            ax.axhline(0, color="firebrick", lw=2)

            ax.set_title(f"Отклик: {y_labels[i]}\nШок: {u_labels[j]}", fontsize=9)
            ax.grid(True, alpha=0.3)

    # Общая легенда
    fig.legend(handles=legend_elements, loc=legend_loc, bbox_to_anchor=legend_bbox, ncol=legend_ncol, fontsize=12)
    fig.suptitle( title + f' ({horizon} периодов)', fontsize=18, y=suptitle_y, fontweight='bold')
    plt.tight_layout()
    if horizon < 13:  # pyright: ignore[reportOptionalOperand]

        plt.xticks(
            ticks = x_axis,
            labels = [str(i) for i in x_axis],
            ha="right"
        )
    plt.show()

    return fig, axes

def plot_cf_policy_space_single(
    cf_results: dict,
    time_index: pd.Index,
    ci_levels: Tuple[int, int] = (68, 90),
    figsize: Tuple[int, int] = (12, 6),
    colors: dict = None,
    policy_shock_labels: Optional[list] = None,
    n_simulations_plot: int = 10,
    suptitle: str = "Контрфактические симуляции шоков политики",
    savefig: bool = False,
    savepath: Optional[str] = None,
    dpi: int = 300,
) -> dict:
    """
    Визуализация результатов cf_policy_space — КАЖДАЯ ПЕРЕМЕННАЯ В ОТДЕЛЬНОМ ГРАФИКЕ.
    
    Parameters
    ----------
    cf_results : dict
        Результаты из метода cf_policy_space (ключи -- имена переменных)
    time_index : pd.Index
        Временная ось для графиков (должна совпадать с длиной рядов)
    ci_levels : tuple
        Уровни доверительных интервалов в процентах (по умолчанию 68% и 90%)
    figsize : tuple
        Размер фигуры (width, height) для каждого графика
    colors : dict, optional
        Словарь цветов для элементов графика
    policy_shock_labels : list, optional
        Названия шоков политики для легенды
    n_simulations_plot : int
        Сколько случайных траекторий симуляций показать (для наглядности)
    suptitle : str
        Общий префикс для заголовков всех графиков
    savefig : bool
        Сохранять ли графики в файлы
    savepath : str, optional
        Путь для сохранения (без расширения)
    dpi : int
        Разрешение для сохранения
    
    Returns
    -------
    fig_axes : dict
        Словарь {var_name: {'fig': fig, 'ax': ax, 'stats': stats}}
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    # === Настройка цветов по умолчанию ===
    if colors is None:
        colors = {
            'original': 'black',
            'cleaned': 'gray',
            'median': 'cadetblue',
            'ci_fill': 'cadetblue',
            'simulations': 'lightgray',
            'zero_line': 'firebrick',
            'cf_serie': 'purple'
        }
    
    # === Подготовка результатов ===
    results_stats = {}
    fig_axes = {}
    
    # === Построение ОТДЕЛЬНОГО графика для каждой переменной ===
    for var_name, data in cf_results.items():
        if var_name != "policy_shocks":
        
            original = data['original']
            cleaned = data['clean']
            if 'cf_serie' in list(data.keys()):
                cf_serie = data['cf_serie']
            else:
                cf_serie = None
            sims = data['sims']  # (n_simulations, nobs)
            n_simulations = sims.shape[0]
            nobs = sims.shape[1]
            
            # Статистики симуляций
            median = np.median(sims, axis=0)
            ci = {}
            for level in ci_levels:
                alpha = (100 - level) / 2
                ci[level] = (
                    np.percentile(sims, alpha, axis=0),
                    np.percentile(sims, 100 - alpha, axis=0)
                )
            
            # Сохраняем статистику
            results_stats[var_name] = {
                'median': median,
                'ci': ci,
                'mean_sim': np.mean(sims, axis=0),
                'std_sim': np.std(sims, axis=0)
            }
            
            # === Создание НОВОЙ фигуры для этой переменной ===
            fig, ax = plt.subplots(figsize=figsize)
            
            # 1. Исходный ряд
            ax.plot(time_index[:nobs], original, 
                    color=colors['original'], linewidth=2, 
                    label='Исходный ряд', zorder=5)
            
            
            # 2. Очищенный ряд (без шоков политики)
            ax.plot(time_index[:nobs], cleaned, 
                    color=colors['cleaned'], linewidth=1.5, linestyle='--', 
                    label='Без шоков политики', zorder=4)
            
            # 3. Медиана симуляций
            ax.plot(time_index[:nobs], median, 
                    color=colors['median'], linewidth=2, 
                    label='Медиана симуляций', zorder=3)
            
            # 4. Доверительные интервалы
            for level in sorted(ci_levels, reverse=True):
                lo, hi = ci[level]
                alpha_fill = 0.20 if level == max(ci_levels) else 0.35
                ax.fill_between(time_index[:nobs], lo, hi, 
                            color=colors['ci_fill'], alpha=alpha_fill , 
                            label=f'{level}% симуляций', zorder=2)
            
            # 5. Нулевая линия
            ax.axhline(0, color=colors['zero_line'], linewidth=2, zorder=1)
            
            # 6. контрфактик
            
            if cf_serie is not None:

                ax.plot(time_index[:nobs], cf_serie, 
                    color=colors['cf_serie'], linewidth=3, 
                    label='Контрфактический ряд', zorder=6)
            
            # Оформление
            title_prefix = suptitle + ": " if suptitle else ""
            ax.set_title(f'{title_prefix}{data['name']}', fontsize=15, fontweight='bold')
            ax.legend(loc='best', fontsize=14, framealpha=0.9)
            ax.grid(alpha=0.3)
            ax.set_xlabel('Годы')
            
            # Форматирование дат
            if isinstance(time_index, (pd.DatetimeIndex, pd.PeriodIndex)):
                ax.tick_params(axis='x', rotation=45, labelsize=12)
            
            plt.tight_layout()
            
            # Сохранение (если нужно)
            if savefig and savepath is not None:
                save_name = f"{savepath}_{var_name}.png"
                plt.savefig(save_name, dpi=dpi, bbox_inches='tight')
                print(f"Сохранено: {save_name}")
            
            plt.show()
            
            # Сохраняем handles для возможного дальнейшего использования
            fig_axes[var_name] = {
                'fig': fig,
                'ax': ax,
                'stats': results_stats[var_name]
            }
    
    return fig_axes

def plot_irf_single_var_grid(
    irf_sims,
    response_var: int = 0,
    y_label: Optional[str] = None,
    u_labels: Optional[Sequence[str]] = None,
    horizon: Optional[int] = None,
    n_cols: int = 2,
    figsize: Tuple[int, int] = (9, 10),
    title: str = "Функции импульсного отклика (одна переменная)",
    ci_color: str = "cadetblue",
    ci_alpha_inner: float = 0.3,
    ci_alpha_outer: float = 0.15,
    median_color: str = "black",
    median_linestyle: str = "--",
    shock_colors: Optional[Sequence[str]] = None,
    legend_bbox: tuple = (0.5, 0.01),
    legend_loc: str = "upper center",
    legend_ncol: int = 4,
    suptitle_y: float = 0.99,
    sign_draws: bool = False
):
    """
    Построить сетку IRF для ОДНОЙ переменной на несколько шоков.

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
    ci_color : str, optional
        Цвет для интервалов доверия.
    ci_alpha_inner : float, optional
        Альфа для внутреннего (68%) CI.
    ci_alpha_outer : float, optional
        Альфа для внешнего (90%) CI.
    median_color : str, optional
        Цвет медианы.
    median_linestyle : str, optional
        Стиль линии для медианного отклика.
    shock_colors : sequence[str], optional
        Цвета для разных шоков (по одному на шок). Если None, используется 
        один цвет для всех.
    legend_bbox, legend_loc, legend_ncol : параметры легенды
    suptitle_y : float, optional
        Вертикальное положение suptitle.
    sign_draws : bool, optional
        Использует медианный отклик, иначе берется нулевая симуляция.

    Возвращает
    -------
    fig, axes : matplotlib.figure.Figure, np.ndarray
        Фигура и массив осей.
    """
    # Приводим к ndarray формы (n_draws, horizon+1, K, K)
    arr = np.asarray(irf_sims)
    if arr.ndim == 3:
        # один draw: (h+1, K, K) -> (1, h+1, K, K)
        arr = arr[None, ...]
        warnings.warn("Найден только один draw, проверьте размерности")
    elif arr.ndim < 3:
        raise ValueError("irf_sims должен быть list/ndarray размерности (h+1,K,K) или (n_draws,h+1,K,K)")

    n_draws, h_plus_1, K1, K2 = arr.shape
    if K1 != K2:
        raise ValueError("Последние два измерения должны быть равны (K x K)")

    K_endo = K1
    n_shocks = K2  # количество шоков

    if horizon is None:
        horizon = h_plus_1 - 1
    elif horizon + 1 > h_plus_1:
        raise ValueError("Параметр horizon слишком большой для переданных IRF")

    # Метки по умолчанию
    if y_label is None:
        y_label = f"Var {response_var}"
    if u_labels is None:
        u_labels = [f"Shock {j}" for j in range(n_shocks)]
    if len(u_labels) != n_shocks:
        raise ValueError("Длина u_labels должна соответствовать числу шоков K")

    # Расчет сетки
    n_rows = int(np.ceil(n_shocks / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Обработка случая, когда axes может быть не массивом
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    fig.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)

    # Легенда (прокси-элементы)
    legend_elements = [
        Line2D([0], [0], color=median_color, label="Медиана" if sign_draws else 'Оригинальный отклик'),
        Line2D([0], [0], color=median_color, ls=median_linestyle, label="Медианный отклик"),
        Patch(facecolor=ci_color, alpha=ci_alpha_inner, label="68% ДИ"),
        Patch(facecolor=ci_color, alpha=ci_alpha_outer, label="90% ДИ"),
    ]

    x_axis = np.arange(horizon + 1)

    # Отрисовка каждого шока
    for shock_idx in range(n_shocks):
        row = shock_idx // n_cols
        col = shock_idx % n_cols
        ax = axes[row, col]

        # responses shape: (n_draws, horizon+1)
        responses = arr[:, : (horizon + 1), response_var, shock_idx]

        # медиана по каждому горизонту
        median = np.median(responses, axis=0)

        # original irf
        original_irf = responses[0, :]

        # находим draw, который наиболее близок к медиане
        ssd = np.sum((responses - median) ** 2, axis=1)
        median_index = int(np.argmin(ssd))
        med_res = responses[median_index, :]

        # доверительные интервалы
        lower68 = np.percentile(responses, 16, axis=0)
        upper68 = np.percentile(responses, 84, axis=0)
        lower90 = np.percentile(responses, 5, axis=0)
        upper90 = np.percentile(responses, 95, axis=0)

        # Выбор цвета для этого шока
        shock_color = shock_colors[shock_idx] if shock_colors is not None else median_color

        # отрисовка
        if sign_draws:
            ax.plot(x_axis, median, color=shock_color, linewidth=2)
        else:
            ax.plot(x_axis, original_irf, color=shock_color, linewidth=2)
        ax.plot(x_axis, med_res, color=shock_color, ls=median_linestyle, linewidth=1.5)
        ax.fill_between(x_axis, lower68, upper68, color=ci_color, alpha=ci_alpha_inner)
        ax.fill_between(x_axis, lower90, upper90, color=ci_color, alpha=ci_alpha_outer)
        ax.axhline(0, color="firebrick", lw=1.5, linestyle='-')

        ax.set_title(f"Шок: {u_labels[shock_idx]}", fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Убираем лишние оси
        if shock_idx >= n_shocks:
            ax.set_visible(False)

    # Общая легенда
    fig.legend(handles=legend_elements, loc=legend_loc, bbox_to_anchor=legend_bbox, 
               ncol=legend_ncol, fontsize=10)
    fig.suptitle(title, fontsize=16, y=suptitle_y, fontweight='bold')
    plt.tight_layout()
    
    if horizon < 13:
        for ax in axes.flat:
            ax.set_xticks(x_axis)
            ax.set_xticklabels([str(i) for i in x_axis], ha="right")
    
    plt.show()

    return fig, axes