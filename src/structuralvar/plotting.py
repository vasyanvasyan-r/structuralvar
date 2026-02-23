import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from typing import Optional, Sequence, Tuple


def plot_irf_single(
    irfs,
    i,
    j,
    horizon,
    u_dict,
    y_dict,
    plot_simulations=False,
    ci_levels=(68, 90),
    figsize=(10, 5),
):

    irfs = np.asarray(irfs)
    x_axis = np.arange(horizon + 1)

    # n_draws × (horizon+1)
    responses = irfs[:, :, i, j]

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
        plt.plot(x_axis, responses.T, color="lightgray", alpha=0.2, lw=0.8)
        plt.plot(x_axis, median, color="black", label="Медиана")
        plt.plot(x_axis, med_res, color="black", ls="--", label="Медианный отклик")
        plt.axhline(0, color="grey", lw=1)
        plt.legend(loc="lower right")
        plt.title(f"{y_dict[i]} на {u_dict[j]} (все симуляции)")
        plt.show()

    # ---------- итоговый график ----------
    plt.figure(figsize=figsize)
    plt.plot(x_axis, median, color="black", label="Медиана")
    plt.plot(x_axis, med_res, color="black", ls="--", label="Медианный отклик")

    for level, (lo, hi) in ci.items():
        alpha = 0.3 if level == max(ci_levels) else 0.15
        plt.fill_between(x_axis, lo, hi, alpha=alpha, label=f"{level}% CI")

    plt.axhline(0, color="grey", lw=1)
    plt.legend()
    plt.title(f"{y_dict[i]} на {u_dict[j]}")
    plt.show()

def plot_hd_bars_signed(df_hd, 
                        variable_name = None, 
                        shocks_labels = None, 
                        cumm = True):
    length = df_hd.shape[1]
    variable_name = variable_name or str(df_hd.index[0])
    shocks_labels = shocks_labels or [i for i in df_hd.index[1:]]
    x = np.arange(df_hd.shape[1])
    x = x - 12 + df_hd.columns.to_list()[0].month - 1

    plt.figure(figsize=(10, 6))

    # Основания для положительных и отрицательных значений
    bottom_pos = np.zeros(length)
    bottom_neg = np.zeros(length)

    # Используем палитру заранее, чтобы цвета не повторялись
    list_of_colors = ['powderblue', 'moccasin', 'salmon', 'lightgreen', 'thistle']
    color_of_variable = 'black'
    colors = list_of_colors[:len(shocks_labels)]

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
        labels=[i.year for i in df_hd.columns[x % 12 == 0]],
        rotation=45,
        ha="right"
    )

    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(f"{'Накопленная историческая декомпозиция' if cumm else 'Историческая декомпозиция'}: {variable_name}")
    plt.legend(loc="lower right", fontsize=9)
    plt.xlabel("Год")
    plt.ylabel("Вклад шоков")
    plt.tight_layout()
    plt.show()


def plot_irf_grid(
    irf_sims,
    y_labels: Optional[Sequence[str]] = None,
    u_labels: Optional[Sequence[str]] = None,
    horizon: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 15),
    title: str = "Функции импульсного отклика (IRF)",
    ci_color: str = "cadetblue",
    ci_alpha_inner: float = 0.3,
    ci_alpha_outer: float = 0.15,
    median_color: str = "black",
    median_linestyle: str = "--",
    legend_bbox: tuple = (0.5, 0.01),
    legend_loc: str = "upper center",
    legend_ncol: int = 4,
    suptitle_y: float = 0.99,
    show: bool = True,
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
    show : bool, optional
        Показывать plt.show() в конце (если False — можно продолжать правки).

    Возвращает
    -------
    fig, axes : matplotlib.figure.Figure, np.ndarray
        Фигура и массив осей (K x K).
    """
    # Приводим к ndarray формы (n_draws, horizon+1, K, K)
    arr = np.array(irf_sims)
    if arr.ndim == 3:
        # один draw: (h+1, K, K) -> (1, h+1, K, K)
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError("IRF_ortho должен быть list/ndarray форм (h+1,K,K) или (n_draws,h+1,K,K)")

    n_draws, h_plus_1, K1, K2 = arr.shape
    if K1 != K2:
        raise ValueError("Последние два измерения должны быть равны (K x K)")

    K_endo = K1

    if horizon is None:
        horizon = h_plus_1 - 1
    if horizon + 1 > h_plus_1:
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
        Line2D([0], [0], color=median_color, label="Медиана"),
        Line2D([0], [0], color=median_color, ls=median_linestyle, label="Медианный отклик"),
        Patch(facecolor=ci_color, alpha=ci_alpha_inner, label="68% CI"),
        Patch(facecolor=ci_color, alpha=ci_alpha_outer, label="90% CI"),
    ]

    x_axis = np.arange(horizon + 1)

    for i in range(K_endo):
        for j in range(K_endo):
            ax = axes[i, j] if K_endo > 1 else axes  # если K==1 axes может быть объектом, а не массивом

            # responses shape: (n_draws, horizon+1)
            responses = arr[:, : (horizon + 1), i, j]

            # медиана по каждому горизонту
            median = np.median(responses, axis=0)

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
            ax.plot(x_axis, median, color=median_color)
            ax.plot(x_axis, med_res, color=median_color, ls=median_linestyle)
            ax.fill_between(x_axis, lower68, upper68, color=ci_color, alpha=ci_alpha_inner)
            ax.fill_between(x_axis, lower90, upper90, color=ci_color, alpha=ci_alpha_outer)
            ax.axhline(0, color="grey", lw=1)

            ax.set_title(f"Отклик: {y_labels[i]}\nШок: {u_labels[j]}", fontsize=9)
            ax.grid(True, alpha=0.3)

    # Общая легенда
    fig.legend(handles=legend_elements, loc=legend_loc, bbox_to_anchor=legend_bbox, ncol=legend_ncol, fontsize=12)
    fig.suptitle(title, fontsize=18, y=suptitle_y)
    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes