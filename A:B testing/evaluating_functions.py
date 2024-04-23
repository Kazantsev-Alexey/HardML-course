import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind

def get_sample_size(mu, std, eff=1.01, alpha=0.05, beta=0.2):
    
    '''stats.norm.ppf возвращает обратное значение функции нормального распределения
       (т.е., значение, соответствующее заданной вероятности) 
       для указанного значения квантиля q,
       с заданным средним значением loc 
       и стандартным отклонением scale
       
       Для стандартного нормального распределения, симметричного относительно нуля,
       stats.norm.ppf(alpha / 2) и stats.norm.ppf(1 - alpha / 2) будут возвращать противоположные по знаку значения,
       но с одинаковой абсолютной величиной.
       '''
    t_alpha = abs(stats.norm.ppf(1 - alpha / 2, loc=0, scale=1))
    t_beta = stats.norm.ppf(1 - beta, loc=0, scale=1)

    mu_diff_squared = (mu - mu * eff) ** 2
    z_scores_sum_squared = (t_alpha + t_beta) ** 2
    disp_sum = 2 * (std ** 2)
    sample_size = int(
        np.ceil(
            z_scores_sum_squared * disp_sum / mu_diff_squared
        )
    )
    return sample_size


def estimate_sample_size(df, metric_name, effects, alpha=0.05, beta=0.2):
    """Оцениваем sample size для списка эффектов.

    df - pd.DataFrame, датафрейм с данными
    metric_name - str, название столбца с целевой метрикой
    effects - List[float], список ожидаемых эффектов. Например, [1.03] - увеличение на 3%
    alpha - float, ошибка первого рода
    beta - float, ошибка второго рода

    return - pd.DataFrame со столбцами ['effect', 'sample_size']    
    
    
    
    df['metric'].values.std() и df['metric'].std() возвращают разное значение
    Если вам нужно, чтобы оба метода возвращали одинаковые значения,
    вы можете явно указать метод расчета стандартного отклонения при использовании df['metric'].std(), 
    используя параметр ddof (степени свободы по умолчанию) и установив его в 0
    """
    
    
    metric_values = df[metric_name].values
    mu = metric_values.mean()
    std = metric_values.std()
    sample_sizes = [get_sample_size(mu, std, effect, alpha, beta) for effect in effects]
    res_df = pd.DataFrame({'effect': effects, 'sample_size': sample_sizes})
    return res_df


def estimate_second_type_error(df_pilot_group, df_control_group, metric_name, effects, alpha=0.05, n_iter=10000, seed=None):
    """Оцениваем ошибки второго рода.

    Бутстрепим выборки из пилотной и контрольной групп тех же размеров, добавляем эффект к пилотной группе,
    считаем долю случаев без значимых отличий.
    
    df_pilot_group - pd.DataFrame, датафрейм с данными пилотной группы
    df_control_group - pd.DataFrame, датафрейм с данными контрольной группы
    metric_name - str, названия столбца с метрикой
    effects - List[float], список размеров эффектов ([1.03] - увеличение на 3%).
    alpha - float, уровень значимости для статтеста
    n_iter - int, кол-во итераций бутстрапа
    seed - int or None, состояние генератора случайных чисел

    return - dict, {размер_эффекта: ошибка_второго_рода}
    """
    if seed is not None:
        np.random.seed(seed)
    pilot_values = df_pilot_group[metric_name].values
    control_values = df_control_group[metric_name].values
    len_pilot = len(pilot_values)
    len_control = len(control_values)
    mean_pilot = np.mean(pilot_values)
    std_pilot = np.std(pilot_values)
    effect_to_second_type_error = dict()
    
    for effect in effects:
        p_values = []
        for i in range(n_iter):
            bs_pilot_values = np.random.choice(pilot_values, len_pilot)
            bs_pilot_values += np.random.normal(mean_pilot * (effect-1), std_pilot/10, len_pilot)
            bs_control_values = np.random.choice(control_values, len_control)
            _, p_value = ttest_ind(bs_pilot_values, bs_control_values)
            p_values.append(p_value)
        second_type_error = np.mean(np.array(p_values)>alpha)
        effect_to_second_type_error[effect] = second_type_error
    return effect_to_second_type_error


def estimate_first_type_error(df_pilot_group, df_control_group, metric_name, alpha=0.05, n_iter=10000, seed=None):
    """Оцениваем ошибку первого рода.

    Бутстрепим выборки из пилотной и контрольной групп тех же размеров, считаем долю случаев с значимыми отличиями.
    
    df_pilot_group - pd.DataFrame, датафрейм с данными пилотной группы
    df_control_group - pd.DataFrame, датафрейм с данными контрольной группы
    metric_name - str, названия столбца с метрикой
    alpha - float, уровень значимости для статтеста
    n_iter - int, кол-во итераций бутстрапа
    seed - int or None, состояние генератора случайных чисел.

    return - float, ошибка первого рода
    """
    np.random.seed(seed)
    results = []
    for i in range(n_iter):
        var1 = np.random.choice(df_pilot_group[metric_name], len(df_pilot_group))
        var2 = np.random.choice(df_control_group[metric_name], len(df_control_group))
        _, p_value = ttest_ind(var1, var2)
        result = int(p_value<=alpha)
        results.append(result)
    return np.mean(results)


def calculate_sales_metrics(df, cost_name, date_name, sale_id_name, period, filters=None):
    """Вычисляет метрики по продажам.
    
    df - pd.DataFrame, датафрейм с данными. Пример
        pd.DataFrame(
            [[820, '2021-04-03', 1, 213]],
            columns=['cost', 'date', 'sale_id', 'shop_id']
        )
    cost_name - str, название столбца с стоимостью товара
    date_name - str, название столбца с датой покупки
    sale_id_name - str, название столбца с идентификатором покупки (в одной покупке может быть несколько товаров)
    period - dict, словать с датами начала и конца периода пилота и препилота.
        Пример, {'begin': '2020-01-01', 'end': '2020-01-08'}.
        Дата начала периода входит в полуинтервал, а дата окончания нет,
        то есть '2020-01-01' <= date < '2020-01-08'.
    filters - dict, словарь с фильтрами. Ключ - название поля, по которому фильтруем, значение - список значений,
        которые нужно оставить. Например, {'user_id': [111, 123, 943]}.
        Если None, то фильтровать не нужно.

    return - pd.DataFrame, в индексах все даты из указанного периода отсортированные по возрастанию, 
        столбцы - метрики ['revenue', 'number_purchases', 'average_check', 'average_number_items'].
    """
    period['begin'] = pd.to_datetime(period['begin'])
    period['end'] = pd.to_datetime(period['end'])
    df[date_name] = pd.to_datetime(df[date_name])

    mask = ((df[date_name] >= period['begin']) & (df[date_name] < period['end'])).values
    if filters:
        for column, values in filters.items():
            mask = mask & df[column].isin(values).values
    df_filtered = df.iloc[mask]

    dates = pd.date_range(start=period['begin'], end=period['end'], freq='D')
    dates = dates[dates < period['end']]
    df_dates = pd.DataFrame(index=dates)

    df_revenue = (
        df_filtered
        .groupby(date_name)[[cost_name]].sum()
        .rename(columns={cost_name: 'revenue'})
    )
    df_number_purchases = (
        df_filtered
        .groupby(date_name)[[sale_id_name]].nunique()
        .rename(columns={sale_id_name: 'number_purchases'})
    )
    df_average_check = (
        df_filtered
        .groupby([date_name, sale_id_name])[[cost_name]].sum()
        .reset_index()
        .groupby(date_name)[[cost_name]].mean()
        .rename(columns={cost_name: 'average_check'})
    )
    df_average_number_items = (
        df_filtered
        .groupby([date_name, sale_id_name])[[cost_name]].count()
        .reset_index()
        .groupby(date_name)[[cost_name]].mean()
        .rename(columns={cost_name: 'average_number_items'})
    )
    list_df = [df_revenue, df_number_purchases, df_average_check, df_average_number_items]
    df_res = df_dates.copy()
    for df_ in list_df:
        df_res = pd.merge(df_res, df_, how='outer', left_index=True, right_index=True)
    df_res.sort_index(inplace=True)
    df_res.fillna(0, inplace=True)
    return df_res



def get_bernoulli_confidence_interval(values: np.array):
    """Вычисляет доверительный интервал для параметра распределения Бернулли.

    :param values: массив элементов из нулей и единиц.
    :return (left_bound, right_bound): границы доверительного интервала.
    """
    n = len(values)
    p = sum(values) / n
    z = 1.96
    
    std = np.sqrt((p * (1 - p))/n)
    left_bound = p - z * std
    right_bound = p + z * std
    ci = np.clip((left_bound, right_bound),0,1)
    return ci