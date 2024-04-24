import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind

def get_sample_size(mu, std, eff=1.01, alpha=0.05, beta=0.2):
    
    '''Первым шагом вычисляются критические значения Z-оценки для уровня значимости alpha/2 и мощности теста (1-beta).
        Это делается с помощью функции stats.norm.ppf() из библиотеки SciPy, 
        которая вычисляет квантили нормального распределения.
       stats.norm.ppf возвращает обратное значение функции нормального распределения
       (т.е., значение, соответствующее заданной вероятности) 
       для указанного значения квантиля q,с заданным средним значением loc и стандартным отклонением scale.
       Для стандартного нормального распределения, симметричного относительно нуля,
       stats.norm.ppf(alpha / 2) и stats.norm.ppf(1 - alpha / 2) будут возвращать противоположные по знаку значения,
       но с одинаковой абсолютной величиной.
       Затем вычисляется квадрат разницы между средними значениями (mu и mu * eff) и умножается на себя.
       Это представляет собой ожидаемое изменение среднего, умноженное на его ожидаемое отклонение.
       Затем сумма квадратов критических значений Z-оценок находится и умножается на двойное значение дисперсии. 
       Это представляет собой сумму дисперсий двух групп, которые мы хотим сравнить.
       Наконец, вычисляется необходимый размер выборки как результат деления суммы Z-оценок на квадрат разницы средних значений
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

def calc_integral(coordinates, values):
    '''Функция принимает два аргумента: coordinates и values,
    предполагая, что coordinates содержит координаты точек,
    а values содержит значения функции в этих точках.
    Внутри функции создается переменная integral, которая инициализируется значением 0. 
    Эта переменная будет использоваться для накопления результата интегрирования.
    Затем происходит итерация по координатам (за исключением последней координаты) с помощью цикла for. 
    Внутри цикла вычисляется разница между следующей и текущей координатами (delta_x), 
    что представляет собой ширину трапеции в текущем сегменте.
    Далее происходит вычисление площади трапеции для текущего сегмента: 
    ширина трапеции (delta_x) умножается на среднее значение функции в текущем и следующем узле
    ((values[i] + values[i + 1]) / 2), и это значение добавляется к integral.
    После завершения цикла возвращается integral, 
    который представляет собой приближенное значение определенного интеграла функции.'''
    integral = 0
    for i in range(len(coordinates) - 1):
        delta_x = coordinates[i + 1] - coordinates[i]
        integral += delta_x * (values[i] + values[i + 1]) / 2
    return integral
    
    
def select_stratified_groups(data, strat_columns, group_size, weights=None, seed=None):
    """Подбирает стратифицированные группы для эксперимента.

    data - pd.DataFrame, датафрейм с описанием объектов, содержит атрибуты для стратификации.
    strat_columns - List[str], список названий столбцов, по которым нужно стратифицировать.
    group_size - int, размеры групп.
    weights - dict, словарь весов страт {strat: weight}, где strat - tuple значений элементов страт,
        например, для strat_columns=['os', 'gender', 'birth_year'] будет ('ios', 'man', 1992).
        Если None, определить веса пропорционально доле страт в датафрейме data.
    seed - int, исходное состояние генератора случайных чисел для воспроизводимости
        результатов. Если None, то состояние генератора не устанавливается.

    return (data_pilot, data_control) - два датафрейма того же формата что и data
        c пилотной и контрольной группами.
    """
    if seed:
        np.random.seed(seed)

    if weights is None:
        len_data = len(data)
        weights = {strat: len(df_) / len_data for strat, df_ in data.groupby(strat_columns)}

    # кол-во элементов страты в группе
    strat_count_in_group = {strat: int(round(group_size * weight)) for strat, weight in weights.items()}

    pilot_dfs = []
    control_dfs = []
    for strat, data_strat in data.groupby(strat_columns):
        if strat in strat_count_in_group:
            count_in_group = strat_count_in_group[strat]
            index_data_groups = np.random.choice(
                np.arange(len(data_strat)),
                count_in_group * 2,
                False
            )
            pilot_dfs.append(data_strat.iloc[index_data_groups[:count_in_group]])
            control_dfs.append(data_strat.iloc[index_data_groups[count_in_group:]])
    data_pilot = pd.concat(pilot_dfs)
    data_control = pd.concat(control_dfs)
    return (data_pilot, data_control)

def get_minimal_determinable_effect(std, sample_size, alpha=0.05, beta=0.2):
    '''Вычисление критических значений Z-оценок:
       t_alpha и t_beta - это критические значения Z-оценок для уровня значимости alpha/2 
       и мощности теста 1-beta соответственно.
       Они используются для определения границ доверительного интервала и мощности теста соответственно.
       Вычисление корня из суммы дисперсий:
            disp_sum_sqrt представляет собой квадратный корень из суммы дисперсий.
            Дисперсия умножается на 2, потому что рассматриваются две группы данных.
       Вычисление минимально обнаружимого эффекта (MDE):
            MDE вычисляется как произведение критических значений Z-оценок
            и корня из суммы дисперсий, деленное на квадратный корень из размера выборки.
       Это позволяет определить, какой размер эффекта можно обнаружить при заданных условиях эксперимента
       (уровень значимости, мощность) и размере выборки.
       Возвращается вычисленное значение MDE.'''
    t_alpha = norm.ppf(1 - alpha / 2, loc=0, scale=1)
    t_beta = norm.ppf(1 - beta, loc=0, scale=1)
    disp_sum_sqrt = (2 * (std ** 2)) ** 0.5
    mde = (t_alpha + t_beta) * disp_sum_sqrt / np.sqrt(sample_size)
    return mde