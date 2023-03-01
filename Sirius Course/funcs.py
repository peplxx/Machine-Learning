from math import sqrt


def sign(x):
    if x > 0:
        return 1
    return -1


def round_to_2(x):
    return round(x, 2)


def correlation_solution(xs, ys):
    _x = sum(xs) / len(xs)
    _y = sum(ys) / len(ys)
    upp = sum([(xs[i] - _x) * (ys[i] - _y) for i in range(len(xs))])
    down = (sum([(xi - _x) ** 2 for xi in xs]) * sum([(yi - _y) ** 2 for yi in ys])) ** 0.5
    return round_to_2(upp / down)


def correlation_solution_line(col_x, col_y):
    return round_to_2(((col_x - col_x.mean()) * (col_y - col_y.mean())).sum() / (((col_x - col_x.mean()) ** 2).sum() * \
                                                                                 (
                                                                                             col_y - col_y.mean()) ** 2).sum() ** 0.5)


def linear_regression_solution(ws, vectors):
    ans = [round_to_2(sum([ws[i] * vectors[j][i] for i in range(len(ws))])) for j in range(len(vectors))]
    return ans


def mean_absolute_difference_solution(real, predicted):
    return round_to_2(sum([abs(real[i] - predicted[i]) for i in range(len(real))]) / len(real))


def new_columns_solution(column):
    cols = list(column)
    empty = [0] * len(cols)
    d = dict()
    for index1, col in enumerate(cols):
        curr = empty.copy()
        for index2, c in enumerate(cols):
            if c == col:
                curr[index2] = 1
        d[col] = curr
    return d


def target_coding_solution(data, factor_column, target_column):
    factors = data[factor_column].unique()
    new_data = data.groupby(factor_column)[target_column].mean()

    data['encoded'] = [0] * len(data)
    data['encoded'] = data['encoded'].astype('float')
    data['encoded'] = [round(new_data[x], 2) for x in data[factor_column]]


def standard_deviation_solution(column):
    n = len(column)
    _x = sum(column) / n
    value = (sum([(xi - _x) ** 2 for xi in column]) / n) ** 0.5
    return round_to_2(value)


def standartize_column_solution(column):
    devitation = standard_deviation_solution(column)
    n = len(column)
    if devitation == 0:
        return pd.Series([0.0] * n)

    _x = sum(column) / n
    result_column = pd.Series([float((column[i] - _x) / devitation) for i in range(n)])
    result_column = result_column.round(2)
    return result_column


def correlation_table_solution(table):
    ans = []
    for coll_x in table.columns:
        k = []
        for coll_y in table.columns:
            k.append(correlation_solution(table[coll_x], table[coll_y]))
        ans.append(k)

    return ans


def avg_metriks(ys, _ys):
    return sum([abs(ys[i] - _ys[i]) for i in range(len(ys))]) / (2 * len(ys))


def perceptron_classify_solution(ws, xs, f=sign):
    if len(xs) == len(ws) - 1:
        xs.append(1)
    k = f(sum([ws[i] * xs[i] for i in range(len(ws))]))
    return k


def multilayer_perceptron_solution(wss, fs, x):
    inputs = x
    for index, layer in enumerate(wss):
        outputs = []
        for ws in layer:
            outputs.append(perceptron_classify_solution(ws=ws, xs=inputs, f=fs[index]))
        inputs = outputs
    return inputs[0]


def perceptron_train_solution(xs, ys, learning_rate=1, k=100):
    current_k = 1
    ws = [0] * (len(xs[0]) + 1)
    while current_k <= k:
        _ys = perceptron_classify_solution(ws, xs)
        for i in range(len(_ys)):
            _ys = perceptron_classify_solution(ws, xs)
            if avg_metriks(ys, _ys) <= 0:
                break
            for w in range(len(ws)):
                ws[w] = ws[w] + learning_rate * (ys[i] - _ys[i]) * xs[i][w]
        current_k += 1
    return ws


def score_model(model, x_val, y_val):
    y_pred = model.predict(x_val)

    res = 0

    for i in range(len(y_val)):
        res += abs(y_pred[i] - y_val[i])

    return res / len(y_val)


def make_featured_data(config, data):
    new_data = []
    for i in range(len(data)):
        row = []
        for feature in config:
            row.append(data[i][feature])
        new_data.append(row)
    return new_data


def grid_search_solution(
        model,
        model_configs,
        x_train, y_train,
        x_val, y_val):
    """
    Принимает набор конфигураций модели линейной регрессии и находит лучшую из них
    с помощью алгоритма grid search.
    Для оценки точности модели на валидационной выборке используется функция score_model.

    Аргументы:
        model: Модель.
        model_configs: Список конфигураций модели линейной регрессии.
                       Каждая конфигурация — список номеров факторов,
                       которые используются для обучения модели.
        x_train: Список объектов обучающей выборки.
        y_train: Список значений предсказываемой характеристики для объектов из обучающей выборки.
                 Значение на $i$-ой позиции в списке соответствует $i$-ому объекту обучающей выборки.
        x_val: Список объектов валидационной выборки.
        y_val: Список значений предсказываемой характеристики для объектов из валидационной выборки.
               Значение на $i$-ой позиции в списке соответствует $i$-ому объекту валидационной выборки.

    Возвращаемое значение:
        Возвращает пару значений: лучшая конфигурация, точность модели с данной конфигурацией
        на валидационной выборке.
    """
    max_cfg = ([-1], 9999)
    for config in model_configs:
        # model.set_params(model_configs[0][0])
        x_featured = make_featured_data(config, x_train)
        model.fit(x_featured, y_train)
        x_val_featured = make_featured_data(config, x_val)
        if (score := score_model(model, x_val_featured, y_val)) < max_cfg[1]:
            max_cfg = (config, score)
    max_cfg = (max_cfg[0], round_to_2(max_cfg[1]))
    return max_cfg


def dichotomous_search_solution(f, a, b, eps):
    """
    Производит поиск минимума заданной функции в заданном интервале с помощью метода дихотомии.

    Аргументы:
        f: Функция от одного аргумента, минимум которой необходимо найти.
        a: Левая граница интервала, в котором происходит поиск минимума.
        b: Правая граница интервала, в котором происходит поиск минимума.
        eps: Допустимая погрешность при поиске минимума.

    Возвращаемое значение:
        Возвращает координату точки, в которой достигается минимальное значение функции.
        Координата должна быть округлена до 2 знаков после запятой.
    """
    l, r = min(a, b), max(a, b)
    while r - l > 3 * eps:
        m = (r + l) / 2
        x1, x2 = m - eps, m + eps
        if f(x1) >= f(x2):
            l = x1
        else:
            r = x2
    return round_to_2((l + r) / 2)


def dist_v(x1, x2):
    return ((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) ** (1 / 2)


def gradient_descent_solution(grad_f, x_0, alpha, eps):
    x_1 = grad_f(x_0)
    x_1 = [x_0[0] - alpha * x_1[0], x_0[1] - alpha * x_1[1]]
    while dist_v(x_1, x_0) > eps:
        x_2 = grad_f(x_1)
        x_2 = [x_1[0] - alpha * x_2[0], x_1[1] - alpha * x_2[1]]
        x_0, x_1 = x_1, x_2
    return [round_to_2(x_1[0]), round_to_2(x_1[1])]


def sub_with_p(random_gen, p, new_x, x):
    """
    С заданной вероятностью возвращает точку `new_x`, в остальных случаях
    возвращает точку x.

    Аргументы:
        random_gen: Генератор случайных чисел.
        p: Вероятность, с которой нужно вернуть первую точку.
        new_x: Точка, которая возвращается с вероятностью p.
        x: Точка, которая возвращает в остальных случаях.

    Возвращаемое значение:
        Точка, которая с вероятностью p будет равна new_x. В противном случае точка будет равна x.
    """

    val = random_gen.random()

    if val > p:
        return x

    return new_x


def rand_sign(random_gen):
    c = random_gen.random()
    if c > 0.5:
        return -1
    return 1


def simulated_annealing_solution(f, x_0, t_0, lam, eps, random_gen):
    """
    Производит поиск минимума функции с помощью метода имитации отжига.

    Аргументы:
        f: Функция, минимум которой необходимо найти.
        x_0: Стартовая точка, из которой начинается процесс поиска минимума.
        t_0: Начальная температура системы.
        lam: Коэффициент охлаждения системы на каждом шаге.
             Охлаждение происходит по правилу T = lam * T.
        eps: Когда температура становится меньше eps, метод имитации отжига останавливает свою работу.
        random_gen: Генератор случайных чисел.

    Возвращаемое значение:
        Точка, которую метод считает минимумом функции.
    """
    T = t_0
    x = x_0
    while T > eps:
        T = lam * T
        x_ = x + random_gen.random() * rand_sign(random_gen)
        delta = f(x_) - f(x)
        if delta > 0:
            x = sub_with_p(random_gen, exp(-delta / T), x_, x)
        else:
            x = x_
    return x


def concentrate(a):
    ans = []
    for e in a:
        ans += [*e]
    return ans


def split_into_k(l, k):
    """
    Разделяет список на k частей.

    Аргументы:
        l: Список с объектами.
        k: Число частей, на которые нужно разделить список.

    Возвращаемое значение:
        Возвращает список из k частей исходного списка.
    """

    l_mod_k = len(l) % k
    l_div_k = len(l) // k

    res = []

    for i in range(k):
        res.append(l[i * l_div_k:(i + 1) * l_div_k])

    for i in range(l_mod_k):
        res[i].append(l[l_div_k * k + i])

    return res


def k_fold_solution(model, data_x, data_y, k, random_gen):
    data_xy = list(zip(data_x, data_y))
    random_gen.shuffle(data_xy)
    unzipped = list(zip(*data_xy))
    data_x, data_y = unzipped[0], unzipped[1]
    chunks = split_into_k(data_x, k)
    chunks_y = split_into_k(data_y, k)
    """
    Проводит кросс-валидацию заданной модели методом k-Fold.

    Аргументы:
        model: Модель, точность которой нужно оценить с помощью кросс-валидации.
        data_x: Список объектов, на основе которых нужно построить модель.
                Каждый объект представлен списком значений факторов.
        data_y: Список значений предсказываемой величины для каждого из объектов.
                На $i$-ой позиции в списке data_y находится предсказываемое
                значение для $i$-го объекта из списка data_x.
             k: Количество частей, на которые нужно разбить данные при кросс-валидации.
        random_gen: Генератор случайных чисел.

    Возвращаемое значение:
        Усреднённая по всем итерациям k-Fold точность модели.
    """
    scores = []
    for i in range(k):
        score_x = list(chunks[i])
        score_y = list(chunks_y[i])
        train_y = concentrate(chunks_y[:i] + chunks_y[i + 1:])
        train_x = concentrate(chunks[:i] + chunks[i + 1:])
        model.fit(train_x, train_y)
        s = score_model(model, score_x, score_y)
        scores.append(round_to_3(s))
    return round_to_3(sum(scores) / k)
