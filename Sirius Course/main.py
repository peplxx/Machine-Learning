import os
from copy import deepcopy
from math import cos
from math import floor
from math import pi
from math import sin
import random

import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from funcs import correlation_solution_line as correlation_solution
from math import exp
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
import matrixs

os.add_dll_directory("C:\dlls")


# from sklearn.linear_model import LinearRegression


class ShuffleOnceRandom():
    """
    Генератор случайных чисел, который запрещает использовать
    функцию shuffle больше одного раза.
    """

    def __init__(self, seed=None):
        import random

        self._random_gen = random.Random(seed)
        self._shuffle_cnt = 0

    def shuffle(self, l):
        if self._shuffle_cnt > 0:
            raise RuntimeError('Нельзя использовать функцию shuffle больше одного раза')

        self._shuffle_cnt += 1
        self._random_gen.shuffle(l)


def custom_compare(x, y):
    if str(x) != str(y):
        raise RuntimeError(f'Ожидаемое значение: {y}. Фактическое: {x}')


class ChoicesNRandom():
    """
    Генератор случайных чисел, который запрещает использовать
    функцию choices больше n раз.
    """

    def __init__(self, seed=None, n=1):
        import random

        self._random_gen = random.Random(seed)
        self._choices_cnt = 0
        self._n = n

    def choices(self, *args, **kwargs):
        if self._choices_cnt >= self._n:
            raise RuntimeError(f'Нельзя использовать функцию choices больше {self._n} раз')

        self._choices_cnt += 1
        return self._random_gen.choices(*args, **kwargs)


def sign(x):
    if x > 0:
        return 1
    return -1


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


def relu(x):
    if x > 0:
        return x

    return 0


def sigmoid(x):
    return round(1 / (1 + exp(-x)), 3)


def round_to_3(x):
    return round(x, 3)


def round_to_2(x):
    return round(x, 2)


def get_col(matrix, i):
    return [col[i] for col in matrix]


def skal(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return sum(v1 * v2)


def matrix_by_matrix(matrix_a, matrix_b):
    n1, m1 = len(matrix_a), len(matrix_a[0])
    n2, m2 = len(matrix_b), len(matrix_b[0])
    if n2 != m1:
        return -1
    result = [[0] * m2 for i in range(n1)]
    for i in range(n1):
        for j in range(m2):
            v2 = get_col(matrix_b, j)
            v1 = matrix_a[i]
            value = skal(v1, v2)
            result[i][j] = value
    return result


def matrix_by_vector(matrix_a, vector_b):
    n1, m1 = len(matrix_a), len(matrix_a[0])
    n2, m2 = len(vector_b), 1
    if n2 != m1:
        return -1
    result = [0 * m2 for i in range(n1)]
    for i in range(n1):
        for j in range(m2):
            v1 = matrix_a[i]
            value = skal(v1, vector_b)
            result[i] = value
    return result


def matrix_multiplication_solution(a, b):
    if type(b[0]) is (np.array):
        return matrix_by_matrix(a, b)
    return matrix_by_vector(a, b)


def ziped(l, n):
    return [get_col(l, i) for i in range(n)]


def transpose_matrix_solution(matrix):
    """
    Производит транспонирование переданной матрицы.

    Аргументы:
        matrix: Матрица, которую нужно транспонировать.

    Возвращаемое значение:
        Возвращает матрицу, которая является результатом транспонирования матрицы-аргумента.
    """
    n, m = len(matrix), len(matrix[0])
    t_matrix = [[0] * n for i in range(m)]
    for i in range(n):
        v = matrix[i]
        for j in range(m):
            t_matrix[j][i] = v[j]
    return t_matrix


def logistic_summary_loss_gradient_solution(w, X, y):
    n = len(y)
    grad = np.zeros(len(w))
    for i in range(n):
        k = sigmoid(-y[i] * skal(X[i], w))
        f = -y[i] * k * X[i]
        grad += f
    grad = np.array([round_to_2(g) for g in grad])
    return grad


def logistic_regression_solve_solution(
        data, factor_names, y_name,
        learning_rate=0.01, eps=0.1):
    """
    С помощью градиентного спуска строит модель логистической регрессии по переданному набору данных.

    Аргументы:
        data: Таблица с объектами обучающей выборки.
              Каждый объект описывается набором численных факторов.
              В данных может быть представлено больше факторов, чем модель должна использовать для предсказания.
              Искусственного константного фактора, который для всех объектов равен 1 и
              который будет использоваться моделью для предсказания, в таблице нет.
        factor_names: Список названий факторов, которые модель должна использовать для предсказания.
        y_name: Название столбца таблицы, в котором для каждого объекта содержится значение предсказываемого класса.
                Класс может иметь значение либо -1, либо 1.
        learning_rate: Опциональный параметр. Коэффициент скорости обучения, который используется в алгоритме градиентного спуска.
        eps: Опциональный параметр. Минимальное расстояние между текущей точкой градиентного спуска и следующей,
             при котором работа алгоритма останавливается.

    Возвращаемое значение:
        Возвращает вектор весов модели.
        Координата вектора с индексом 0 соответствует свободному коэффициенту модели.
        Координата вектора с индексом i соответствует фактору с индексом i - 1 в списке factor_names.
    """

    X = data[factor_names].to_numpy()
    y = data[y_name].to_numpy()

    ones = np.ones((len(y), 1))
    n = len(y)

    X = np.hstack([ones, X])

    # Необходимо задать стартовый набор весов, который является начальной
    # точкой для алгоритма градиентного спуска.
    # В качестве стартового набора весов необходимо использовать вектор, состоящий из значений 0
    w_cur = np.zeros(len(X[0]))

    while True:
        # Вычисление градиента с помощью функции logistic_summary_loss_gradient_solution.
        # Важно убрать округление результата работы функции до 2 знаков после запятой
        gradient_value = logistic_summary_loss_gradient_solution(w_cur, X, y)

        # Полезно уменьшить значение градиента, разделив каждую его координату на число объектов
        # в выборке, на которой происходит обучение модели
        gradient_value /= len(y)

        # Классический шаг градиентного спуска: переход из текущей точки в направлении,
        # противоположном вектору градиента
        w_new = w_cur - gradient_value * learning_rate

        # Проверка того, что расстояние между текущей точкой и новой не стало меньше или равно eps
        if np.linalg.norm(w_new - w_cur) <= eps:
            break

        w_cur = w_new

    return w_cur.round(2)


def logistic_regression_solve_res():
    # Загрузка набора данных для тестирования алгоритмов классификации
    iris = load_iris()

    # Приведение классов, которые необходимо научиться предсказывать, к значениям -1 и 1
    y = (iris.target > 1).astype('int64').reshape((len(iris.target), 1))
    y[y == 0] = -1

    # Создание таблицы на основе набора данных.
    # Факторы, которые есть в данных, будут называться 'x1', 'x2', 'x3' и 'x4'.
    # Классы объектов помещаются в колонку 'y'
    data = pd.DataFrame(
        columns=['x1', 'x2', 'x3', 'x4', 'y'],
        data=np.hstack([iris.data, y])
    )

    # Для предсказания будут использоваться только факторы 'x1' и 'x2'
    factor_names = ['x1', 'x2']
    # Предсказываемая характеристика — 'y'
    y_name = 'y'

    # Определение оптимальных весов для разработанной модели логистической регрессии
    ws = logistic_regression_solve_solution(data, factor_names, y_name,
                                            learning_rate=0.01, eps=0.001)
    for i in range(len(ws)):
        print(f'w{i}:', ws[i])

logistic_regression_solve_res()