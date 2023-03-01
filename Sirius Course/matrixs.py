import numpy


def round_to_2(x):
    return round(x, 2)

def get_col(matrix, i):
    if type(matrix[0]) is int:
        return matrix
    return [col[i] for col in matrix]


def skal(v1, v2):
    return round_to_2(sum([float(v1[i] * v2[i]) for i in range(len(v1))]))


def l(v1):
    return skal(v1, v1)


def dec(v1, v2):
    return [v1[i] - v2[i] for i in range(len(v1))]


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


def matrix_multiplication_solution(matrix_a, matrix_b):
    n1, m1 = len(matrix_a), len(matrix_a[0])
    n2, m2 = len(matrix_b), 1 if type(matrix_b[0]) is not type([]) else len(matrix_b[0])
    if n2 != m1:
        return -1
    if type(matrix_b[0]) is not type([]):
        result = [0 * m2 for i in range(n1)]
        for i in range(n1):
            for j in range(m2):
                v1 = matrix_a[i]
                value = skal(v1, matrix_b)
                result[i] = value
    else:
        result = [[0] * m2 for i in range(n1)]
        for i in range(n1):
            for j in range(m2):
                v2 = get_col(matrix_b, j)
                v1 = matrix_a[i]
                value = skal(v1, v2)
                result[i][j] = value
    return result


