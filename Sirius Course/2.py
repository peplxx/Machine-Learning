from sklearn.datasets import load_iris
from sklearn.l

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
    ws =
    ws = logistic_regression_solve_solution(data, factor_names, y_name,
                                            learning_rate=0.01, eps=0.001)

    for i in range(len(ws)):
        print(f'w{i}:', ws[i])