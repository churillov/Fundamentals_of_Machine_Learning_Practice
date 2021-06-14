# В следующей ячейке мы импортируем библиотеки и фиксируем случайность. Значение сида(seed),
# а в нашем случае, 42, фиксирует случайность. Что это значит?
#
# После написания такой строчки, операции из ```numpy```, например, генерация датасета, будут все еще случайными,
# но для всех запускающих этот код -- одинаковыми.
#
# Пожалуйста, обращайте на него внимание во всех заданиях. Это требуется для проверки ваших решений и
# его удаление или изменение может повлечь за собой ошибки.

from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from mnist import MNIST

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Задание 1
# 1. Получите выборку с помощью следующего вызова:
# ```
# sklearn.datasets.make_moons(n_samples=1000, noise=0.5, random_state=10)

x, y = make_moons(n_samples=1000, noise=0.5, random_state=10)

# 2. Разбейте выборку на `train` и `test` с помощью функции `train_test_split`.
# Через аргументы функции зафиксируйте `random_state=10` и `test_size=0.5`.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=10)

# 3. Обучите класс `GridSearchCV` на обучающей выборке. Переберите параметр `n_neighbors` от 1 до 20.
# Чтобы получить детерменированный результат,
# передайте в параметр `cv` объект класса `KFold(n_splits=5, random_state=10)`.
# Этот класс будет отвечать за разбиение выборки во время кросс-валидации.
grid_searcher = GridSearchCV(KNeighborsClassifier(), param_grid={"n_neighbors": list(range(1, 20))}, cv=KFold(n_splits=5, random_state=10, shuffle=True))
grid_searcher.fit(x_train, y_train)
# Отправьте значение доли верных ответов(`accuracy`), которое получается,
# если применить обученный `GridSearchCV` к тестовой выбоке.
accuracy = accuracy_score(y_test, grid_searcher.predict(x_test))
print(accuracy)

# Задание 2
# Напишите функцию train_grid_search.
# Функция принимает выборку на вход.
# Она должна создать объект GridSearchCV,
# который переберет соседей от 1 до 20.
# Обучите GridSearchCV.
# Функция должна возвращать значение ключа mean_test_score
# у атрибута cv_results_ в классе GridSearchCV.
# Это поле содержит информацию о значении метрики для каждого параметра.
# В данном задании использовать KFold не требуется.
#
# Код ниже строит изображение зависимости качества от количества соседей.


def train_grid_search(X, y):
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.5, random_state=10)
    grid_searcher2 = GridSearchCV(KNeighborsClassifier(), param_grid={"n_neighbors": list(range(1, 21))})
    grid_searcher2.fit(x_train2, y_train2)
    print(len(grid_searcher2.cv_results_["mean_test_score"]))
    return grid_searcher2.cv_results_["mean_test_score"]


# аккуратно, не меняйте random_state
X, y = make_moons(n_samples=1000, noise=0.5, random_state=10)
test_score = train_grid_search(X, y)

# небольшая проверка: если не выдалась ошибка, то можете продолжать
assert (type(test_score) == np.ndarray), 'Переменная test_score должна иметь тип np.array()'
assert (len(test_score) == 20), 'Переменная test_score должна иметь 20 значений (по 1 для каждого кол-ва соседей)'


plt.plot(np.arange(1, 21), test_score)
# plt.show()
# проверяться будет переменная ```test_score```


# Задание 3

# Загрузим данные: pip install python-mnist


mndata = MNIST('', gz=True)
images, labels = mndata.load_training()

# Далее мы берем только 5000 картинок и меток, чтобы не ждать обучения слишком долго:
images, labels = np.array(images)[:5000, :], np.array(labels)[:, 5000]

# Так вы можете посмотреть на данные. Например, это картинка номер 0. Можете посмотреть на любую другую!
plt.imshow(images[0].reshape(28, 28))
plt.show()


# В этом задании вы будете работать с классическим датасетом MNIST.
# Код выше загрузил данные в переменные images и labels.
# Обучите knn с 30 ближайшими соседями.
# Предварительно разбейте выборку на train и test в соотношении 80/20
# и ```random_state=10```. Какой `accuracy` вы получаете на тестовой выборке?
# Ответ округлите до 3 знаков после запятой.


x_train3, x_test3, y_train3, y_test3 = train_test_split(images, labels, test_size=0.2, random_state=10)
knn = KNeighborsClassifier(n_neighbors=30)

knn.fit(x_train3, y_train3)

accuracy = accuracy(y_test3, knn.predict(x_test3))
print(accuracy)
