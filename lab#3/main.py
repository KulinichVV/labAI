import numpy as np

from keras.layers import Input, Dense
from keras import Sequential, Model
from keras.losses import sparse_categorical_crossentropy
from keras.metrics import sparse_categorical_accuracy
from keras.optimizers import Adam

import pandas as pd

# Считывание тренировочных наборов данных
train = pd.read_csv("mnist_train.csv").values
Y_train = train[:, 0]
X_train = train[:, 1:]

# Считывание тестовых наборов данных
test = pd.read_csv("mnist_test.csv").values
Y_test = test[:, 0]
X_test = test[:, 1:]

# Нормализация входных данных [0, 1]
X_train, X_test = (X_train/255.0), (X_test/255.0)

# Транспонирование (чтоб данные находились в столбцах)
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Создание слоев нейросети
x = Input(shape=(784,))
h1 = Dense(64, activation="relu")(x)
h2 = Dense(64, activation="relu")(h1)
h3 = Dense(64, activation="relu")(h2)
out = Dense(10, activation="softmax")(h3)
# Создание модели нейронной сети
model = Model(inputs=x, outputs=out)

# Оптимизация
opt = Adam(learning_rate=0.001)

# Компиляция модели
model.compile(
    optimizer=opt,
    loss=sparse_categorical_crossentropy,
    metrics=[sparse_categorical_accuracy],
)

bs = 64
n_epoch = 10

# Обучение модели
model.fit(
    X_train,
    Y_train,
    batch_size=bs,
    epochs=n_epoch,
    validation_data=(X_test, Y_test),
)

# Проверка модели на тестовых данных
pdc = model.predict(X_test)

correct_predictions = 0
total_num = len(Y_test)

for real, predicted in zip(Y_test, model.predict(X_test)):
    max_index = np.argmax(predicted)

    if real == max_index:
        print("Величина {} была распознана как {}".format(real, max_index))
        # Подсчет корректных предсказаний
        correct_predictions += 1
    else:
        print("Величина {} была неверно распознана как {}".format(real, max_index))

# Вычисление точности распознавания
accuracy = correct_predictions / total_num
print("Точность распознавания равна {:.2%}".format(accuracy))
