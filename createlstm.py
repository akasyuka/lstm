import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Загрузка данных
def load_data(filename, seq_len, split):
    data = pd.read_csv(filename)
    data = data.dropna()  
    data = data.values  
    data = data[::-1]  
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    result = np.array(result)
    row = round(split * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1][:,-1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,-1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test, scaler]

# Загрузка данных
X_train, y_train, X_test, y_test, scaler = load_data('exchange_rate.csv', 50, 0.8)

# Определение формы входных данных
input_shape = (X_train.shape[1], X_train.shape[2]) 

# Гиперпараметры модели
units = 50  
dropout = 0.2  
epochs = 100  
batch_size = 512  

# Создание LSTM модели
model = Sequential([
    LSTM(units=units, input_shape=input_shape, return_sequences=True),
    Dropout(dropout),
    LSTM(units=units, return_sequences=False),
    Dropout(dropout),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

# Создать модель
#model.save('currency_prediction_model.h5')
# Предсказание
predictions = model.predict(X_test)

# Обратное преобразование (обратная операция нормализации) для предсказанных значений
predicted_exchange_rate = scaler.inverse_transform(predictions)

# Вычисляем медианное значение предсказаний
median_predicted_exchange_rate = np.median(predicted_exchange_rate)

# Выводим медианное значение
print("Median predicted exchange rate:", median_predicted_exchange_rate)