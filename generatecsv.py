import pandas as pd
import numpy as np

# Создаем случайные данные для курса доллара к рублю за 2020 год
np.random.seed(0)  # Для воспроизводимости результатов
num_days = 366  # В 2020 году был високосный год
exchange_rate = np.random.uniform(low=60, high=80, size=num_days)  # Произвольные значения в диапазоне от 60 до 80

# Создаем фрейм данных
data = pd.DataFrame({'ExchangeRate': exchange_rate})

# Сохраняем данные в CSV файл
data.to_csv('exchange_rate.csv', index=False)
