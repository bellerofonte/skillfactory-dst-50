# [DST-50] Project 5
# Car price prediction model

## Цель работы:
Спрогнозировать стоимость автомобиля по его зарактеристикам, полученным из объявлений на сайте [auto.ru](https://auto.ru)

## Задачи работы:
* Освоить функционал библиотек `python` по работе с API произвольного ресурса (на примере сайта auto.ru)
* Закрепить знания по использованию алгоритмов бустинга и стекинга
* Получить удовольствие от процесса

## Исходные данные

Тестовый дата-сет `test.csv` необходимо скачать с Kaggle.com по [ссылке](https://www.kaggle.com/c/sf-dst-car-price-prediction/data?select=test.csv) или с использованием **Kaggle API**

```kaggle competitions download -c sf-dst-car-price-prediction```

## Структура проекта
* в ноутбуке `prepare_test` проводится анализ исходных данных из файла `test.csv`, их очистка от невалидных значений и дубликатов, уменьшение размерности, на выходе получается файл `test_clean.csv`
* в ноутбуке `collect_data` осуществляется выгрузка свежих объявлений с сайта auto.ru в соответствии с форматом файла `test_clean.csv`, на выходе получается файл `train.csv`
* в ноутбуке `project_5` осуществляется моделирование цен авто

Файлы `*.csv` доступны для загрузки по [ссылке](https://cloud.mail.ru/public/ACtq/CS6ZjFg2M)

## Результаты
Kaggle score сабмишна по текущей версии проекта - **`21.68184`**