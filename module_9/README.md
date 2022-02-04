# Project 8. Car Price prediction

<img src="https://whatcar.vn/media/2018/09/car-lot-940x470.jpg"/>

## Цель работы
прогнозирование стоимости автомобиля по характеристикам

## Задачи работы
применить на практике навыки работы с:
* моделями NLP (natural language processing)
* multi-input нейронными сетями
* CNN для анализа изображений
* моделей ML
* ансамблированием моделей


## tl;dr

В таблице ниже собраны результаты всех моих попыток решения задачи прогнозирования цены:

Model | val_MAPE | kaggle score
:-----|---------------------|--------------
Catboost | 12.81 | 13.22920 
Tabular NN | 11.65 | 11.75849 
TNN + NLP | 11.72 | 11.77384 
TNN + NLP + CNN | 11.95 | 11.81828 
Blend | 🤷‍♂️ | **11.64974** 