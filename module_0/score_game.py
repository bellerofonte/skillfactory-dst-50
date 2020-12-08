import numpy as np

def game_core_binary(number, min, max):
    '''
    Функция принимает загаданное число, нижнюю и верхнюю границы угадывания 
    Функия возвращает число попыток
    Используем бинарный поиск (метод деления пополам) для оптимизации угадывания.
    '''
    count = 0
    ''' 
    инициализируем границы диапазона поиска
    нижняя граница должна быть на 1 меньше, а верхняя граница должна быть на 1 больше, 
    т.к. данная реализация бинарного поиска ищет заданное число в диапазоне (min, max)
    т.е. начальные границы не входят в зону поиска
    '''
    min -= 1
    max += 1
    while True:
        count += 1
        predict = int(0.5 * (min+max)) # предполагаемое число - среднее текущих границ
        if number > predict:
            min = predict # сужаем границу поиска снизу
        elif number < predict:
            max = predict # сужаем границу поиска сверху
        else:
            return count # выход из цикла, если угадали
        
        
def score_game(game_core, min=1, max=100, size=1000):
    '''Запускаем игру 1000 раз, чтобы узнать, как быстро игра угадывает число'''
    count_ls = []
    np.random.seed(1)  # фиксируем RANDOM SEED, чтобы ваш эксперимент был воспроизводим!
    random_array = np.random.randint(min, max+1, size) 
    for number in random_array:
        count_ls.append(game_core(number, min, max))
    score = int(np.mean(count_ls))
    print(f"Ваш алгоритм угадывает число в среднем за {score} попыток")
    return(score)


# запускаем
score_game(game_core_binary)
