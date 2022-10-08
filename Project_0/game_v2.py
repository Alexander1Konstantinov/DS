import numpy as np


def random_predict(number: int = 1) -> int:
    count = 0
    begin = 1
    end = 101
    while True:
        count += 1
        predict_number = np.random.randint(begin, end)
        if number == predict_number:
            break
        elif predict_number < number:
            begin = predict_number
        elif predict_number > number:
            end = predict_number
    return count


def score_game(random_predict) -> int:
    count_ls = []
    # np.random.seed(1)  # фиксируем сид для воспроизводимости
    random_array = np.random.randint(
        1, 101, size=(1000))  # загадали список чисел

    for number in random_array:
        count_ls.append(random_predict(number))

    score = int(np.mean(count_ls))
    print(f"Ваш алгоритм угадывает число в среднем за:{score} попыток")
    return score


if __name__ == "__main__":
    # RUN
    score_game(random_predict)
