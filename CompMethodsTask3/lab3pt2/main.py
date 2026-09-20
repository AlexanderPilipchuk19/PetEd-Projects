import numpy as np
import scipy.linalg
from PIL import Image
import matplotlib.pyplot as plt


def maximize_svd_difference():
    W, H = 800, 533
    filename = "low_rank_gradient.bmp"

    # Создание изображения низкого ранга (Rank = 1)
    col_vec = np.linspace(0, 255, H).reshape(-1, 1)
    row_vec = np.linspace(0, 1, W).reshape(1, -1)  

    matrix_perfect = np.dot(col_vec, row_vec)

    # Приводим к uint8 и сохраняем (это добавляет шум квантования, ранг чуть вырастет, но останется малым)
    img_data = matrix_perfect.astype(np.uint8)
    Image.fromarray(img_data, mode='L').save(filename)

    matrix_process = np.asarray(Image.open(filename), dtype=np.float64)

    # Вычисление SVD разными методами
    print("Вычисляем Numpy SVD (обычно 'gesdd')...")
    # Numpy возвращает по убыванию
    s_np_raw = np.linalg.svd(matrix_process, compute_uv=False)

    print("Вычисляем Scipy SVD (принудительно 'gesvd')...")
    # Scipy возвращает по убыванию
    s_sp_raw = scipy.linalg.svd(matrix_process, compute_uv=False, lapack_driver='gesvd')

    # Реализация заданной метрики

    # Сортировка по возрастанию
    a = np.sort(s_np_raw)
    b = np.sort(s_sp_raw)

    # Составление вектора c
    # c_j = max(a/b, b/a). Обработка деления на 0.

    # Используем masked arrays или np.where для безопасности
    # Если оба 0 -> 0
    # Если один 0 -> 0 (по условию)

    # Чтобы избежать RuntimeWarning при делении на очень малые числа/нули:
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_1 = np.divide(a, b)
        ratio_2 = np.divide(b, a)

        # Заменяем inf и nan на 0
        ratio_1 = np.nan_to_num(ratio_1, posinf=0.0, neginf=0.0)
        ratio_2 = np.nan_to_num(ratio_2, posinf=0.0, neginf=0.0)

        c = np.maximum(ratio_1, ratio_2)

    # Оценка L2 нормы
    l2_norm = np.linalg.norm(c)

    print("\n" + "=" * 40)
    print(f"РЕЗУЛЬТАТЫ (L2 норма вектора частных)")
    print("=" * 40)
    print(f"Размер изображения: {W}x{H}")
    print(f"Тип изображения:    Градиент (Low Rank)")
    print("-" * 40)
    print(f"Значение метрики (чем больше, тем лучше): {l2_norm:.4f}")
    print("-" * 40)

    ideal_norm = np.sqrt(min(H, W))
    print(f"(Для сравнения: при полном совпадении норма была бы ≈ {ideal_norm:.2f})")

    # Визуализация вектора C
    plt.figure(figsize=(12, 5))
    plt.plot(c, label='Element-wise Ratio max(a/b, b/a)')
    plt.yscale('log')
    plt.title("Значения вектора C (логарифмическая шкала)")
    plt.xlabel("Индекс (после сортировки по возрастанию)")
    plt.ylabel("Отношение значений")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    maximize_svd_difference()