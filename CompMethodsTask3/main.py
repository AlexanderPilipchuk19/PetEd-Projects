import numpy as np
import struct
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt


class SVDCompressor:
    
    SIGNATURE = b'SVDX'
    HEADER_FORMAT = '<4sIII' 
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    FLOAT_SIZE = 4 

    @staticmethod
    def compress(input_path: str, output_path: str, n_ratio: float):
        """Сжимает изображение в .svd формат с коэффициентом сжатия N."""

        #Загрузка и конвертация в ЧБ
        try:
            img = Image.open(input_path).convert('L')
        except Exception as e:
            print(f"Ошибка при открытии файла: {e}")
            return False

        img_matrix = np.asarray(img, dtype=np.float32)
        h, w = img_matrix.shape

        # Размер исходных данных
        original_size = h * w
        target_size = original_size / n_ratio

        # SVD Разложение
        U, S, Vt = np.linalg.svd(img_matrix, full_matrices=False)

        # Вычисление ранга k
        # Формула: Header + k*(H + 1 + W)*4 
        available_bytes = target_size - SVDCompressor.HEADER_SIZE
        bytes_per_rank_unit = (h + w + 1) * SVDCompressor.FLOAT_SIZE

        k = int(available_bytes // bytes_per_rank_unit)

        # Защита: ранг не может быть меньше 1 и больше реального количества сингулярных чисел
        k = max(1, min(k, len(S)))

        print(
            f"   [Сжатие N={n_ratio}] Исходный: {original_size / 1024:.1f}KB -> Цель: {target_size / 1024:.1f}KB. Выбран ранг k={k}")

        # Усечение матриц
        U_k = U[:, :k].astype(np.float32)
        S_k = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)

        with open(output_path, 'wb') as f:
            header = struct.pack(SVDCompressor.HEADER_FORMAT,SVDCompressor.SIGNATURE, h, w, k)
            f.write(header)
            f.write(U_k.tobytes())
            f.write(S_k.tobytes())
            f.write(Vt_k.tobytes())

        return True

    @staticmethod
    def decompress(input_path: str, output_path: str):
        """Восстанавливает изображение из .svd файла."""
        if not os.path.exists(input_path):
            return None

        with open(input_path, 'rb') as f:
            header_data = f.read(SVDCompressor.HEADER_SIZE)
            sig, h, w, k = struct.unpack(SVDCompressor.HEADER_FORMAT, header_data)

            if sig != SVDCompressor.SIGNATURE:
                raise ValueError("Неизвестный формат файла!")

            buffer_u = f.read(h * k * SVDCompressor.FLOAT_SIZE)
            buffer_s = f.read(k * SVDCompressor.FLOAT_SIZE)
            buffer_vt = f.read(k * w * SVDCompressor.FLOAT_SIZE)

            U_k = np.frombuffer(buffer_u, dtype=np.float32).reshape((h, k))
            S_k = np.frombuffer(buffer_s, dtype=np.float32)
            Vt_k = np.frombuffer(buffer_vt, dtype=np.float32).reshape((k, w))

        # Восстановление A = U * S * Vt
        reconstructed = np.dot(U_k, np.dot(np.diag(S_k), Vt_k))
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

        img_out = Image.fromarray(reconstructed)
        img_out.save(output_path)
        return img_out


def get_user_file_path():
    print("\n" + "=" * 60)
    print("SVD Image Compressor")
    print("=" * 60)

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        print("введите путь к BMP изображению.")
        path = input("Путь к файлу > ")

    path = path.strip().strip('"').strip("'")

    if not os.path.exists(path):
        print(f"\nОШИБКА: Файл не найден: {path}")
        return None

    return path


def main():
    input_file = get_user_file_path()
    if not input_file:
        return

    base_dir = os.path.dirname(input_file)
    filename = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = os.path.join(base_dir, f"{filename}_svd_results")
    os.makedirs(output_dir, exist_ok=True)

    n_values = [2, 4, 8]
    results = []

    print(f"\nРезультаты будут сохранены в папку: {output_dir}\n")

    for n in n_values:
        compressed_name = os.path.join(output_dir, f"compressed_n{n}.svd")
        restored_name = os.path.join(output_dir, f"restored_n{n}.bmp")

        # Сжатие
        success = SVDCompressor.compress(input_file, compressed_name, n)
        if not success:
            continue

        # Восстановление
        restored_img = SVDCompressor.decompress(compressed_name, restored_name)
        results.append((n, restored_img))

        # Статистика
        svd_size = os.path.getsize(compressed_name)
        print(f"   -> Файл сохранен: {os.path.basename(compressed_name)} ({svd_size} байт)")

    print("\nВсе операции завершены. Построение графика...")

    # Визуализация
    try:
        orig_img = Image.open(input_file).convert('L')
        plt.figure(figsize=(14, 5))

        # Оригинал
        plt.subplot(1, 4, 1)
        plt.title("Оригинал")
        plt.imshow(orig_img, cmap='gray')
        plt.axis('off')

        # Сжатые версии
        for i, (n, img) in enumerate(results):
            plt.subplot(1, 4, i + 2)
            plt.title(f"Сжатие в {n} раз")
            plt.imshow(img, cmap='gray')
            plt.axis('off')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Не удалось построить график (возможно нет модуля matplotlib): {e}")


if __name__ == "__main__":
    main()