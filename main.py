from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import random
import time
from pathlib import Path
from typing import Sequence

# ---------------------------------------------------------
# Глобальные переменные процесса-воркера
# ---------------------------------------------------------
# Каждый процесс из пула один раз получает матрицы A и B
# через initializer. Благодаря этому в каждую задачу мы
# передаем только координаты элемента (i, j), а не обе
# матрицы целиком.
WORKER_A: list[list[float]] = []
WORKER_B: list[list[float]] = []


def init_worker(matrix_a: list[list[float]], matrix_b: list[list[float]]) -> None:
    """
    Инициализирует глобальные переменные внутри дочернего процесса.
    """
    global WORKER_A, WORKER_B
    WORKER_A = matrix_a
    WORKER_B = matrix_b


def parse_args() -> argparse.Namespace:
    """
    Разбирает аргументы командной строки.

    В программе два режима:
    1) files  - читаем матрицы из файлов и умножаем;
    2) stream - дополнительное задание:
                один процесс генерирует матрицы,
                другой умножает их по мере появления.
    """
    parser = argparse.ArgumentParser(
        description="Лабораторная работа по параллельному программированию на Python 3.10"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ---------- Основной режим ----------
    files_parser = subparsers.add_parser(
        "files",
        help="Считать две матрицы из файлов и перемножить их параллельно",
    )
    files_parser.add_argument("--a", required=True, help="Путь к первой матрице")
    files_parser.add_argument("--b", required=True, help="Путь ко второй матрице")
    files_parser.add_argument(
        "--out",
        default="result_matrix.txt",
        help="Файл для итоговой матрицы",
    )
    files_parser.add_argument(
        "--tmp",
        default="intermediate_matrix.txt",
        help="Промежуточный файл, обновляемый сразу после вычисления элемента",
    )
    files_parser.add_argument(
        "--processes",
        type=int,
        default=4,
        help="Фиксированное количество процессов",
    )
    files_parser.add_argument(
        "--auto-workers",
        action="store_true",
        help="Автоматически подобрать количество процессов",
    )

    # ---------- Дополнительный режим ----------
    stream_parser = subparsers.add_parser(
        "stream",
        help="Асинхронная демонстрация: генератор матриц + процесс умножения",
    )
    stream_parser.add_argument(
        "--size",
        type=int,
        default=3,
        help="Размер квадратных матриц NxN",
    )
    stream_parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Сколько пар матриц сгенерировать",
    )
    stream_parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Пауза между генерацией очередной пары матриц в секундах",
    )
    stream_parser.add_argument(
        "--output-dir",
        default="stream_output",
        help="Папка для результатов потоковой обработки",
    )
    stream_parser.add_argument(
        "--processes",
        type=int,
        default=4,
        help="Фиксированное количество процессов для умножения",
    )
    stream_parser.add_argument(
        "--auto-workers",
        action="store_true",
        help="Автоматически подобрать количество процессов",
    )

    return parser.parse_args()


def read_matrix_from_file(file_path: str | Path) -> list[list[float]]:
    """
    Читает матрицу из текстового файла.

    Формат файла:
    каждая строка = одна строка матрицы,
    числа разделены пробелами.

    Пример:
    1 2 3
    4 5 6
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    matrix: list[list[float]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()

            # Пустые строки просто пропускаем.
            if not line:
                continue

            try:
                row = [float(value) for value in line.split()]
            except ValueError as error:
                raise ValueError(
                    f"Ошибка чтения числа в файле {path}, строка {line_number}: {raw_line!r}"
                ) from error

            matrix.append(row)

    if not matrix:
        raise ValueError(f"Файл {path} пустой или не содержит матрицу")

    # Проверяем, что у всех строк одинаковая длина.
    row_length = len(matrix[0])
    for row_index, row in enumerate(matrix, start=1):
        if len(row) != row_length:
            raise ValueError(
                f"Некорректная матрица в файле {path}: "
                f"строка {row_index} имеет длину {len(row)}, ожидалось {row_length}"
            )

    return matrix


def format_number(value: float | None) -> str:
    """
    Красиво форматирует число для записи в файл.

    None используется для промежуточной матрицы, где
    часть элементов еще не вычислена.
    """
    if value is None:
        return "..."

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value)


def write_matrix_to_file(
    file_path: str | Path,
    matrix: Sequence[Sequence[float | None]],
) -> None:
    """
    Записывает матрицу в текстовый файл.

    Если элемент еще не вычислен, на его месте будет '...'.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in matrix:
            file.write(" ".join(format_number(value) for value in row) + "\n")


def validate_matrices_for_multiplication(
    matrix_a: Sequence[Sequence[float]],
    matrix_b: Sequence[Sequence[float]],
) -> None:
    """
    Проверяет возможность классического умножения матриц.

    Для A x B должно выполняться:
    количество столбцов A == количеству строк B
    """
    if not matrix_a or not matrix_b:
        raise ValueError("Одна из матриц пустая")

    cols_a = len(matrix_a[0])
    rows_b = len(matrix_b)

    if cols_a != rows_b:
        raise ValueError(
            "Матрицы нельзя умножить: "
            f"у A столбцов {cols_a}, а у B строк {rows_b}"
        )


def choose_process_count(task_count: int, fixed_processes: int, auto_workers: bool) -> int:
    """
    Выбирает количество процессов для пула.

    Если включен auto_workers:
    - берем минимум из числа ядер CPU и числа задач.
    Иначе:
    - берем фиксированное значение.
    """
    if task_count <= 0:
        return 1

    if auto_workers:
        return max(1, min(mp.cpu_count(), task_count))

    return max(1, fixed_processes)


def compute_element(index: tuple[int, int]) -> tuple[int, int, float]:
    """
    Вычисляет один элемент результирующей матрицы C = A x B.

    index = (i, j)
    где:
    - i это номер строки,
    - j это номер столбца.
    """
    i, j = index
    result = 0.0

    # Общая размерность: число столбцов A = число строк B.
    common_size = len(WORKER_A[0])

    for k in range(common_size):
        result += WORKER_A[i][k] * WORKER_B[k][j]

    return i, j, result


def multiply_matrices_parallel(
    matrix_a: list[list[float]],
    matrix_b: list[list[float]],
    processes: int = 4,
    auto_workers: bool = False,
    intermediate_file: str | Path | None = None,
) -> list[list[float]]:
    """
    Параллельно перемножает две матрицы через multiprocessing.Pool.

    Алгоритм:
    1. Проверяем размеры.
    2. Создаем задачи для каждого элемента результата.
    3. Пул процессов считает элементы независимо.
    4. Главный процесс собирает ответы и сразу обновляет промежуточный файл.
    """
    validate_matrices_for_multiplication(matrix_a, matrix_b)

    rows_a = len(matrix_a)
    cols_b = len(matrix_b[0])

    # Пока результат не готов полностью, храним None.
    result: list[list[float | None]] = [
        [None for _ in range(cols_b)]
        for _ in range(rows_a)
    ]

    # Каждая задача — это вычисление одного элемента C[i][j].
    tasks = [(i, j) for i in range(rows_a) for j in range(cols_b)]

    worker_count = choose_process_count(
        task_count=len(tasks),
        fixed_processes=processes,
        auto_workers=auto_workers,
    )

    print(f"[INFO] Запускаем пул процессов: {worker_count}", flush=True)

    # Сразу создадим промежуточный файл с "пустой" матрицей.
    if intermediate_file is not None:
        write_matrix_to_file(intermediate_file, result)

    with mp.Pool(
        processes=worker_count,
        initializer=init_worker,
        initargs=(matrix_a, matrix_b),
    ) as pool:
        # imap_unordered возвращает результаты по мере готовности.
        for i, j, value in pool.imap_unordered(compute_element, tasks):
            result[i][j] = value

            # По условию задания обновляем промежуточный файл
            # сразу после вычисления очередного элемента.
            if intermediate_file is not None:
                write_matrix_to_file(intermediate_file, result)

            print(f"[READY] Элемент C[{i}][{j}] = {format_number(value)}", flush=True)

    # К этому моменту None уже не осталось.
    final_result = [[float(value) for value in row] for row in result]
    return final_result


def multiply_from_files(
    matrix_a_path: str | Path,
    matrix_b_path: str | Path,
    result_path: str | Path,
    intermediate_path: str | Path,
    processes: int = 4,
    auto_workers: bool = False,
) -> list[list[float]]:
    """
    Полный сценарий основного задания:
    - читаем матрицы из файлов,
    - параллельно умножаем,
    - сохраняем промежуточный и итоговый файлы.
    """
    matrix_a = read_matrix_from_file(matrix_a_path)
    matrix_b = read_matrix_from_file(matrix_b_path)

    result = multiply_matrices_parallel(
        matrix_a=matrix_a,
        matrix_b=matrix_b,
        processes=processes,
        auto_workers=auto_workers,
        intermediate_file=intermediate_path,
    )

    write_matrix_to_file(result_path, result)

    print(f"[OK] Итоговая матрица сохранена в: {result_path}", flush=True)
    print(
        f"[OK] Промежуточный файл обновлялся по ходу вычислений: {intermediate_path}",
        flush=True,
    )

    return result


def generate_random_square_matrix(
    size: int,
    min_value: int = 0,
    max_value: int = 9,
) -> list[list[int]]:
    """
    Генерирует случайную квадратную матрицу size x size.
    """
    return [
        [random.randint(min_value, max_value) for _ in range(size)]
        for _ in range(size)
    ]


def matrix_generator_process(
    task_queue: mp.Queue,
    stop_event: mp.Event,
    size: int,
    count: int,
    delay: float,
) -> None:
    """
    Процесс-генератор для дополнительного задания.

    Он:
    - создает случайные квадратные матрицы;
    - складывает пары матриц в очередь;
    - умеет останавливаться по stop_event.
    """
    try:
        for batch_number in range(1, count + 1):
            if stop_event.is_set():
                print("[GEN] Получен сигнал остановки. Генерация прекращена.", flush=True)
                break

            matrix_a = generate_random_square_matrix(size)
            matrix_b = generate_random_square_matrix(size)

            task_queue.put((batch_number, matrix_a, matrix_b))
            print(f"[GEN] Сгенерирована пара матриц #{batch_number}", flush=True)

            time.sleep(delay)
    finally:
        # None — специальный сигнал для потребителя:
        # новых задач больше не будет.
        task_queue.put(None)
        print("[GEN] Генератор завершил работу.", flush=True)


def matrix_consumer_process(
    task_queue: mp.Queue,
    stop_event: mp.Event,
    output_dir: str | Path,
    processes: int,
    auto_workers: bool,
) -> None:
    """
    Процесс-потребитель для дополнительного задания.

    Он:
    - получает пары матриц из очереди;
    - умножает их;
    - сохраняет результат в отдельные файлы.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    while True:
        if stop_event.is_set():
            print("[CONSUMER] Получен сигнал остановки. Завершение.", flush=True)
            break

        try:
            item = task_queue.get(timeout=0.5)
        except queue.Empty:
            # Очередь пока пуста — продолжаем ждать.
            continue

        if item is None:
            print("[CONSUMER] Получен сигнал, что задач больше не будет.", flush=True)
            break

        batch_number, matrix_a, matrix_b = item
        print(f"[CONSUMER] Начинаю обработку пары #{batch_number}", flush=True)

        result_file = output_path / f"result_{batch_number}.txt"
        intermediate_file = output_path / f"intermediate_{batch_number}.txt"

        result = multiply_matrices_parallel(
            matrix_a=matrix_a,
            matrix_b=matrix_b,
            processes=processes,
            auto_workers=auto_workers,
            intermediate_file=intermediate_file,
        )

        write_matrix_to_file(result_file, result)
        print(
            f"[CONSUMER] Пара #{batch_number} обработана. Результат: {result_file}",
            flush=True,
        )

    print("[CONSUMER] Потребитель завершил работу.", flush=True)


def run_stream_demo(
    size: int,
    count: int,
    delay: float,
    output_dir: str | Path,
    processes: int,
    auto_workers: bool,
) -> None:
    """
    Запускает дополнительное задание:
    - один процесс генерирует матрицы;
    - второй процесс их умножает;
    - главный процесс может остановить все по Ctrl+C.
    """
    task_queue: mp.Queue = mp.Queue(maxsize=10)
    stop_event = mp.Event()

    generator = mp.Process(
        target=matrix_generator_process,
        args=(task_queue, stop_event, size, count, delay),
        name="MatrixGenerator",
    )

    consumer = mp.Process(
        target=matrix_consumer_process,
        args=(task_queue, stop_event, output_dir, processes, auto_workers),
        name="MatrixConsumer",
    )

    generator.start()
    consumer.start()

    print("[MAIN] Асинхронная обработка запущена.", flush=True)
    print("[MAIN] Нажмите Ctrl+C, чтобы остановить процессы вручную.", flush=True)

    try:
        generator.join()
        consumer.join()
    except KeyboardInterrupt:
        print("\n[MAIN] Получен Ctrl+C. Останавливаем процессы...", flush=True)
        stop_event.set()

        generator.join()
        consumer.join()

    print("[MAIN] Дополнительное задание завершено.", flush=True)


def main() -> None:
    """
    Главная точка входа в программу.
    """
    args = parse_args()

    if args.mode == "files":
        multiply_from_files(
            matrix_a_path=args.a,
            matrix_b_path=args.b,
            result_path=args.out,
            intermediate_path=args.tmp,
            processes=args.processes,
            auto_workers=args.auto_workers,
        )

    elif args.mode == "stream":
        run_stream_demo(
            size=args.size,
            count=args.count,
            delay=args.delay,
            output_dir=args.output_dir,
            processes=args.processes,
            auto_workers=args.auto_workers,
        )


if __name__ == "__main__":
    # Полезно для Windows и для некоторых сценариев запуска frozen-приложений.
    mp.freeze_support()
    main()
