
import numpy as np

from optimize_intervals import (
    build_pseudo_source,
    optimize_interval_pair,
    normalize
)


def grid_check(A0, B0, interval0,
               A1, B1, interval1, num=201):
    """
    Независимый полный перебор по паре интервалов.
    """

    Q0 = build_pseudo_source(A0, B0, interval0)
    Q1 = build_pseudo_source(A1, B1, interval1)

    e0 = normalize(B0 - A0)
    e1 = normalize(B1 - A1)

    ts = np.linspace(
        interval0["start"], interval0["stop"], num
    )
    us = np.linspace(
        interval1["start"], interval1["stop"], num
    )

    P0 = A0 + ts[:, None] * e0
    P1 = A1 + us[:, None] * e1

    lengths = (
        np.linalg.norm(P0 - Q0, axis=1)[:, None]
        + np.linalg.norm(
            P0[:, None, :] - P1[None, :, :],
            axis=2
        )
        + np.linalg.norm(P1 - Q1, axis=1)[None, :]
        + interval0["d"]
        + interval1["d"]
    )

    i, j = np.unravel_index(
        np.argmin(lengths), lengths.shape
    )

    return float(lengths[i, j]), ts[i], us[j]


def random_interval(rng, edge_length):
    """
    Генерирует случайный интервал MMP
    с допустимыми геометрическими параметрами.
    """

    bounds = np.sort(
        rng.uniform(0, edge_length, size=2)
    )

    return {
        "start": float(bounds[0]),
        "stop": float(bounds[1]),
        "d": float(rng.uniform(0, 2)),
        "pseudo_x": float(
            rng.uniform(-1, edge_length + 1)
        ),
        "pseudo_y": float(
            -rng.uniform(0.05, 2.5)
        )
    }


def run_tests(num_tests=30):
    rng = np.random.default_rng(42)

    max_error = 0.0
    failures = 0
    interior_both = 0
    boundary_first = 0
    boundary_second = 0
    boundary_both = 0

    for test_id in range(num_tests):

        # Два случайных ребра в 3D
        A0 = rng.uniform(-2, 2, size=3)
        B0 = A0 + normalize(
            rng.normal(size=3)
        ) * rng.uniform(1, 3)

        A1 = rng.uniform(3, 7, size=3)
        B1 = A1 + normalize(
            rng.normal(size=3)
        ) * rng.uniform(1, 3)

        interval0 = random_interval(
            rng, np.linalg.norm(B0 - A0)
        )
        interval1 = random_interval(
            rng, np.linalg.norm(B1 - A1)
        )

        # Наш алгоритм с бисекцией
        result = optimize_interval_pair(
            A0, B0, interval0,
            A1, B1, interval1
        )

        # Полный перебор
        grid_length, grid_t, grid_u = grid_check(
            A0, B0, interval0,
            A1, B1, interval1
        )

        # Наш результат может быть лучше сетки
        error = result["length"] - grid_length
        max_error = max(max_error, error)

        # Проверяем принадлежность точек интервалам
        e0 = normalize(B0 - A0)
        e1 = normalize(B1 - A1)

        t = np.dot(result["P0"] - A0, e0)
        u = np.dot(result["P1"] - A1, e1)
        
        
        # Определяем, лежат ли точки внутри интервалов
        tol = 1e-6

        inside0 = (
            interval0["start"] + tol < t
            < interval0["stop"] - tol
        )

        inside1 = (
            interval1["start"] + tol < u
            < interval1["stop"] - tol
        )

        if inside0 and inside1:
            interior_both += 1
        elif not inside0 and not inside1:
            boundary_both += 1
        elif not inside0:
            boundary_first += 1
        else:
            boundary_second += 1

        
        valid = (
            interval0["start"] - 1e-7 <= t
            <= interval0["stop"] + 1e-7
            and interval1["start"] - 1e-7 <= u
            <= interval1["stop"] + 1e-7
        )

        passed = error <= 1e-7 and valid

        if not passed:
            failures += 1

        status = "OK" if passed else "FAIL"

        print(
            f"Тест {test_id + 1:2d}: {status} | "
            f"бисекция={result['length']:.8f} | "
            f"сетка={grid_length:.8f} | "
            f"разница={error:+.2e}"
        )

    print("\n=== ИТОГИ ===")
    print("Количество тестов:", num_tests)
    print("Ошибок:", failures)
    print("Максимальное превышение сетки:", max_error)
    
    print("\n=== РАСПОЛОЖЕНИЕ МИНИМУМОВ ===")
    print("Внутри обоих интервалов:", interior_both)
    print("На границе первого:", boundary_first)
    print("На границе второго:", boundary_second)
    print("На границах обоих:", boundary_both)

    
def test_interior_minimum(num_tests=100):
    """
    Проверяем конфигурации, в которых
    минимум расположен внутри обоих интервалов.
    """
    rng = np.random.default_rng(123)

    failures = 0
    max_error = 0.0
    interior_count = 0

    for test_id in range(num_tests):

        # Два параллельных ребра
        A0 = np.array([0.0, 0.0, 0.0])
        B0 = np.array([2.0, 0.0, 0.0])

        A1 = np.array([0.0, 2.0, 0.0])
        B1 = np.array([2.0, 2.0, 0.0])

        # Случайное положение оптимума
        shift = rng.uniform(0.4, 1.6)

        # Псевдоисточники с одинаковой
        # горизонтальной координатой
        interval0 = {
            "start": 0.0,
            "stop": 2.0,
            "d": 0.0,
            "pseudo_x": shift,
            "pseudo_y": -1.0
        }

        interval1 = {
            "start": 0.0,
            "stop": 2.0,
            "d": 0.0,
            "pseudo_x": shift,
            "pseudo_y": -1.0
        }

        result = optimize_interval_pair(
            A0, B0, interval0,
            A1, B1, interval1
        )

        grid_length, _, _ = grid_check(
            A0, B0, interval0,
            A1, B1, interval1,
            num=201
        )

        e0 = normalize(B0 - A0)
        e1 = normalize(B1 - A1)

        t = np.dot(result["P0"] - A0, e0)
        u = np.dot(result["P1"] - A1, e1)

        inside = (
            1e-6 < t < 2.0 - 1e-6
            and 1e-6 < u < 2.0 - 1e-6
        )

        if inside:
            interior_count += 1

        error = result["length"] - grid_length
        max_error = max(max_error, error)

        passed = inside and error <= 1e-7

        if not passed:
            failures += 1

        print(
            f"Тест {test_id + 1:3d}: "
            f"{'OK' if passed else 'FAIL'} | "
            f"t={t:.6f}, u={u:.6f} | "
            f"разница={error:+.2e}"
        )

    print("\n=== ИТОГИ ВНУТРЕННИХ ТЕСТОВ ===")
    print("Всего тестов:", num_tests)
    print("Внутренних минимумов:", interior_count)
    print("Ошибок:", failures)
    print("Максимальное превышение сетки:", max_error)

    
if __name__ == "__main__":
    test_interior_minimum(100)
