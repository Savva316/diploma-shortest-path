
import numpy as np


EPS = 1e-12


def normalize(v):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)

    if norm < EPS:
        raise ValueError("Нельзя нормировать нулевой вектор.")

    return v / norm




def build_pseudo_source(A, B, interval):
    """
    Строит 3D-псевдоисточник для интервала MMP.

    A, B — координаты концов ребра.
    interval — словарь из get_edge_intervals().

    Возвращает координаты Q в 3D.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    # Единичный вектор вдоль ребра
    e = B - A
    e = e / np.linalg.norm(e)

    # Выбираем вектор, не параллельный ребру
    axis = np.array([0.0, 0.0, 1.0])

    if abs(np.dot(e, axis)) > 0.9:
        axis = np.array([0.0, 1.0, 0.0])

    # Строим перпендикулярный единичный вектор
    n = np.cross(e, axis)
    n = n / np.linalg.norm(n)

    # Координаты псевдоисточника из MMP
    px = interval["pseudo_x"]
    py = interval["pseudo_y"]

    Q = A + px * e + py * n

    return Q




def shortest_path_on_interval(q0, q1, A, B, start, stop):
    """
    Минимизирует |q0-P| + |P-q1|
    на ограниченном интервале ребра AB.

    Используется геометрическое решение.
    """
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    edge_length = np.linalg.norm(B - A)

    if edge_length < 1e-12:
        raise ValueError("Ребро имеет нулевую длину")

    e = (B - A) / edge_length

    if not (0 <= start <= stop <= edge_length + 1e-10):
        raise ValueError("Некорректные границы интервала")

    # Проекции точек на направление ребра
    a0 = np.dot(q0 - A, e)
    a1 = np.dot(q1 - A, e)

    # Перпендикулярные расстояния до прямой
    h0 = np.linalg.norm(q0 - A - a0 * e)
    h1 = np.linalg.norm(q1 - A - a1 * e)

    if h0 + h1 < 1e-12:
        # Обе точки лежат на прямой ребра
        t = (a0 + a1) / 2
    else:
        # Геометрическая формула минимума
        t = (h1 * a0 + h0 * a1) / (h0 + h1)

    # Ограничение интервалом MMP
    t = np.clip(t, start, stop)

    return A + t * e

class PathVariation:
    def __init__(self, Q0, Q1,
                 A0, B0, interval0,
                 A1, B1, interval1):

        self.Q0 = np.asarray(Q0, dtype=float)
        self.Q1 = np.asarray(Q1, dtype=float)

        self.A0 = np.asarray(A0, dtype=float)
        self.B0 = np.asarray(B0, dtype=float)

        self.A1 = np.asarray(A1, dtype=float)
        self.B1 = np.asarray(B1, dtype=float)

        self.interval0 = interval0
        self.interval1 = interval1

        # Единичное направление первого ребра
        self.e0 = (self.B0 - self.A0)
        self.e0 /= np.linalg.norm(self.e0)

    def __call__(self, t):

        # Точка на первом интервале
        P0 = self.A0 + t * self.e0

        # Оптимальная точка на втором интервале
        P1 = shortest_path_on_interval(
            P0, self.Q1,
            self.A1, self.B1,
            self.interval1["start"],
            self.interval1["stop"]
        )

        # Уточняем точку на первом интервале
        P0_new = shortest_path_on_interval(
            self.Q0, P1,
            self.A0, self.B0,
            self.interval0["start"],
            self.interval0["stop"]
        )

        # Функция профессора g(t)
        g = np.dot(P0_new - P0, self.e0)

        return float(g)


def optimize_interval_pair(
    A0, B0, interval0,
    A1, B1, interval1,
    eps=1e-8
):
    """
    Находит оптимальные точки на паре интервалов MMP.

    Использует геометрическую конструкцию
    профессора и бисекцию.
    """

    Q0 = build_pseudo_source(A0, B0, interval0)
    Q1 = build_pseudo_source(A1, B1, interval1)

    A0 = np.asarray(A0, dtype=float)
    B0 = np.asarray(B0, dtype=float)
    A1 = np.asarray(A1, dtype=float)
    B1 = np.asarray(B1, dtype=float)

    e0 = normalize(B0 - A0)

    g = PathVariation(
        Q0, Q1,
        A0, B0, interval0,
        A1, B1, interval1
    )

    left = float(interval0["start"])
    right = float(interval0["stop"])

    g_left = g(left)
    g_right = g(right)

    if g_left <= 0:
        t = left

    elif g_right >= 0:
        t = right

    else:
        # Бисекция
        while right - left > eps:
            mid = (left + right) / 2
            g_mid = g(mid)

            if g_mid > 0:
                left = mid
            else:
                right = mid

        t = (left + right) / 2

    P0 = A0 + t * e0

    P1 = shortest_path_on_interval(
        P0, Q1,
        A1, B1,
        interval1["start"],
        interval1["stop"]
    )

    length = (
        np.linalg.norm(Q0 - P0)
        + np.linalg.norm(P0 - P1)
        + np.linalg.norm(P1 - Q1)
        + interval0["d"]
        + interval1["d"]
    )

    return {
        "P0": P0,
        "P1": P1,
        "t": t,
        "length": float(length)
    }




if __name__ == "__main__":

    # 1. Задаём два ребра
    A0 = np.array([0.0, 0.0, 0.0])
    B0 = np.array([2.0, 0.0, 0.0])

    A1 = np.array([0.0, 2.0, 0.0])
    B1 = np.array([2.0, 2.0, 0.0])

    # 2. Задаём тестовые интервалы
    interval0 = {
        "start": 0.0,
        "stop": 2.0,
        "d": 0.0,
        "pseudo_x": 0.0,
        "pseudo_y": -1.0
    }

    interval1 = {
        "start": 0.0,
        "stop": 2.0,
        "d": 0.0,
        "pseudo_x": 2.0,
        "pseudo_y": -1.0
    }

    # 3. Строим псевдоисточники
    Q0 = build_pseudo_source(A0, B0, interval0)
    Q1 = build_pseudo_source(A1, B1, interval1)

    print("Q0 =", Q0)
    print("Q1 =", Q1)

    # 4. Исследуем функцию g(t)
    g = PathVariation(
        Q0, Q1,
        A0, B0, interval0,
        A1, B1, interval1
    )

    print("\n=== ФУНКЦИЯ g(t) ===")

    for t in np.linspace(0, 2, 11):
        print(f"t = {t:.2f}, g(t) = {g(t):.8f}")

    # 5. Оптимизация бисекцией
    print("\n=== ОПТИМИЗАЦИЯ ===")

    result = optimize_interval_pair(
        A0, B0, interval0,
        A1, B1, interval1
    )

    print("P0 =", result["P0"])
    print("P1 =", result["P1"])
    print("t =", result["t"])
    print("Длина =", result["length"])

    # 6. Проверка полным перебором
    ts = np.linspace(0.0, 2.0, 501)
    us = np.linspace(0.0, 2.0, 501)

    e0 = normalize(B0 - A0)
    e1 = normalize(B1 - A1)

    P0_grid = A0 + ts[:, None] * e0
    P1_grid = A1 + us[:, None] * e1

    lengths = (
        np.linalg.norm(P0_grid - Q0, axis=1)[:, None]
        + np.linalg.norm(
            P0_grid[:, None, :] - P1_grid[None, :, :],
            axis=2
        )
        + np.linalg.norm(P1_grid - Q1, axis=1)[None, :]
    )

    i, j = np.unravel_index(
        np.argmin(lengths),
        lengths.shape
    )

    print("\n=== ПОЛНЫЙ ПЕРЕБОР ===")
    print("t =", ts[i])
    print("u =", us[j])
    print("Длина =", lengths[i, j])

    print("\nРазница длин:")
    print(abs(result["length"] - lengths[i, j]))
