
import numpy as np


EPS = 1e-12


def normalize(v):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)

    if norm < EPS:
        raise ValueError("Нельзя нормировать нулевой вектор.")

    return v / norm


def rotation_matrix(a, b):
    """
    Матрица вращения, переводящая направление a
    в направление b.
    """
    a = normalize(a)
    b = normalize(b)

    cross = np.cross(a, b)
    sin_angle = np.linalg.norm(cross)
    cos_angle = np.clip(np.dot(a, b), -1.0, 1.0)

    if sin_angle < EPS:
        if cos_angle > 0:
            return np.eye(3)

        # Противоположные направления: поворот на 180 градусов.
        axis = np.cross(a, [1.0, 0.0, 0.0])

        if np.linalg.norm(axis) < EPS:
            axis = np.cross(a, [0.0, 1.0, 0.0])

        axis = normalize(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    K = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0]
    ])

    return (
        np.eye(3)
        + K
        + K @ K * ((1.0 - cos_angle) / sin_angle**2)
    )


def intersect_lines_in_3d(a, da, b, db):
    """
    Пересечение прямых:
        a + alpha * da
        b + beta * db

    Если из-за погрешности прямые немного скрещиваются,
    возвращает точку на второй прямой, полученную
    методом наименьших квадратов.
    """
    a = np.asarray(a, dtype=float)
    da = np.asarray(da, dtype=float)
    b = np.asarray(b, dtype=float)
    db = np.asarray(db, dtype=float)

    matrix = np.column_stack((da, -db))
    rhs = b - a

    solution, _, _, _ = np.linalg.lstsq(
        matrix, rhs, rcond=None
    )

    alpha, beta = solution

    point_first = a + alpha * da
    point_second = b + beta * db

    if np.linalg.norm(point_first - point_second) > 1e-7:
        raise ValueError(
            "Прямые не пересекаются с требуемой точностью."
        )

    return point_second


def shortest_path(q0, q1, p, v):
    """
    Аналог shortestPath() преподавателя.

    Возвращает точку s на прямой p + lambda*v.
    """
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    p = np.asarray(p, dtype=float)
    v = np.asarray(v, dtype=float)

    w0 = q0 - p
    n0 = np.cross(w0, v)

    w1 = q1 - p
    n1 = np.cross(v, w1)

    if np.linalg.norm(n0) < EPS:
        raise ValueError(
            "Вырождение: q0 лежит на прямой (p, v)."
        )

    if np.linalg.norm(n1) < EPS:
        raise ValueError(
            "Вырождение: q1 лежит на прямой (p, v)."
        )

    R = rotation_matrix(n0, n1)

    q0_image = p + R @ w0

    s = intersect_lines_in_3d(
        q0_image,
        q1 - q0_image,
        p,
        v
    )

    return s


def function_root(f, t0, t1, eps=1e-6, max_iter=1000):
    """
    Аналог functionRoot() преподавателя.
    Метод бисекции.
    """
    if t0 >= t1:
        raise ValueError("Необходимо t0 < t1.")

    f0 = float(f(t0))
    f1 = float(f(t1))

    if abs(f0) < eps:
        return t0

    if abs(f1) < eps:
        return t1

    if f0 * f1 > 0:
        raise ValueError(
            "На концах интервала нет смены знака."
        )

    for _ in range(max_iter):
        if abs(t1 - t0) <= eps:
            break

        tc = (t0 + t1) / 2.0

        if tc <= t0 or tc >= t1:
            break

        fc = float(f(tc))

        if abs(fc) < eps:
            return tc

        if f0 * fc < 0:
            t1 = tc
            f1 = fc
        else:
            t0 = tc
            f0 = fc

    return (t0 + t1) / 2.0


class PathVariation:
    """
    Аналог класса PathVariation из C++.
    """

    def __init__(self, q0, q1, p0, v0, p1, v1):
        self.q0 = np.asarray(q0, dtype=float)
        self.q1 = np.asarray(q1, dtype=float)
        self.p0 = np.asarray(p0, dtype=float)
        self.v0 = normalize(v0)
        self.p1 = np.asarray(p1, dtype=float)
        self.v1 = normalize(v1)

    def __call__(self, t):
        # Точка на первой прямой.
        s0 = self.p0 + self.v0 * t

        # Точка на второй прямой.
        s1 = shortest_path(
            s0, self.q1, self.p1, self.v1
        )

        # Повторное уточнение точки на первой прямой.
        s00 = shortest_path(
            self.q0, s1, self.p0, self.v0
        )

        # Подписанное отклонение.
        return float(np.dot(s00 - s0, self.v0))


def shortest_path2(
    q0,
    q1,
    p0,
    v0,
    p1,
    v1,
    t0=-10.0,
    t1=10.0,
    eps=1e-6,
    max_expand=100
):
    """
    Аналог shortestPath2() преподавателя.

    Возвращает длину пути q0 -> s0 -> s1 -> q1,
    а также точки s0 и s1 и диагностические значения.
    """
    if t0 >= t1:
        raise ValueError("Необходимо t0 < t1.")

    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    u0 = normalize(v0)
    u1 = normalize(v1)

    path_defect = PathVariation(
        q0, q1, p0, u0, p1, u1
    )


    for t_test in [-10, -5, -2, 0, 2, 5, 10]:
        print(f"t={t_test:5.1f}, g(t)={path_defect(t_test): .8f}")

    value0 = path_defect(t0)
    value1 = path_defect(t1)

    # Как у преподавателя: расширяем интервал,
    # пока значения не охватят ноль.
    for _ in range(max_expand):
        if value0 * value1 <= 0:
            break

        dt = t1 - t0

        if abs(value0) <= abs(value1):
            # Расширяем интервал влево.
            t1 = t0
            t0 -= dt

            value1 = value0
            value0 = path_defect(t0)

        else:
            # Расширяем интервал вправо.
            t0 = t1
            t1 += dt

            value0 = value1
            value1 = path_defect(t1)

    else:
        raise RuntimeError(
            "Не удалось найти интервал со сменой знака g(t)."
        )

    # Ищем корень g(t) = 0.
    t = function_root(
        path_defect, t0, t1, eps=eps
    )

    # Финальные точки пути.
    s0 = p0 + u0 * t

    s1 = shortest_path(
        s0, q1, p1, u1
    )

    length = (
        np.linalg.norm(s0 - q0)
        + np.linalg.norm(s1 - s0)
        + np.linalg.norm(q1 - s1)
    )

    return {
        "s0": s0,
        "s1": s1,
        "t": t,
        "g": path_defect(t),
        "length": float(length),
        "interval": (t0, t1),
    }


def grid_check(q0, q1, p0, v0, p1, v1, bound=20, num=1001):
    t_values = np.linspace(-bound, bound, num)
    u_values = np.linspace(-bound, bound, num)

    points0 = p0 + t_values[:, None] * normalize(v0)
    points1 = p1 + u_values[:, None] * normalize(v1)

    d0 = np.linalg.norm(points0 - q0, axis=1)
    d1 = np.linalg.norm(points1 - q1, axis=1)

    middle = np.linalg.norm(
        points0[:, None, :] - points1[None, :, :],
        axis=2
    )

    lengths = d0[:, None] + middle + d1[None, :]

    i, j = np.unravel_index(np.argmin(lengths), lengths.shape)

    return t_values[i], u_values[j], lengths[i, j]



if __name__ == "__main__":
    # Тест вырожденной геометрической конфигурации
    # Начальная точка q0 совпадает с точкой p1 на второй прямой.

    q0 = np.array([3., 2., 2.])
    q1 = np.array([6., 5., 4.])

    p0 = np.array([1., 1., 0.])
    v0 = np.array([1., 0., 0.])

    p1 = np.array([3., 2., 2.])
    v1 = np.array([1., 0., 0.])

    print("=" * 50)
    print("Тест вырожденной конфигурации")
    print("q0 =", q0)
    print("q1 =", q1)
    print("p0 =", p0)
    print("v0 =", v0)
    print("p1 =", p1)
    print("v1 =", v1)
    print("q0 == p1:", np.allclose(q0, p1))

    result = shortest_path2(
        q0, q1, p0, v0, p1, v1,
        t0=-10,
        t1=10
    )

    s0 = result["s0"]
    s1 = result["s1"]

    print("\nРезультат алгоритма:")
    print("s0 =", s0)
    print("s1 =", s1)
    print(f"t = {result['t']:.8f}")
    print(f"g(t) = {result['g']:.3e}")
    print("Интервал =", result["interval"])
    print(f"Длина пути = {result['length']:.8f}")

    print("\nПроверка принадлежности прямым:")
    print(
        "||cross(s0 - p0, v0)|| =",
        np.linalg.norm(np.cross(s0 - p0, v0))
    )
    print(
        "||cross(s1 - p1, v1)|| =",
        np.linalg.norm(np.cross(s1 - p1, v1))
    )

    print("\nРасстояния:")
    print("q0 -> s0:", np.linalg.norm(q0 - s0))
    print("s0 -> s1:", np.linalg.norm(s0 - s1))
    print("s1 -> q1:", np.linalg.norm(s1 - q1))
