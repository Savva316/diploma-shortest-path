
import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "myMMP")
)

from convex_hull import ConvexHull
from pygeodesic.geodesic import PyGeodesicAlgorithmExact


def point_on_triangle(p, a, b, c, tol=1e-7):
    ab = b - a
    ac = c - a
    ap = p - a

    normal = np.cross(ab, ac)
    normal_norm = np.linalg.norm(normal)

    if normal_norm < 1e-12:
        return False

    distance = abs(np.dot(ap, normal)) / normal_norm
    if distance > tol:
        return False

    d00 = np.dot(ab, ab)
    d01 = np.dot(ab, ac)
    d11 = np.dot(ac, ac)
    d20 = np.dot(ap, ab)
    d21 = np.dot(ap, ac)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return False

    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom

    return u >= -tol and v >= -tol and u + v <= 1 + tol


# Строим выпуклую оболочку
np.random.seed(42)
points = np.random.uniform(-10, 10, size=(50, 3))
hull = ConvexHull(points, save_iteration=5)

V = np.asarray(hull.points, dtype=np.float64)
F = np.asarray(hull.faces, dtype=np.int32)

print("Вершин:", len(V))
print("Треугольных граней:", len(F))
print("Рёбер:", len(hull.edges))

# Запускаем MMP от вершины 0
algorithm = PyGeodesicAlgorithmExact(V, F)
algorithm.propagate_from_vertex(0)

# Проверяем два ребра около их концов
test_edges = [42]
test_positions = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

num_tests = 0
num_failures = 0
max_target_error = 0.0
max_source_error = 0.0
max_distance_error = 0.0
all_paths_on_surface = True

tol = 1e-7

for edge_id in test_edges:
    v0, v1 = algorithm.get_edge_vertices(edge_id)
    A, B = V[v0], V[v1]
    edge_length = np.linalg.norm(B - A)

    for t in test_positions:
        num_tests += 1

        target = A + t * (B - A)

        path = np.asarray(
            algorithm.trace_back_to_edge(edge_id, t),
            dtype=np.float64
        )
        vertex = hull.points[8]  # если hull.vertices — массив координат вершин
        print("Вершина 8:", vertex)

        for i, point in enumerate(path):
            print(
                f"Точка {i}: "
                f"расстояние до вершины 8 = "
                f"{np.linalg.norm(point - vertex):.12e}"
            )

        print("\nДлины сегментов:")
        for i in range(len(path) - 1):
            segment = np.linalg.norm(path[i + 1] - path[i])
            print(f"{i} -> {i + 1}: {segment:.12e}")
        # Длина пути — сумма длин всех сегментов
        segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
        path_length = segments.sum()

        # Расстояние MMP до той же точки
        target_distance = algorithm.distance_on_edge(
            edge_id, t * edge_length
        )

        distance_error = abs(path_length - target_distance)

        target_error = np.linalg.norm(path[0] - target)
        source_error = np.linalg.norm(path[-1] - V[0])

        max_target_error = max(max_target_error, target_error)
        max_source_error = max(max_source_error, source_error)
        max_distance_error = max(max_distance_error, distance_error)

        # Проверяем, лежат ли точки каждого сегмента на поверхности
        path_on_surface = True

        for i in range(len(path) - 1):
            for s in np.linspace(0, 1, 11):
                p = path[i] + s * (path[i + 1] - path[i])

                on_surface = any(
                    point_on_triangle(
                        p,
                        V[face[0]],
                        V[face[1]],
                        V[face[2]]
                    )
                    for face in F
                )

                if not on_surface:
                    path_on_surface = False
                    break

            if not path_on_surface:
                break

        if not path_on_surface:
            all_paths_on_surface = False

        print(
            f"\nedge={edge_id}, t={t:.8f}, "
            f"points={len(path)}"
        )
        print("Длины сегментов:", segments)
        print("Цель:", path[0])
        print("Первая точка после цели:", path[1])
        print("Весь путь:")
        print(path)
        for i, p in enumerate(path[1:-1], start=1):
            distances = np.linalg.norm(V - p, axis=1)
            nearest_vertex = np.argmin(distances)

            print(
                f"Промежуточная точка {i}: "
                f"ближайшая вершина {nearest_vertex}, "
                f"расстояние до неё {distances[nearest_vertex]:.3e}"
            )
        print("Источник:", path[-1])
        print(f"Длина пути:       {path_length:.12f}")
        print(f"Расстояние MMP:   {target_distance:.12f}")
        print(f"Разница:          {distance_error:.3e}")

        if (
            target_error > tol
            or source_error > tol
            or distance_error > tol
            or not path_on_surface
        ):
            num_failures += 1
            print("ПРОВЕРКА НЕ ПРОЙДЕНА")

print("\n--- ИТОГИ ТЕСТИРОВАНИЯ ---")
print(f"Всего проверок: {num_tests}")
print(f"Ошибок: {num_failures}")
print(f"Максимальная ошибка цели: {max_target_error:.2e}")
print(f"Максимальная ошибка источника: {max_source_error:.2e}")
print(f"Максимальная ошибка расстояния: {max_distance_error:.2e}")
print(f"Все пути на поверхности: {all_paths_on_surface}")
