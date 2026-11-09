
import numpy as np
from pygeodesic import geodesic
from optimize_intervals import build_pseudo_source
# 1. Вершины куба
vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]
], dtype=np.float64)

# 2. Треугольные грани куба
faces = np.array([
    [0, 2, 1], [0, 3, 2],
    [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4],
    [1, 2, 6], [1, 6, 5],
    [2, 3, 7], [2, 7, 6],
    [3, 0, 4], [3, 4, 7]
], dtype=np.int32)

# 3. Создаём алгоритм MMP
algorithm = geodesic.PyGeodesicAlgorithmExact(vertices, faces)

# Источник S — вершина 0
source_index = 0
algorithm.propagate_from_vertex(source_index)


# 4. Проверяем все рёбра куба

total_intervals = 0
max_intervals = 0

for edge_id in range(18):
    v0, v1 = algorithm.get_edge_vertices(edge_id)
    intervals = algorithm.get_edge_intervals(edge_id)

    count = len(intervals)
    total_intervals += count
    max_intervals = max(max_intervals, count)

    print(
        f"Ребро {edge_id:2d} ({v0}, {v1}): "
        f"{count} интервалов"
    )

    if count > 1:
        for i, interval in enumerate(intervals):
            print(
                f"    Интервал {i}: "
                f"[{interval['start']:.6f}, "
                f"{interval['stop']:.6f}]"
            )

print("\nВсего интервалов:", total_intervals)
print("Максимум на одном ребре:", max_intervals)

# Исследуем ребро с двумя интервалами
edge_id = 10
intervals = algorithm.get_edge_intervals(edge_id)

print("\n=== ИССЛЕДОВАНИЕ РЕБРА 10 ===")

for i, interval in enumerate(intervals):
    print(f"\nИнтервал {i}:")
    print(interval)

    x = (interval["start"] + interval["stop"]) / 2

    d = algorithm.distance_on_edge(edge_id, x)

    print(f"Середина: {x:.8f}")
    print(f"Расстояние: {d:.8f}")

# Проверяем границу переключения
boundary = intervals[0]["stop"]
eps = 1e-6

print("\n=== ГРАНИЦА ИНТЕРВАЛОВ ===")

for x in [boundary - eps, boundary, boundary + eps]:
    d = algorithm.distance_on_edge(edge_id, x)
    print(f"x = {x:.8f}, d(x) = {d:.10f}")



print("\n=== ПРОВЕРКА ПСЕВДОИСТОЧНИКОВ В 3D ===")

edge_id = 10

# Порядок вершин важен: координата pseudo_x
# отсчитывается от первой вершины ребра.
v0, v1 = algorithm.get_edge_vertices(edge_id)

A = vertices[v0]
B = vertices[v1]

edge_vector = B - A
edge_length = np.linalg.norm(edge_vector)
e = edge_vector / edge_length

intervals = algorithm.get_edge_intervals(edge_id)

for i, interval in enumerate(intervals):
    print(f"\nИнтервал {i}")

    # Строим псевдоисточник в 3D
    Q = build_pseudo_source(A, B, interval)

    print("Псевдоисточник Q:", Q)

    # Проверяем несколько точек внутри интервала
    positions = np.linspace(
        interval["start"],
        interval["stop"],
        5
    )

    for x in positions:
        # Точка на ребре
        P = A + x * e

        # Расстояние через 3D-псевдоисточник
        d_pseudo = np.linalg.norm(Q - P) + interval["d"]

        # Расстояние через MMP
        d_mmp = algorithm.distance_on_edge(edge_id, x)

        error = abs(d_pseudo - d_mmp)

        print(
            f"x={x:.6f} | "
            f"MMP={d_mmp:.10f} | "
            f"3D={d_pseudo:.10f} | "
            f"ошибка={error:.2e}"
        )
