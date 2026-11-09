
import numpy as np
from pygeodesic import geodesic

from optimize_intervals import (
    optimize_interval_pair,
    normalize
)


# 1. Геометрия двух кубов

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

faces = np.array([
    [0, 2, 1], [0, 3, 2],
    [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4],
    [1, 2, 6], [1, 6, 5],
    [2, 3, 7], [2, 7, 6],
    [3, 0, 4], [3, 4, 7]
], dtype=np.int32)

vertices0 = vertices.copy()
vertices1 = vertices + np.array([3.0, 0.0, 0.0])

alg0 = geodesic.PyGeodesicAlgorithmExact(vertices0, faces)
alg1 = geodesic.PyGeodesicAlgorithmExact(vertices1, faces)

# Источники на двух кубах
S0_index = 0
S1_index = 1

alg0.propagate_from_vertex(S0_index)
alg1.propagate_from_vertex(S1_index)



# 2. Находим рёбра обращённых граней

def get_facing_edges(algorithm, vertices, x_value):
    """
    Возвращает рёбра, лежащие в плоскости x=x_value.
    """
    edges = []

    for edge_id in range(18):
        v0, v1 = algorithm.get_edge_vertices(edge_id)

        A = vertices[v0]
        B = vertices[v1]

        if (
            np.isclose(A[0], x_value)
            and np.isclose(B[0], x_value)
        ):
            edges.append((edge_id, A, B))

    return edges


edges0 = get_facing_edges(alg0, vertices0, 1.0)
edges1 = get_facing_edges(alg1, vertices1, 3.0)

print("Рёбер первого куба:", len(edges0))
print("Рёбер второго куба:", len(edges1))
print("Всего пар:", len(edges0) * len(edges1))

assert len(edges0) == 5
assert len(edges1) == 5


# 3. Получаем интервалы MMP

interval_data0 = {}
interval_data1 = {}

for edge_id, A, B in edges0:
    interval_data0[edge_id] = alg0.get_edge_intervals(edge_id)

for edge_id, A, B in edges1:
    interval_data1[edge_id] = alg1.get_edge_intervals(edge_id)


# 4. Оптимизация бисекцией

best_bisection = None
total_interval_pairs = 0

for edge0_id, A0, B0 in edges0:
    for edge1_id, A1, B1 in edges1:

        intervals0 = interval_data0[edge0_id]
        intervals1 = interval_data1[edge1_id]

        for i, interval0 in enumerate(intervals0):
            for j, interval1 in enumerate(intervals1):

                total_interval_pairs += 1

                result = optimize_interval_pair(
                    A0, B0, interval0,
                    A1, B1, interval1
                )

                if (
                    best_bisection is None
                    or result["length"] < best_bisection["length"]
                ):
                    best_bisection = {
                        **result,
                        "edge_ids": (edge0_id, edge1_id),
                        "interval_ids": (i, j)
                    }


print("\n=== РЕЗУЛЬТАТ БИСЕКЦИИ ===")

print("Всего пар интервалов:", total_interval_pairs)
print("Рёбра:", best_bisection["edge_ids"])
print("Интервалы:", best_bisection["interval_ids"])
print("P0 =", best_bisection["P0"])
print("P1 =", best_bisection["P1"])
print("Длина =", best_bisection["length"])


# 5. Независимый перебор по всем парам рёбер

num = 301
best_grid = None

for edge0_id, A0, B0 in edges0:
    for edge1_id, A1, B1 in edges1:

        e0 = normalize(B0 - A0)
        e1 = normalize(B1 - A1)

        length0 = np.linalg.norm(B0 - A0)
        length1 = np.linalg.norm(B1 - A1)

        ts = np.linspace(0, length0, num)
        us = np.linspace(0, length1, num)

        points0 = A0 + ts[:, None] * e0
        points1 = A1 + us[:, None] * e1

        # Геодезические расстояния напрямую из MMP
        d0 = np.array([
            alg0.distance_on_edge(edge0_id, t)
            for t in ts
        ])

        d1 = np.array([
            alg1.distance_on_edge(edge1_id, u)
            for u in us
        ])

        middle = np.linalg.norm(
            points0[:, None, :] - points1[None, :, :],
            axis=2
        )

        lengths = (
            d0[:, None]
            + middle
            + d1[None, :]
        )

        i, j = np.unravel_index(
            np.argmin(lengths),
            lengths.shape
        )

        current_length = float(lengths[i, j])

        if (
            best_grid is None
            or current_length < best_grid["length"]
        ):
            best_grid = {
                "length": current_length,
                "P0": points0[i],
                "P1": points1[j],
                "edge_ids": (edge0_id, edge1_id)
            }


print("\n=== РЕЗУЛЬТАТ ПЕРЕБОРА ===")

print("Рёбра:", best_grid["edge_ids"])
print("P0 =", best_grid["P0"])
print("P1 =", best_grid["P1"])
print("Длина =", best_grid["length"])


# 6. Проверяем результат бисекции через MMP

edge0_id, edge1_id = best_bisection["edge_ids"]

A0, B0 = next(
    (A, B) for eid, A, B in edges0 if eid == edge0_id
)

A1, B1 = next(
    (A, B) for eid, A, B in edges1 if eid == edge1_id
)

P0 = best_bisection["P0"]
P1 = best_bisection["P1"]

t = float(np.dot(P0 - A0, normalize(B0 - A0)))
u = float(np.dot(P1 - A1, normalize(B1 - A1)))

d0 = alg0.distance_on_edge(edge0_id, t)
d1 = alg1.distance_on_edge(edge1_id, u)

direct_length = d0 + np.linalg.norm(P0 - P1) + d1


print("\n=== СРАВНЕНИЕ ===")

print("Бисекция:", best_bisection["length"])
print("Перебор:", best_grid["length"])
print(
    "Разница:",
    best_bisection["length"] - best_grid["length"]
)

print("\nПроверка через MMP:")
print("Длина напрямую:", direct_length)
print(
    "Ошибка:",
    abs(direct_length - best_bisection["length"])
)

# Геометрическая проверка положения моста
assert np.isclose(P0[0], 1.0)
assert np.isclose(P1[0], 3.0)

print("\nМост находится между кубами.")
