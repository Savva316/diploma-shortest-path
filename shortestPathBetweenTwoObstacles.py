from convex_hull import ConvexHull
import numpy as np
import pygeodesic.geodesic as geodesic
import copy

def remap_faces(faces, index_map):
    return [tuple(index_map[int(v)] for v in face)for face in faces]
def get_edge_id_dictionary(algorithm, faces):
    # Получаем все уникальные рёбра сетки
    mesh_edges = set()

    for face in faces:
        i, j, k = map(int, face)

        mesh_edges.add(tuple(sorted((i, j))))
        mesh_edges.add(tuple(sorted((j, k))))
        mesh_edges.add(tuple(sorted((k, i))))

    # Строим словарь: пара вершин -> edge_id из pygeodesic
    edge_id_by_pair = {}

    for edge_id in range(len(mesh_edges)):
        v0, v1 = algorithm.get_edge_vertices(edge_id)
        pair = tuple(sorted((int(v0), int(v1))))

        edge_id_by_pair[pair] = edge_id

    # Проверяем, что API нашёл все рёбра сетки
    if set(edge_id_by_pair) != mesh_edges:
        raise ValueError(
            "Рёбра из pygeodesic не совпадают с рёбрами сетки"
        )

    return edge_id_by_pair
def shortestPathBetweenTwoObstacles(hull1:ConvexHull, hull2:ConvexHull, p1: np.ndarray, p2: np.ndarray):

    prev1,new1,p1_index=hull1.add_one_point(p1)
    index_map=hull1.compact()

    p1_index = index_map[p1_index]
    new1 = remap_faces(new1, index_map)
    prev1 = remap_faces(prev1, index_map)

    prev2,new2,p2_index=hull2.add_one_point(p2)
    index_map=hull2.compact()

    p2_index = index_map[p2_index]
    new2 = remap_faces(new2, index_map)
    prev2 = remap_faces(prev2, index_map)

    edges_prev1=set()
    for face in prev1:
        i,j,k=face
        edges_prev1.add(tuple(sorted((i, j))))
        edges_prev1.add(tuple(sorted((j, k))))
        edges_prev1.add(tuple(sorted((k, i))))
    edges_prev1 = np.array(sorted(edges_prev1), dtype=np.int32)


    edges_prev2=set()
    for face in prev2:
        i,j,k=face
        edges_prev2.add(tuple(sorted((i, j))))
        edges_prev2.add(tuple(sorted((j, k))))
        edges_prev2.add(tuple(sorted((k, i))))
    edges_prev2 = np.array(sorted(edges_prev2), dtype=np.int32)


    # Создаём MMP для двух препятствий
    algorithm1 = geodesic.PyGeodesicAlgorithmExact(hull1.points,hull1.faces)

    print("Методы для восстановления пути:")
    print([name for name in dir(algorithm1) if "path" in name.lower() or "geodesic" in name.lower()])

    algorithm2 = geodesic.PyGeodesicAlgorithmExact(hull2.points,hull2.faces)

    edge_id_by_pair1 = get_edge_id_dictionary(algorithm1, hull1.faces)

    edge_id_by_pair2 = get_edge_id_dictionary(algorithm2, hull2.faces)

    edges_prev1_to_id = {tuple(sorted(map(int, edge))): edge_id_by_pair1[tuple(sorted(map(int, edge)))] for edge in edges_prev1}

    edges_prev2_to_id = {tuple(sorted(map(int, edge))): edge_id_by_pair2[tuple(sorted(map(int, edge)))]for edge in edges_prev2}

    # for edge, edge_id in edges_prev1_to_id.items():
    #     print("Ребро:", edge)
    #     print("ID в pygeodesic:", edge_id)

    # Один раз распространяем расстояние
    # от p1 по первой оболочке
    algorithm1.propagate_from_vertex(p1_index)

    distance_test, path_test = algorithm1.geodesicDistance(
    int(p1_index),
    int(algorithm1.get_edge_vertices(next(iter(edges_prev1_to_id.values())))[0]))

    print("Тестовое геодезическое расстояние:", distance_test)
    print("Тестовый путь:")
    print(path_test)

    # Один раз распространяем расстояние
    # от p2 по второй оболочке
    algorithm2.propagate_from_vertex(p2_index)

    print("MMP для двух препятствий построен")

    def optimize_edge_pair(algorithm1,algorithm2,hull1,hull2,edge_id0,edge_id1,num_samples=101):
        # Первое ребро
        v00, v01 = algorithm1.get_edge_vertices(edge_id0)
        A0 = hull1.points[v00]
        B0 = hull1.points[v01]
        length0 = np.linalg.norm(B0 - A0)

        for t in np.linspace(0.0, 1.0, 11):
            position = t * length0
            point = A0 + t * (B0 - A0)
            distance = algorithm1.distance_on_edge(edge_id0, position)

            # print(
            #     f"t={t:.1f}, "
            #     f"position={position:.4f}, "
            #     f"point={point}, "
            #     f"distance={distance:.6f}"
            # )

        # Второе ребро
        v10, v11 = algorithm2.get_edge_vertices(edge_id1)
        A1 = hull2.points[v10]
        B1 = hull2.points[v11]
        length1 = np.linalg.norm(B1 - A1)

        # print("\nПроверка 9 комбинаций:")

        for t_test in [0.0, 0.5, 1.0]:
            for u_test in [0.0, 0.5, 1.0]:

                S0_test = A0 + t_test * (B0 - A0)
                S1_test = A1 + u_test * (B1 - A1)

                d0_test = algorithm1.distance_on_edge(
                    edge_id0, t_test * length0
                )
                d1_test = algorithm2.distance_on_edge(
                    edge_id1, u_test * length1
                )

                bridge_test = np.linalg.norm(
                    S1_test - S0_test
                )

                F_test = d0_test + bridge_test + d1_test

                # print(
                #     f"t={t_test:.1f}, "
                #     f"u={u_test:.1f}, "
                #     f"F={F_test:.6f}"
                # )


        # print("Первое ребро:", v00, v01)
        # print("Его координаты:", A0, B0)
        # print("Длина первого ребра:", length0)

        # print("Второе ребро:", v10, v11)
        # print("Его координаты:", A1, B1)
        # print("Длина второго ребра:", length1)

        

        best = None

        for t in np.linspace(0.0, 1.0, num_samples):
            S0 = A0 + t * (B0 - A0)
            d0 = algorithm1.distance_on_edge(edge_id0, t * length0)

            for u in np.linspace(0.0, 1.0, num_samples):
                S1 = A1 + u * (B1 - A1)
                d1 = algorithm2.distance_on_edge(edge_id1, u * length1)

                bridge = np.linalg.norm(S1 - S0)
                total = d0 + bridge + d1

                if best is None or total < best["length"]:
                    best = {
                        "S0": S0.copy(),
                        "S1": S1.copy(),
                        "t": t,
                        "u": u,
                        "d0": d0,
                        "bridge": bridge,
                        "d1": d1,
                        "length": total,
                    }

        return best
    
    best_global = None

    for edge0, edge_id0 in edges_prev1_to_id.items():
        for edge1, edge_id1 in edges_prev2_to_id.items():

            result = optimize_edge_pair(
                algorithm1=algorithm1,
                algorithm2=algorithm2,
                hull1=hull1,
                hull2=hull2,
                edge_id0=edge_id0,
                edge_id1=edge_id1,
                num_samples=101
            )

            if (
                best_global is None
                or result["length"] < best_global["length"]
            ):
                best_global = {
                    **result,
                    "edge0": edge0,
                    "edge1": edge1,
                    "edge_id0": edge_id0,
                    "edge_id1": edge_id1
                }

    if best_global is None:
        raise ValueError("Нет пар рёбер для проверки")

    print("\n=== Лучший результат ===")
    print("Первое ребро:", best_global["edge0"])
    print("Второе ребро:", best_global["edge1"])
    print("ID первого ребра:", best_global["edge_id0"])
    print("ID второго ребра:", best_global["edge_id1"])
    print("t:", best_global["t"])
    print("u:", best_global["u"])
    print("S0:", best_global["S0"])
    print("S1:", best_global["S1"])
    print("Длина пути:", best_global["length"])

    
    # Вставляем S0 и S1 в оболочки inplace
    p1_index_new, s0_index = hull1.insert_point_on_edge(
        edge=best_global["edge0"],
        point=best_global["S0"],
        p_index=p1_index
    )

    p2_index_new, s1_index = hull2.insert_point_on_edge(
        edge=best_global["edge1"],
        point=best_global["S1"],
        p_index=p2_index
    )

    # Создаём новые алгоритмы для обновлённых сеток
    algorithm1_new = geodesic.PyGeodesicAlgorithmExact(
        hull1.points, hull1.faces
    )

    algorithm2_new = geodesic.PyGeodesicAlgorithmExact(
        hull2.points, hull2.faces
    )

    # Ищем геодезический путь от p1 до S0
    d0, path0 = algorithm1_new.geodesicDistance(
        int(p1_index_new),
        int(s0_index)
    )

    # Ищем геодезический путь от p2 до S1
    d1, path1 = algorithm2_new.geodesicDistance(
        int(p2_index_new),
        int(s1_index)
    )

    print("\n=== Геодезические участки ===")
    print("Длина p1 -> S0:", d0)
    print("Длина p2 -> S1:", d1)
    print("path0:", path0)
    print("path1:", path1)

    return {
        "p1_index_new": p1_index_new,
        "p2_index_new": p2_index_new,
        "s0_index": s0_index,
        "s1_index": s1_index,
        "path0": path0,
        "path1": path1,
        "geodesic_length0": d0,
        "geodesic_length1": d1,
        "S0": best_global["S0"],
        "S1": best_global["S1"],
        "length": best_global["length"],
    }
