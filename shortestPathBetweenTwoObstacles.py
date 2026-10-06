from convex_hull import ConvexHull
import numpy as np
import pygeodesic.geodesic as geodesic

def remap_faces(faces, index_map):
    return [tuple(index_map[int(v)] for v in face)for face in faces]
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

    algorithm2 = geodesic.PyGeodesicAlgorithmExact(hull2.points,hull2.faces)

    # Один раз распространяем расстояние
    # от p1 по первой оболочке
    algorithm1.propagate_from_vertex(p1_index)

    # Один раз распространяем расстояние
    # от p2 по второй оболочке
    algorithm2.propagate_from_vertex(p2_index)

    print("MMP для двух препятствий построен")

    edge_id = 0

    v0, v1 = algorithm1.get_edge_vertices(edge_id)

    A = hull1.points[v0]
    B = hull1.points[v1]

    edge_length = np.linalg.norm(B - A)

    print("edge:", edge_id)
    print("vertices:", v0, v1)
    print("length:", edge_length)

    for t in np.linspace(0, 1, 5):
        position = t * edge_length
        distance = algorithm1.distance_on_edge(edge_id, position)

        point = A + t * (B - A)

        print(
            f"t={t:.2f}, "
            f"point={point}, "
            f"distance={distance:.6f}"
        )

    return 