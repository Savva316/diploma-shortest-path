import numpy as np
from convex_hull import ConvexHull
from graphics import *


def main():
    # Задаем облако точек
    n = 30
    points = np.random.uniform(-8, 8, size=(n, 3))
    PlotCloudOfPoints(points)

    hull = ConvexHull(points,save_iteration=9)
    PlotFirstFaceOfHull(hull.points,hull.faces)
    PlotIntermediateHull(hull.points,hull.saved_faces_1)
    PlotNextIntermediateHull(hull.points,hull.saved_faces_2)
    PlotConvexHull(hull.points, len(hull.points), len(hull.edges), len(hull.faces), hull.faces, hull.iterations)

    # Новые точки
    p1 = np.array([10.0, 10.0, 10.0])
    p2 = np.array([-10.0, -10.0, -10.0])
    #p2 = np.array([0.0, 10.0, 9.0])

    PlotConvexHullAndTwoPoints(hull.points, hull.faces, p1, p2)

    if not hull.isConvexHullObstacle(p1, p2):
        print(f"Оболочка не является препятствием, кратчайший путь найден. Дистанция: {np.linalg.norm(p2 - p1):.3f} ")
        PlotConvexHullAndTwoPointsWithStraightShortestPath(hull.points, hull.faces, p1, p2)

    else:
        print("Оболочка является препятствием, ищем кратчайший путь")
        prev1,new1,p1_index=hull.add_one_point(p1)
        PlotHullUpdatedByOnePoint(hull.points,hull.V, hull.E, hull.F, prev1,new1)
        PlotConvexHull(hull.points, hull.V, hull.E, hull.F, hull.faces, hull.iterations)
     
        prev2,new2,p2_index=hull.add_one_point(p2)
        
        middle_part = set(map(tuple, prev1)).intersection(map(tuple, prev2))
        middle_part = [list(face)for face in set(map(tuple, prev1)).intersection(map(tuple, prev2))]
        
        PlotHullUpdatedByTwoPoints(hull.points,hull.V, hull.E, hull.F,new1,middle_part,new2)
        PlotConvexHull(hull.points, hull.V, hull.E, hull.F, hull.faces, hull.iterations)

        index_map=hull.compact()

        p1_index = index_map[p1_index]
        p2_index = index_map[p2_index]

        def remap_faces(faces, index_map):
            return [tuple(index_map[int(v)] for v in face)for face in faces]

        new1 = remap_faces(new1, index_map)
        new2 = remap_faces(new2, index_map)
        middle_part = remap_faces(middle_part, index_map)

        shortest_path=hull.find_shortest_path(p1_index, p2_index)
        PlotConvexHullAndShortestPath(hull.points,new1,middle_part,new2,shortest_path)
        PlotConvexHull(hull.points, len(hull.points), len(hull.edges), len(hull.faces), hull.faces, hull.iterations)


if __name__ == "__main__":
    main()
