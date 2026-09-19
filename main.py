import numpy as np
from convex_hull import ConvexHull
#from convex_hull import add_one_point
from graphics import *


def main():
    # Задаем облако точек
    n = 30
    points = np.random.uniform(-10, 10, size=(n, 3))
    PlotCloudOfPoints(points)

    hull = ConvexHull(points,save_iteration=9)
    PlotFirstFaceOfHull(hull.points,hull.faces)
    PlotIntermediateHull(hull.points,hull.saved_faces_1)
    PlotNextIntermediateHull(hull.points,hull.saved_faces_2)
    PlotConvexHull(hull.points, hull.V, hull.E, hull.F, hull.faces, hull.iterations)

    # Новые точки
    p1 = np.array([10.0, 10.0, 10.0])
    p2 = np.array([-10.0, -10.0, -10.0])

    PlotConvexHullAndTwoPoints(hull.points, hull.faces, p1, p2)
    prev1,new1=hull.add_one_point(p1)
    p1_index=hull.p1_index
    PlotHullUpdatedByOnePoint(hull.points,hull.V,hull.E,hull.F, prev1,new1)
    PlotConvexHull(hull.points, hull.V, hull.E, hull.F, hull.faces, hull.iterations)

    prev2,new2=hull.add_one_point(p2)
    p2_index=hull.p1_index
    middle_part = set(prev1).intersection(prev2)
    
    PlotHullUpdatedByTwoPoints(hull.points,hull.V,hull.E,hull.F,new1,middle_part,new2)
    PlotConvexHull(hull.points, hull.V, hull.E, hull.F, hull.faces, hull.iterations)

    shortest_path=hull.find_shortest_path(p1_index, p2_index)
    PlotConvexHullAndShortestPath(hull.points,new1,middle_part,new2,shortest_path)
    PlotConvexHull(hull.points, hull.V, hull.E, hull.F, hull.faces, hull.iterations)


if __name__ == "__main__":
    main()
