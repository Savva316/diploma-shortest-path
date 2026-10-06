import numpy as np
from convex_hull import ConvexHull
from graphics import *
from shortestPathBetweenTwoObstacles import *


def main():
    # Задаем первое препятствие
    n = 30
    points = np.random.uniform(0, 10, size=(n, 3))
    hull1 = ConvexHull(points,save_iteration=9)
    # Задаем второе препятствие
    points = np.random.uniform(15, 25, size=(n, 3))
    hull2 = ConvexHull(points,save_iteration=9)
    
    PlotTwoConvexHulls(hull1, hull2)

    # Новые точки
    p1 = np.array([30.0, 30.0, 30.0])
    p2 = np.array([-5.0, -5.0, -5.0])
    # p2 = np.array([0.0, 10.0, 9.0])

    #p1 = np.array([5.0, 5.0, 15.0])
    #p2 = np.array([20.0, 20.0, -5.0])

    PlotTwoConvexHulls(hull1, hull2,p1,p2)

    if not hull1.isConvexHullObstacle(p1, p2) and not hull2.isConvexHullObstacle(p1, p2):
        print(f"Оболочка не является препятствием, кратчайший путь найден. Дистанция: {np.linalg.norm(p2 - p1):.3f} ")
        path=np.array([p1,p2])
        PlotTwoConvexHullsAndShortestPath(hull1, hull2,path)
    else:
        shortestPathBetweenTwoObstacles(hull1, hull2,p1,p2)
        pass

if __name__ == "__main__":
    main()
