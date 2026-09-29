import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def PlotCloudOfPoints(vertices: np.ndarray):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Точки
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=20, color='red')

    #ax.scatter(center[0], center[1], center[2], s=40, color='green')

    # Подписи осей
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Красиво отобразить
    ax.view_init(elev=10, azim=-45,roll=10)
    ax.set_axis_off()
    #plt.savefig(r"/Users/savva/Desktop/MSU/2st sem/курсач/figure2.1.pdf")
    plt.show()

def PlotFirstFaceOfHull(vertices,faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Рисуем точки
    ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], color='red', s=20)
    # ax.scatter(center[0], center[1], center[2], s=40, color='green')

    # Рисуем грань
    poly3d = []
    triangle = [vertices[faces[0][0]], vertices[faces[0][1]], vertices[faces[0][2]]]
    poly3d.append(triangle)

    ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=0.5, edgecolors='k', alpha=0.5))

    # Подписи осей
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Красивый ракурс
    ax.view_init(elev=10, azim=-45,roll=10)
    ax.set_axis_off()
    #plt.savefig(r"/Users/savva/Desktop/MSU/2st sem/курсач/figure2.2.pdf")
    plt.show()

def PlotIntermediateHull(vertices,faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], color='red', s=20)

    DrawHull(ax, vertices, faces)
    #plt.savefig(r"/Users/savva/Desktop/MSU/2st sem/курсач/figure3.1.pdf")
    plt.show()

def PlotNextIntermediateHull(vertices,faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], color='red', s=20)

    DrawHull(ax, vertices, faces[:-1])

    last_face=faces[-1]
    triangle = [vertices[last_face[0]], vertices[last_face[1]], vertices[last_face[2]]]
    ax.add_collection3d(Poly3DCollection([triangle], facecolors='crimson', linewidths=0.5, edgecolors='k', alpha=0.5))
    #plt.savefig(r"/Users/savva/Desktop/MSU/2st sem/курсач/figure3.2.pdf")
    plt.show()

def DrawHull(ax, vertices, faces,c='cyan',visibility=0.5):
    poly3d = []
    for face in faces:
        triangle = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
        poly3d.append(triangle)

    ax.add_collection3d(Poly3DCollection(poly3d, facecolors=c,
                         linewidths=0.5, edgecolors='k', alpha=visibility))

    ax.view_init(elev=10, azim=-45, roll=10)
    ax.set_axis_off()

def PlotConvexHull(vertices,V,E,F,faces,iterations):
    # Отладочная печать
    print(f"При построении выпуклой оболочки произведено {iterations} итераций")

    # Отладочная печать
    # В + Г - Р = 2
    print(f"При построении выпуклой оболочки получено: ",end=" ")
    print(f"{V} вершин",end=" ")
    print(f"{F} граней",end=" ")
    print(f"{E} ребер")
    print(f"Эйлерова характеристика равна {V+F-E} ")
    print("###","###","###",sep='\n')
    # Визуализация выпуклой оболочки
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], color='red', s=20)
    DrawHull(ax, vertices, faces)
    #plt.savefig(r"/Users/savva/Desktop/MSU/2st sem/курсач/figure4.pdf")
    plt.show()

def PlotConvexHullAndTwoPoints(vertices, faces,p1,p2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], color='red', s=20)

    DrawHull(ax, vertices, faces)
    ax.scatter(p1[0],p1[1],p1[2],color='green',s=60)
    ax.scatter(p2[0],p2[1],p2[2],color='green',s=60)
    #plt.savefig(r"/Users/savva/Desktop/MSU/2st sem/курсач/figure5.pdf")

    plt.show()
def PlotConvexHullAndTwoPointsWithStraightShortestPath(vertices,faces, p1, p2):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], color='red', s=20)
    
        DrawHull(ax, vertices, faces)
        ax.scatter(p1[0],p1[1],p1[2],color='green',s=60)
        ax.scatter(p2[0],p2[1],p2[2],color='green',s=60)
        path=np.array([p1,p2])
        ax.plot(path[:,0], path[:,1], path[:,2], color='magenta', linewidth=4)
        #plt.savefig(r"/Users/savva/Desktop/MSU/2st sem/курсач/figure5.pdf")
    
        plt.show()
def PlotHullUpdatedByOnePoint(vertices,V,E,F, prev,new):
    # Отладочная печать
    #В-Р+Г=2
    print('Построение первого дополнения к выпуклой оболочке')
    print(f"Эйлерова характеристика равна {V+F-E} ")
    print("###","###","###",sep='\n')
    # Визуализация выпуклой оболочки
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], color='red', s=20)
    ax.scatter(vertices[-1][0], vertices[-1][1], vertices[-1][2], s=60, color='green')
    DrawHull(ax, vertices, prev)
    DrawHull(ax, vertices, new,'orange')

    #plt.savefig(r"/Users/savva/Desktop/MSU/2st sem/курсач/figure6.pdf")

    plt.show()  

def PlotHullUpdatedByTwoPoints(vertices,V,E,F, new1,middle,new2):
    # Отладочная печать
        #В-Р+Г=2
        print('Построение второго дополнения к выпуклой оболочке')
        print(f"Эйлерова характеристика равна {V+F-E} ")
        print("###","###","###",sep='\n')
        # Визуализация выпуклой оболочки
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], color='red', s=20)
        ax.scatter(vertices[-2:,0], vertices[-2:,1], vertices[-2:,2], s=60, color='green')
        DrawHull(ax, vertices, middle)
        DrawHull(ax, vertices, new1,'orange')
        DrawHull(ax, vertices, new2,'seagreen')
        #plt.savefig(r"/Users/savva/Desktop/MSU/2st sem/курсач/figure6.pdf")
    
        plt.show()  
def PlotConvexHullAndShortestPath(vertices,new1,middle_part,new2,path):
    # Визуализация выпуклой оболочки
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], color='red', s=20)
            ax.scatter(vertices[-2:,0], vertices[-2:,1], vertices[-2:,2], s=60, color='green')
            DrawHull(ax, vertices, middle_part,'cyan',visibility=0.15)
            DrawHull(ax, vertices, new1,'orange',visibility=0.15)
            DrawHull(ax, vertices, new2,'seagreen',visibility=0.15)

            # Рисуем путь
            ax.plot(path[:,0], path[:,1], path[:,2], color='magenta', linewidth=4)
            ax.scatter(path[1:-1,0], path[1:-1,1], path[1:-1,2], color='yellow',edgecolors='black', s=60)

            # Выделяем старт и финиш ярче
            ax.scatter(path[0,0], path[0,1], path[0,2], s=60, color='green', edgecolors='black', label='Start')
            ax.scatter(path[-1,0], path[-1,1], path[-1,2], s=60, color='green', edgecolors='black', label='End')
            #plt.savefig(r"/Users/savva/Desktop/MSU/2st sem/курсач/figure8.pdf")
        
            plt.show()  
