import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

eps=1e-12

ez=np.array([0,0,1]) # нормаль к oXY

n=30 #количество точек
idx_start_point=n
idx_end_point=n+1

First_vertices = np.random.uniform(-10, 10, size=(n, 3)) # массив из n рандомных точек

#First_vertices = np.array([[0.0, 4.0, 0.0],   [4.0, 0.0, 0.0],    [0.0, 4.0, 3.0],   [0.0, 0.0, 4.0],   [1.0, 5.0, 5.0]])

vertices = First_vertices
used = np.ones(len(vertices), dtype=bool)
center=np.mean(First_vertices,axis=0)


#%% Визуализация облака точек
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Точки
ax.scatter(First_vertices[:, 0], First_vertices[:, 1], First_vertices[:, 2], s=20, color='red')

#ax.scatter(center[0], center[1], center[2], s=40, color='green')

# Подписи осей
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Красиво отобразить
ax.view_init(elev=10, azim=-45,roll=10)
ax.set_axis_off()
plt.savefig(r"/Users/savva/Desktop/MSU/2st sem/курсач/figure2.1.pdf")
plt.show()


#%% Инициализация алгоритма
# Найдем первую точку
A_index = np.argmin(vertices[:,2])
A = vertices[A_index]
# Нашли минимальную по координате z точку

# Найдем коорддинаты второй точки 

cos_alpha=1

for i in range(n):
    if i==A_index:
        continue
    N=vertices[i]
    NA=A-N
    if np.linalg.norm(NA)<eps:
        continue
    cos_alpha_i=(np.dot(NA, -ez))/np.linalg.norm(NA)
    if (cos_alpha_i<cos_alpha):
        B=N
        B_index=i
        cos_alpha=cos_alpha_i
AB=B-A
# Найдем третью точку

cos_beta=-1

for i in range(n):
    if i == A_index or i==B_index:
        continue

    N=vertices[i]
    BN=N-B

    if np.linalg.norm(np.cross(AB, BN))<eps:
        continue

    n_ABN=(np.cross(AB, BN))/np.linalg.norm(np.cross(AB, BN))

    if n_ABN[2]<0:
        n_ABN=-n_ABN

    cos_beta_i=(np.dot(n_ABN, ez))
    if (cos_beta_i>cos_beta):
        cos_beta=cos_beta_i
        C=N
        C_index=i
        n_ABC=n_ABN


# чтобы у нас была нормаль смотрящая наружу и правая тройка точек (при виде из вне обход против часовой стрелки)
    
A=vertices[A_index]
B=vertices[B_index]
C=vertices[C_index]

n=np.cross(B-A,C-A)/np.linalg.norm(np.cross(B-A,C-A))

if np.dot(n,ez)>0:
    ai=A_index
    bi=B_index
    A_index=bi
    B_index=ai #поменяли местами две точки чтобы была правая тройка

A=vertices[A_index]
B=vertices[B_index]
C=vertices[C_index]

n=np.cross(B-A,C-A)/np.linalg.norm(np.cross(B-A,C-A))

n_ABC=n
# Найдены три начальные точки A B C, инициализация алгоритма завершена
#функция нахождения нормали по правой тройке
def normal_by_right_tr(A_index, B_index,C_index, vertices):
    A=vertices[A_index]
    B=vertices[B_index]
    C=vertices[C_index]
    n=np.cross(B-A,C-A)/np.linalg.norm(np.cross(B-A,C-A))
    return n

# функция нахождения следующей точки
def next_point(A_index, B_index,C_index, n_face, vertices, eps=1e-12):
   
    n_ABC = n_face
    
    best_angle = np.pi
    #best_cos=-1
    


    for i in range(len(vertices)):
        # если точка это точка плоскости - пропускаем шаг
        if i in [A_index, B_index, C_index]: 
            continue
        N = vertices[i]
        A=vertices[A_index]
        B=vertices[B_index]
        
        AN=N-A
        AB=B-A

        n_ABN = np.cross(AN,AB)/np.linalg.norm(np.cross(AN,AB))
        #n_ABN = np.cross(AB,AN)/np.linalg.norm(np.cross(AN,AB))
        
        #if np.linalg.norm(n_ABC-n_ABN)<eps:
            
            #continue

        COS = np.dot(n_ABC, n_ABN)
        angle=np.arccos(COS) # можно прописать функцию для поиска угла со знаком
        if angle < best_angle:
        #if COS>best_cos:
            best_angle = angle
            #best_cos=COS
            best_index = i
            n=n_ABN

    return best_index,n

#%% задание списков 

faces = [(A_index, B_index, C_index)]
normal_to_faces=[n_ABC]

edges_in_hull=[tuple((A_index, B_index)),tuple((B_index, C_index)),tuple((C_index, A_index))]

edges=[[tuple((A_index, B_index)),0,True]]
edges.append([tuple((B_index, C_index)),0,True])
edges.append([tuple((C_index, A_index)),0,True])
edges2=edges
max_iterations = 10000
iteration = 0

# %% построение первой грани
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Рисуем точки
ax.scatter(First_vertices[:,0], First_vertices[:,1], First_vertices[:,2], color='red', s=20)
# ax.scatter(center[0], center[1], center[2], s=40, color='green')

# Рисуем грани
poly3d = []
for face in faces:
    triangle = [First_vertices[face[0]], First_vertices[face[1]], First_vertices[face[2]]]
    poly3d.append(triangle)

ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=0.5, edgecolors='k', alpha=0.5))

# Подписи осей
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Красивый ракурс
ax.view_init(elev=10, azim=-45,roll=10)
ax.set_axis_off()
plt.savefig(r"/Users/savva/Desktop/MSU/2st sem/курсач/figure2.2.pdf")
plt.show()

# %% Построение выпуклой оболочки
while edges and iteration<max_iterations:
    
    iteration +=1

    edge = edges.pop()  
    N_index, M_index = edge[0]
    face_index=edge[1]
    edge_status=edge[2]

    if edge_status==False:
        continue

    face = faces[face_index]
    n_face=normal_to_faces[face_index]

    for k in face:
        if ((N_index!=k) and (M_index!=k)):
            L_index=k

    D_index,norm=next_point(N_index, M_index, L_index,n_face, vertices,eps=1e-12)

    faces.append((M_index, N_index, D_index))
    normal_to_faces.append(norm)

    new_face_index=len(faces)-1

    for e in [(N_index, D_index), (D_index, M_index)]:
            (F,G)=e
            e_rev=(G,F)
            if e not in edges_in_hull and e_rev not in edges_in_hull :
                edges_in_hull.append(e)
                edges.append([e,new_face_index,True])
                
            else:
                for i in range(len(edges)):
                    ed=edges[i] 
                    K,L=ed[0]
                    # Создаем оба возможных порядка ребра
                    edge_forward = (K, L)
                    edge_reverse = (L, K)
                    if (edge_forward==e or edge_reverse==e):
                        edges[i][2]=False

    if iteration == 15:
        # %%  Визуализация выпуклой оболочки
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Рисуем точки
        ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], color='red', s=20)

        #ax.scatter(center[0], center[1], center[2], s=40, color='green')

        # Рисуем грани
        poly3d = []
        for face in faces:
            triangle = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
            poly3d.append(triangle)

        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=0.5, edgecolors='k', alpha=0.5))

        # Подписи осей
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Красивый ракурс
        ax.view_init(elev=10, azim=-45,roll=10)
        ax.set_axis_off()
        plt.savefig(r"/Users/savva/Desktop/MSU/2st sem/курсач/figure3.1.pdf")
        plt.show()
    if iteration == 16:
        # %%  Визуализация выпуклой оболочки
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Рисуем точки
        ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], color='red', s=20)

        #ax.scatter(center[0], center[1], center[2], s=40, color='green')

        # Рисуем грани
        poly3d = []
        for face in faces[:-1]:
            triangle = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
            poly3d.append(triangle)

        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=0.5, edgecolors='k', alpha=0.5))
        last_face=faces[-1]
        triangle = [vertices[last_face[0]], vertices[last_face[1]], vertices[last_face[2]]]
        ax.add_collection3d(Poly3DCollection([triangle], facecolors='crimson', linewidths=0.5, edgecolors='k', alpha=0.5))
        # Подписи осей
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Красивый ракурс
        ax.view_init(elev=10, azim=-45,roll=10)
        ax.set_axis_off()
        plt.savefig(r"/Users/savva/Desktop/MSU/2st sem/курсач/figure3.2.pdf")
        plt.show()

V=set()
for edge in edges_in_hull:
    V.add(edge[0])
    V.add(edge[1])
vertices_in_hull=len(V)
faces_in_hull=len(faces)

# Отладочная печать
print(f"При построении выпуклой оболочки произведено {iteration} итераций")

# Отладочная печать
# В + Г - Р = 2
print(f"При построении выпуклой оболочки получено: ")
print(f"{vertices_in_hull} вершин")
print(f"{faces_in_hull} граней")
print(f"{len(edges_in_hull)} ребер")
print(f"Эйлерова характеристика равна {vertices_in_hull+faces_in_hull-len(edges_in_hull)} ")
print("###","###","###",sep='\n')
# %%  Визуализация выпуклой оболочки
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Рисуем точки
ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], color='red', s=20)

#ax.scatter(center[0], center[1], center[2], s=40, color='green')

# Рисуем грани
poly3d = []
for face in faces:
    triangle = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
    poly3d.append(triangle)

ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=0.5, edgecolors='k', alpha=0.5))

# Подписи осей
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Красивый ракурс
ax.view_init(elev=10, azim=-45,roll=10)
ax.set_axis_off()
plt.savefig(r"/Users/savva/Desktop/MSU/2st sem/курсач/figure4.pdf")
plt.show()
# %%  Визуализация выпуклой оболочки и двух точек
p1 = np.array([10.0, 10.0, 10.0])
p2 = np.array([-10.0, -10.0, -10.0])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Рисуем точки
ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], color='red', s=20)
ax.scatter(p1[0], p1[1], p1[2], s=60, color='green')
ax.scatter(p2[0], p2[1], p2[2], s=60, color='green')

# Рисуем грани
ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=0.5, edgecolors='k', alpha=0.5))
# Подписи осей
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Красивый ракурс
ax.view_init(elev=10, azim=-45,roll=10)
plt.savefig(r"/Users/savva/Desktop/MSU/2st sem/курсач/figure5.pdf")
plt.show()

# %% Построение выпуклой оболочки с одной новой точкой
# Зададим две точки между которыми хотим найти кратчайшее расстояние

p1 = np.array([10.0, 10.0, 10.0])
vertices = np.vstack([vertices, p1])
p1_index=len(vertices)-1

p2 = np.array([-10.0, -10.0, -10.0])
vertices = np.vstack([vertices, p2])
p2_index=len(vertices)-1

shelter = []  # Хранилище ребер: каждый элемент [ребро, флаг]
i = 0
iterations=0
it=0
while i < len(faces):
    iterations+=1
    n = normal_to_faces[i]
    A_index, B_index, C_index = faces[i]
    n1=normal_by_right_tr(A_index, B_index,C_index, vertices)
    if np.linalg.norm(n1-n)>eps:
        assert("Проблема c нормалью и правой тройкой")
    v1 = p1 - vertices[A_index]
    
    if np.dot(n, v1) > eps:
        it+=1
        # Удаляем грань
        removed_face = faces.pop(i)
        normal_to_faces.pop(i)
        A_index, B_index, C_index = removed_face
        
        # Обрабатываем каждое ребро грани
        for e in [(A_index, B_index), (B_index, C_index), (C_index, A_index)]:
            e_rev = (e[1], e[0])
            
            # Получаем все ребра из shelter для проверки
            shelter_edges = [item[0] for item in shelter]

            if e not in shelter_edges and e_rev not in shelter_edges:
                shelter.append([e,True])
            else:
                for k in range(len(shelter)):
                    edge,flag=shelter[k]
                    # Создаем оба возможных порядка ребра
                    edge_forward = edge
                    edge_reverse = (edge[1],edge[0])
                    if (edge_forward==e or edge_reverse==e):
                        shelter[k][1]=False
                        
    else:
        i += 1  # Увеличиваем только если не удаляли

# Формируем first_part из активных ребер
first_part = []
for edge_data in shelter:
    if edge_data[1]:  # если флаг True
        i, j = edge_data[0]
        first_part.append((i, j))

faces_in_first_part=[]
for edges in first_part:
    i,j=edges
    faces_in_first_part.append((i,j, p1_index))

face_in_new_hull=faces_in_first_part+faces
new_V=set()
edges_in_hull=set()
for face in face_in_new_hull:
    i,j,k=face
    new_V.add(i)
    new_V.add(j)
    new_V.add(k)
    edges_in_hull.add((i,j)); edges_in_hull.add((j,i))
    edges_in_hull.add((j,k)); edges_in_hull.add((k,j))
    edges_in_hull.add((k,i)); edges_in_hull.add((i,k))


# Отладочная печать
#В-Р+Г=2
print('Построение первого дополнения к выпуклой оболочке')
print(f"Основная часть состоит из {len(faces)} граней , дополнительная из {len(faces_in_first_part)} граней, общее количество {len(face_in_new_hull)}")

print(f"В новой структуре {int(len(edges_in_hull)/2)} ребер")
print(f"В новой структуре {len(new_V)} вершин")
print(f"Эйлерова характеристика равна {len(new_V)+len(face_in_new_hull)-int(len(edges_in_hull)/2)} ")
print("###","###","###",sep='\n')


# %%  Визуализация выпуклой оболочки с одной новой точкой

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Рисуем точки
ax.scatter(vertices[:-2,0], vertices[:-2,1], vertices[:-2,2], color='red', s=20)

ax.scatter(p1[0], p1[1], p1[2], s=60, color='green')
ax.scatter(p2[0], p2[1], p2[2], s=60, color='green')


# Рисуем грани
poly3d_main = []
poly3d_first_part=[]
for face in faces:
    triangle = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
    poly3d_main.append(triangle)

for  face in faces_in_first_part:
    triangle = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
    poly3d_first_part.append(triangle)

ax.add_collection3d(Poly3DCollection(poly3d_main, facecolors='cyan', linewidths=0.5, edgecolors='k', alpha=0.5))
ax.add_collection3d(Poly3DCollection(poly3d_first_part, facecolors='orange', linewidths=0.5, edgecolors='k', alpha=0.5))

# Подписи осей
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Красивый ракурс
ax.view_init(elev=10, azim=-45,roll=10)
ax.set_axis_off()
plt.savefig(r"/Users/savva/Desktop/MSU/2st sem/курсач/figure6.pdf")

plt.show()

# %% Построение выпуклой оболочки со второй новой точкой
 #Вторая часть         

shelter = []  # Хранилище ребер: каждый элемент [ребро, флаг]
i = 0
iterations=0
it=0

while i < len(faces):
    iterations+=1
    
    A_index, B_index, C_index = faces[i]
    n2=normal_by_right_tr(A_index, B_index,C_index, vertices)
    
    v2 = p2 - vertices[A_index]
    
    if np.dot(n2, v2) > eps:
        it+=1
        # Удаляем грань
        removed_face = faces.pop(i)
        
        A_index, B_index, C_index = removed_face
        
        # Обрабатываем каждое ребро грани
        for e in [(A_index, B_index), (B_index, C_index), (C_index, A_index)]:
            e_rev = (e[1], e[0])
            
            # Получаем все ребра из shelter для проверки
            shelter_edges = [item[0] for item in shelter]

            if e not in shelter_edges and e_rev not in shelter_edges:
                shelter.append([e,True])
            else:
                for k in range(len(shelter)):
                    edge,flag=shelter[k]
                    # Создаем оба возможных порядка ребра
                    edge_forward = edge
                    edge_reverse = (edge[1],edge[0])
                    if (edge_forward==e or edge_reverse==e):
                        shelter[k][1]=False
                        
    else:
        i += 1  # Увеличиваем только если не удаляли

# Формируем second_part из активных ребер
second_part = []
for edge_data in shelter:
    if edge_data[1]:  # если флаг True
        i, j = edge_data[0]
        second_part.append((i, j))

faces_in_second_part=[]
for edges in second_part:
    i,j=edges
    faces_in_second_part.append((i,j, p2_index))

face_in_new_hull=faces_in_second_part+faces_in_first_part+faces
new_V=set()
edges_in_hull=set()
for face in face_in_new_hull:
    i,j,k=face
    new_V.add(i)
    new_V.add(j)
    new_V.add(k)
    edges_in_hull.add((i,j)); edges_in_hull.add((j,i))
    edges_in_hull.add((j,k)); edges_in_hull.add((k,j))
    edges_in_hull.add((k,i)); edges_in_hull.add((i,k))


# Отладочная печать
#В-Р+Г=2
print('Построение второго дополнения к выпуклой оболочке')
print(f"Основная часть состоит из {len(faces+faces_in_first_part)} граней , дополнительная из {len(faces_in_second_part)} граней, общее количество {len(face_in_new_hull)}")

print(f"В новой структуре {int(len(edges_in_hull)/2)} ребер")
print(f"В новой структуре {len(new_V)} вершин")
print(f"Эйлерова характеристика равна {len(new_V)+len(face_in_new_hull)-int(len(edges_in_hull)/2)} ")
print("###","###","###",sep='\n')


# %%  Визуализация выпуклой оболочки со второй новой точкой

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Рисуем точки
ax.scatter(vertices[:-2,0], vertices[:-2,1], vertices[:-2,2], color='red', s=20)

ax.scatter(p1[0], p1[1], p1[2], s=60, color='green')
ax.scatter(p2[0], p2[1], p2[2], s=60, color='green')

# Рисуем грани
poly3d_main = []
poly3d_first_part=[]
poly3d_second_part=[]
for face in faces:
    triangle = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
    poly3d_main.append(triangle)

for  face in faces_in_first_part:
    triangle = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
    poly3d_first_part.append(triangle)

for  face in faces_in_second_part:
    triangle = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
    poly3d_second_part.append(triangle)

ax.add_collection3d(Poly3DCollection(poly3d_main, facecolors='cyan', linewidths=0.5, edgecolors='k', alpha=0.5))
ax.add_collection3d(Poly3DCollection(poly3d_first_part, facecolors='orange', linewidths=0.5, edgecolors='k', alpha=0.5))
ax.add_collection3d(Poly3DCollection(poly3d_second_part, facecolors='seagreen', linewidths=0.5, edgecolors='k', alpha=0.5))


# Подписи осей
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Красивый ракурс
ax.view_init(elev=10, azim=-45,roll=10)

ax.set_axis_off()
plt.savefig(r"/Users/savva/Desktop/MSU/2st sem/курсач/figure7.pdf")
plt.show()

# %% Поиск кратчайшего пути в новой выпуклой оболочке
faces_raw=faces+faces_in_first_part+faces_in_second_part
vert=set()
for face in faces_raw:
    i,j,k=face
    vert.add(i);    vert.add(j);    vert.add(k)
    
import pygeodesic.geodesic as geodesic


sorted_vert_list = sorted(list(vert))


V = vertices[sorted_vert_list].astype(np.float64)

#Создаем переводчик со старых индексов на новые
index_map = {}
for new_idx, old_val in enumerate(sorted_vert_list):
    index_map[int(old_val)] = new_idx


F_list = []
for face in faces_raw:
    new_face = [index_map[int(face[0])], index_map[int(face[1])], index_map[int(face[2])]]
    F_list.append(new_face)

F = np.array(F_list, dtype=np.int32)

idx_start=index_map[int(idx_start_point)]
idx_end=index_map[int(idx_end_point)]

try:
    geo = geodesic.PyGeodesicAlgorithmExact(V, F)
    distance, path = geo.geodesicDistance(idx_start, idx_end)
    
    if path is not None:
        print(f"Путь найден. Дистанция: {distance:.4f}")
        offset_path = path + (path - center) * 0.02
    else:
        print("Алгоритм вернул None. Проблема индексации.")
except Exception as e:
    print(f"Критическая ошибка: {e}")
    
# %% Визуализация кратчайшего пути

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Рисуем точки
ax.scatter(First_vertices[:,0], First_vertices[:,1], First_vertices[:,2], color='red', s=20)

# Рисуем грани

ax.add_collection3d(Poly3DCollection(poly3d_main, facecolors='cyan', linewidths=0.5, edgecolors='k', alpha=0.15))
ax.add_collection3d(Poly3DCollection(poly3d_first_part, facecolors='orange', linewidths=0.5, edgecolors='k', alpha=0.15))
ax.add_collection3d(Poly3DCollection(poly3d_second_part, facecolors='seagreen', linewidths=0.5, edgecolors='k', alpha=0.15))
# Рисуем путь
ax.plot(path[:,0], path[:,1], path[:,2], color='magenta', linewidth=4, label=f'Path: {distance:.2f}')
ax.scatter(path[1:-1,0], path[1:-1,1], path[1:-1,2], color='yellow',edgecolors='black', s=60)

# Рисуем смещенный путь
#ax.plot(offset_path[:,0], offset_path[:,1], offset_path[:,2], color='magenta', linewidth=4, zorder=10)
#ax.scatter(offset_path[1:-1,0], offset_path[1:-1,1], offset_path[1:-1,2], color='yellow',edgecolors='black', s=100)

# Выделяем старт и финиш ярче
ax.scatter(path[0,0], path[0,1], path[0,2], s=60, color='green', edgecolors='black', label='Start')
ax.scatter(path[-1,0], path[-1,1], path[-1,2], s=60, color='green', edgecolors='black', label='End')
# Подписи осей
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Красивый ракурс
ax.view_init(elev=10, azim=-45,roll=10)
ax.set_axis_off()
plt.savefig(r"/Users/savva/Desktop/MSU/2st sem/курсач/figure8.pdf")
plt.show()