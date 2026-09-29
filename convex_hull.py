import numpy as np
import pygeodesic.geodesic as geodesic

class ConvexHull:
    def __init__(self, points: np.ndarray,save_iteration=None, max_iterations=10000,eps=1e-12):

        self.points = np.asarray(points)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
                    raise ValueError("points должен иметь форму (N, 3)")
                
        if len(self.points) < 4:
            raise ValueError("Для 3D выпуклой оболочки нужно минимум 4 точки")
        self.save_iteration=save_iteration

        self.max_iterations=max_iterations
        self.eps=eps
        self.n = len(self.points)
        self.faces,self.normal_to_faces, self.V, self.E, self.F = self._build_hull()

    def _build_hull(self):
        A_index, B_index, C_index,n_ABC= self._find_initial_points()
        faces = [(A_index, B_index, C_index)]
        normal_to_faces=[n_ABC]

        edges_in_hull=[tuple((A_index, B_index)),tuple((B_index, C_index)),tuple((C_index, A_index))]

        edges=[[tuple((A_index, B_index)),0,True]]
        edges.append([tuple((B_index, C_index)),0,True])
        edges.append([tuple((C_index, A_index)),0,True])
        iteration = 0
        # Построение выпуклой оболочки
        while edges and iteration<self.max_iterations:
            
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

            D_index,norm=self._next_point(N_index, M_index, L_index,n_face)

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

            if iteration == self.save_iteration:
                self.saved_faces_1=faces.copy()
    
            if iteration == self.save_iteration+1:
                self.saved_faces_2=faces.copy()

        self.iterations=iteration
        V=set()
        for edge in edges_in_hull:
            V.add(edge[0])
            V.add(edge[1])
        vertices_in_hull=len(V)
        faces_in_hull=len(faces)

        return faces,normal_to_faces, vertices_in_hull,len(edges_in_hull),faces_in_hull 
    
    def _find_initial_points(self):
        ez=np.array([0,0,1]) # нормаль к oXY
        A_index = np.argmin(self.points[:,2])
        A = self.points[A_index]
        # Нашли минимальную по координате z точку

        # Найдем коорддинаты второй точки 

        cos_alpha=1

        for i in range(self.n):
            if i==A_index:
                continue
            N=self.points[i]
            NA=A-N
            if np.linalg.norm(NA)<self.eps:
                continue
            cos_alpha_i=(np.dot(NA, -ez))/np.linalg.norm(NA)
            if (cos_alpha_i<cos_alpha):
                B=N
                B_index=i
                cos_alpha=cos_alpha_i
        AB=B-A
        # Найдем третью точку

        cos_beta=-1

        for i in range(self.n):
            if i == A_index or i==B_index:
                continue

            N=self.points[i]
            BN=N-B

            if np.linalg.norm(np.cross(AB, BN))<self.eps:
                continue

            n_ABN=(np.cross(AB, BN))/np.linalg.norm(np.cross(AB, BN))

            if n_ABN[2]<0:
                n_ABN=-n_ABN

            cos_beta_i=(np.dot(n_ABN, ez))
            if (cos_beta_i>cos_beta):
                cos_beta=cos_beta_i
                C=N
                C_index=i



        # чтобы у нас была нормаль смотрящая наружу и правая тройка точек (при виде из вне обход против часовой стрелки)
            
        A=self.points[A_index]
        B=self.points[B_index]
        C=self.points[C_index]

        n=np.cross(B-A,C-A)/np.linalg.norm(np.cross(B-A,C-A))

        if np.dot(n,ez)>0:
            ai=A_index
            bi=B_index
            A_index=bi
            B_index=ai #поменяли местами две точки чтобы была правая тройка

        A=self.points[A_index]
        B=self.points[B_index]
        C=self.points[C_index]

        n=np.cross(B-A,C-A)/np.linalg.norm(np.cross(B-A,C-A))
        # Найдены три начальные точки A B C, инициализация алгоритма завершена
        return A_index, B_index, C_index, n

    #функция нахождения нормали по правой тройке
    def _normal_by_right_tr(self, A_index: int, B_index: int,C_index: int):
        A=self.points[A_index]
        B=self.points[B_index]
        C=self.points[C_index]
        n=np.cross(B-A,C-A)/np.linalg.norm(np.cross(B-A,C-A))
        return n

    # функция нахождения следующей точки
    def _next_point(self, A_index:int, B_index:int,C_index:int, n_face: np.ndarray,):
        n_ABC = n_face
        
        best_angle = np.pi
        #best_cos=-1
        


        for i in range(self.n):
            # если точка это точка плоскости - пропускаем шаг
            if i in [A_index, B_index, C_index]: 
                continue
            N = self.points[i]
            A=self.points[A_index]
            B=self.points[B_index]
            
            AN=N-A
            AB=B-A

            cross = np.cross(AN, AB)

            if np.linalg.norm(cross) < self.eps:
                continue

            n_ABN = cross / np.linalg.norm(cross)

            COS = np.dot(n_ABC, n_ABN)
            angle=np.arccos(COS) # можно прописать функцию для поиска угла со знаком
            if angle < best_angle:
            #if COS>best_cos:
                best_angle = angle
                #best_cos=COS
                best_index = i
                n=n_ABN

        return best_index,n


    def add_one_point(self, p1:np.ndarray):

        self.points = np.vstack([self.points, p1])
        self.p1_index=self.n
        self.n+=1
        p1_index=len(self.points)-1

        shelter = []  # Хранилище ребер: каждый элемент [ребро, флаг]
        i = 0
        iterations=0
        while i < len(self.faces):
            iterations+=1
            n = self.normal_to_faces[i]
            A_index, B_index, C_index = self.faces[i]
            n1=self._normal_by_right_tr(A_index, B_index,C_index)
            if np.linalg.norm(n1-n)>self.eps:
                assert("Проблема c нормалью и правой тройкой")
            v1 = p1 - self.points[A_index]
            
            if np.dot(n, v1) > self.eps:
            
                # Удаляем грань
                removed_face = self.faces.pop(i)
                self.normal_to_faces.pop(i)
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
        normal_to_faces_in_first_part=[]
        for edges in first_part:
            i,j=edges
            faces_in_first_part.append((i,j, p1_index))
            # добавим нормали
            normal_to_faces_in_first_part.append(self._normal_by_right_tr(i, j,p1_index))
        prev_faces=self.faces.copy()
        self.faces=prev_faces+faces_in_first_part
        self.normal_to_faces=self.normal_to_faces+normal_to_faces_in_first_part
        new_V=set()
        edges_in_hull=set()
        for face in self.faces:
            i,j,k=face
            new_V.add(i)
            new_V.add(j)
            new_V.add(k)
            edges_in_hull.add((i,j)); edges_in_hull.add((j,i))
            edges_in_hull.add((j,k)); edges_in_hull.add((k,j))
            edges_in_hull.add((k,i)); edges_in_hull.add((i,k))

        self.V, self.E, self.F =len(new_V),len(edges_in_hull)//2,len(self.faces)

        return prev_faces,faces_in_first_part
    
    def find_shortest_path(self,p1_index:int, p2_index:int):
        center=np.mean(self.points,axis=0)
        vert=set()
        for face in self.faces:
            i,j,k=face
            vert.add(i);    vert.add(j);    vert.add(k)
        sorted_vert_list = sorted(list(vert))

        V = self.points[sorted_vert_list].astype(np.float64)

        #Создаем переводчик со старых индексов на новые
        index_map = {}
        for new_idx, old_val in enumerate(sorted_vert_list):
            index_map[int(old_val)] = new_idx


        F_list = []
        for face in self.faces:
            new_face = [index_map[int(face[0])], index_map[int(face[1])], index_map[int(face[2])]]
            F_list.append(new_face)

        F = np.array(F_list, dtype=np.int32)

        idx_start=index_map[p1_index]
        idx_end=index_map[p2_index]

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

        return offset_path

    def isConvexHullObstacle(self,A: np.ndarray,B: np.ndarray,) -> bool:

        direction = B - A

        # Изначально рассматриваем весь отрезок:
        # 0 <= t <= 1
        t_min = 0.0
        t_max = 1.0

        for face, normal in zip(self.faces, self.normal_to_faces):

            # Любая точка грани
            p0 = self.points[face[0]]

            # Уравнение плоскости:
            # normal · x + d = 0
            d = -np.dot(normal, p0)

            # Значение выражения в A:
            f_A = np.dot(normal, A) + d

            # Коэффициент при t:
            f_direction = np.dot(normal, direction)

            # Отрезок параллелен плоскости грани
            if abs(f_direction) < self.eps:

                # Весь отрезок находится снаружи
                if f_A > self.eps:
                    return False

                # Эта грань больше не ограничивает t
                continue

            # Точка пересечения прямой с плоскостью грани
            t = -f_A / f_direction

            if f_direction < 0:
                # После этой точки движемся внутрь
                t_min = max(t_min, t)

            else:
                # После этой точки движемся наружу
                t_max = min(t_max, t)

            # Допустимый интервал стал пустым
            if t_min > t_max + self.eps:
                return False

        # Явно пересекаем полученный интервал с (0, 1)
        left = max(t_min, 0.0)
        right = min(t_max, 1.0)

        # Есть ненулевой участок отрезка внутри оболочки
        return left < right - self.eps




