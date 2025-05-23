import random
import math
import numpy as np
from collections import defaultdict

def nearest_neighbor(graph, start_node):
    """Finds the nearest neighbor path."""
    graph_copy = graph.copy()

    if graph_copy is None or not graph_copy.nodes() or start_node is None or start_node not in graph_copy.nodes():
        return None, 0

    nodes = list(graph_copy.nodes())
    path = [start_node]
    unvisited = set(nodes)
    unvisited.remove(start_node)
    total_cost = 0

    while unvisited:
        current_node = path[-1]
        nearest_neighbor = None
        min_distance = float('inf')

        for neighbor in unvisited:
            try:
                distance = graph_copy[current_node][neighbor]['weight']
            except KeyError:
                distance = float('inf')

            if distance < min_distance:
                nearest_neighbor = neighbor
                min_distance = distance

        if nearest_neighbor is None:
            return None, 0

        path.append(nearest_neighbor)
        unvisited.remove(nearest_neighbor)
        total_cost += min_distance

    try:
        total_cost += graph_copy[path[-1]][path[0]]['weight']
        path.append(path[0])
    except KeyError:
        return None, 0

    return path, total_cost

class AntColonyOptimizer:
    def __init__(self, graph, n_ants, n_iterations, 
                 alpha, beta, evaporation_rate, 
                 q, use_templates, template_update_frequency=10):  # Добавим частоту обновления шаблонов
        self.graph = graph
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # влияние феромона
        self.beta = beta    # влияние эвристики (1/расстояние)
        self.evaporation_rate = evaporation_rate
        self.q = q  # константа для обновления феромона
        self.use_templates = use_templates
        self.template_update_frequency = template_update_frequency # Как часто обновлять шаблоны
        
        self.nodes = list(graph.nodes())
        self.n_nodes = len(self.nodes)
        self.node_indices = {node: i for i, node in enumerate(self.nodes)}
        
        # Инициализация матрицы феромонов
        self.pheromone = np.ones((self.n_nodes, self.n_nodes))
        
        # Инициализация матрицы расстояний
        self.distances = np.zeros((self.n_nodes, self.n_nodes))
        for i, u in enumerate(self.nodes):
            for j, v in enumerate(self.nodes):
                if u == v:
                    self.distances[i,j] = 0
                else:
                    try:
                        self.distances[i,j] = graph[u][v]['weight']
                    except KeyError:
                        self.distances[i,j] = float('inf')
        
        # Шаблоны для модификации (изначально пустой список)
        self.templates = [] if use_templates else None  
        self.edge_counts = defaultdict(int) # Словарь для хранения количества вхождений ребер
        self.iteration_count = 0

    def _generate_template_from_path(self, path):
        """Создает шаблон из данного пути."""
        # Можно добавить логику для фильтрации или упрощения шаблона,
        # чтобы избежать слишком специфичных шаблонов.
        return path[:]  # Возвращаем копию пути, чтобы избежать изменений исходного пути

    def _update_templates(self, ants_paths):
        """Обновляет шаблоны на основе путей, пройденных муравьями."""
        if not self.use_templates:
            return

        for path, cost in ants_paths:
            if cost > 0: # Исключаем невалидные пути
                # Создаем шаблон из пути
                new_template = self._generate_template_from_path(path)

                # Добавляем шаблон, если он еще не существует
                if new_template not in self.templates:
                    self.templates.append(new_template)

                # Ограничиваем количество шаблонов (опционально)
                max_templates = 10 # Максимальное количество шаблонов
                if len(self.templates) > max_templates:
                    # Удаляем наименее полезный шаблон (здесь - случайный)
                    self.templates.pop(random.randint(0, len(self.templates) - 1))
    
    def _calculate_path_cost(self, path):
        """Вычисляет стоимость пути."""
        cost = 0
        for i in range(len(path)-1):
            u = path[i]
            v = path[i+1]
            cost += self.distances[self.node_indices[u], self.node_indices[v]]
        return cost
    
    def _select_next_node(self, current_node, visited):
        """Выбирает следующую вершину для муравья."""
        current_idx = self.node_indices[current_node]
        unvisited = [node for node in self.nodes if node not in visited]
        
        if not unvisited:
            return None
        
        # Вычисляем вероятности для всех непосещенных вершин
        probabilities = []
        total = 0.0
        
        for node in unvisited:
            node_idx = self.node_indices[node]
            if self.distances[current_idx, node_idx] == float('inf'):
                probabilities.append(0.0)
                continue
                
            pheromone = self.pheromone[current_idx, node_idx] ** self.alpha
            heuristic = (1.0 / self.distances[current_idx, node_idx]) ** self.beta
            prob = pheromone * heuristic
            probabilities.append(prob)
            total += prob
        
        if total == 0:
            return random.choice(unvisited)
            
        # Нормализуем вероятности
        probabilities = [p/total for p in probabilities]
        
        # Выбираем следующую вершину согласно вероятностям
        return np.random.choice(unvisited, p=probabilities)
    
    def _update_pheromone(self, ants_paths):
        """Обновляет матрицу феромонов, учитывая частоту ребер."""
        # Испарение феромона
        self.pheromone *= (1 - self.evaporation_rate)

        # Сброс edge_counts перед каждым обновлением феромонов
        self.edge_counts.clear()

        # Подсчет вхождений ребер в маршруты
        for path, _ in ants_paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                # Учитываем ребро только в одном направлении для ориентированного графа
                self.edge_counts[(u, v)] += 1
        
        # Добавление нового феромона
        for path, cost in ants_paths:
            if cost == 0:
                continue

            delta_pheromone = self.q / cost
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                u_idx, v_idx = self.node_indices[u], self.node_indices[v]

                # Учитываем частоту ребра при обновлении феромона
                edge_frequency = self.edge_counts[(u, v)]
                self.pheromone[u_idx, v_idx] += delta_pheromone * (1 + edge_frequency) # Умножаем на (1 + частота)
    
    def _apply_template_modification(self, path):
        """Применяет модификацию шаблонов к пути."""
        if not self.templates:
            return path
            
        # Выбираем случайный шаблон
        template = random.choice(self.templates)
        
        # Смешиваем путь с шаблоном
        mixed_path = []
        template_set = set(template)
        path_set = set(path[:-1])  # исключаем последнюю вершину (она совпадает с первой)
        
        # Добавляем вершины, которые есть и в пути, и в шаблоне
        common = list(path_set & template_set)
        random.shuffle(common)
        mixed_path.extend(common)
        
        # Добавляем оставшиеся вершины из пути
        remaining = [node for node in path[:-1] if node not in mixed_path]
        mixed_path.extend(remaining)
        
        # Замыкаем цикл
        mixed_path.append(mixed_path[0])
        
        return mixed_path
    
    def run(self):  # Убираем параметр start_node
        """Запускает алгоритм муравьиной колонии."""
        start_node = random.choice(self.nodes) # Выбираем случайную начальную вершину
        
        best_path = None
        best_cost = float('inf')
        
        for i in range(self.n_iterations):
            self.iteration_count = i
            ants_paths = []
            
            for _ in range(self.n_ants):
                # Муравей начинает путь со стартовой вершины
                current_node = start_node
                visited = {current_node}
                path = [current_node]
                
                # Строим путь
                while len(visited) < self.n_nodes:
                    next_node = self._select_next_node(current_node, visited)
                    if next_node is None:
                        break
                    
                    path.append(next_node)
                    visited.add(next_node)
                    current_node = next_node
                
                # Замыкаем цикл
                if len(path) == self.n_nodes and self.distances[self.node_indices[path[-1]], self.node_indices[path[0]]] != float('inf'):
                    path.append(path[0])
                    cost = self._calculate_path_cost(path)
                    
                    # Применяем модификацию шаблонов
                    if self.use_templates and random.random() < 0.3:  # 30% chance to apply template
                        new_path = self._apply_template_modification(path)
                        new_cost = self._calculate_path_cost(new_path)
                        if new_cost < cost:
                            path = new_path
                            cost = new_cost
                    
                    ants_paths.append((path, cost))
                    
                    if cost < best_cost:
                        best_path = path
                        best_cost = cost
            
            if ants_paths:
                self._update_pheromone(ants_paths)

            # Обновляем шаблоны с заданной частотой
            if self.use_templates and (i + 1) % self.template_update_frequency == 0:
                self._update_templates(ants_paths)
        
        return best_path, best_cost

def ant_colony(graph, n_ants, n_iterations, 
               alpha, beta, evaporation_rate, 
               q, use_templates, template_update_frequency=10):
    """Интерфейс для вызова муравьиного алгоритма."""
    if not graph.nodes():
        return None, 0
    
    aco = AntColonyOptimizer(graph, n_ants, n_iterations, alpha, beta, 
                            evaporation_rate, q, use_templates, template_update_frequency)
    return aco.run()  #  Больше не передаем start_node

def calculate_total_cost(graph, path):
    """Calculates path cost, returns None if any edge is missing."""
    total_cost = 0
    for i in range(len(path)-1):
        if not graph.has_edge(path[i], path[i+1]):
            return None
        total_cost += graph[path[i]][path[i+1]]['weight']
    
    if not graph.has_edge(path[-1], path[0]):
        return None
    total_cost += graph[path[-1]][path[0]]['weight']
    return total_cost