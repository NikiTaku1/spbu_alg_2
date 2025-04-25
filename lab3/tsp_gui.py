import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import tkinter.simpledialog
import networkx as nx
import math
from tkinter import filedialog
import pandas as pd
import random
import time
from tsp_solver import ant_colony, nearest_neighbor, calculate_total_cost

class TSPInteractiveGUI:
    def __init__(self, master):
        self.master = master
        master.title("Интерактивная задача коммивояжера (Муравьиный алгоритм)")

        # Graph Data Structures
        self.graph = nx.DiGraph()
        self.vertices = {}
        self.edges = {}
        self.edge_objects = {}
        self.edge_text_objects = {}
        self.start_vertex = None
        self.vertex_radius = 15
        self.edge_line_width = 2
        self.edge_color = "gray"
        self.vertex_color = "skyblue"
        self.curve_offset = 20
        self.arrow_size = 10
        self.path_objects = []
        self.best_path = None
        self.best_cost = None
        self.last_edge = None
        self.edge_history = []

        self.selected_edge = None

        # Default parameters for Ant Colony
        self.n_ants = 10
        self.n_iterations = 1000
        self.alpha = 1.0
        self.beta = 3.0
        self.evaporation_rate = 0.3
        self.q = 100
        
        # Main Frame
        self.main_frame = ttk.Frame(master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create GUI Elements
        self.create_graph_canvas(self.main_frame)
        self.create_distance_matrix_frame(self.main_frame)
        self.create_control_frame(master)
        self.create_result_frame(master)

        self.path = None
        self.selected_vertices = []



    def create_graph_canvas(self, master):
        """Создает поле для интерактивного рисования графа."""
        self.canvas_width = 600
        self.canvas_height = 300

        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10, sticky=tk.NSEW)
        master.columnconfigure(0, weight=1)

        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def create_distance_matrix_frame(self, master):
        """Создает фрейм для отображения матрицы расстояний."""
        self.matrix_frame = ttk.Frame(master)
        self.matrix_frame.grid(row=0, column=1, padx=10, pady=10, sticky=tk.NSEW)

        self.matrix_text = tk.Text(self.matrix_frame, width=30, height=10, font=('Courier', 10))
        self.matrix_text.pack(fill=tk.BOTH, expand=True)
        self.matrix_text.config(state=tk.DISABLED)
        master.columnconfigure(1, weight=2)

    def update_distance_matrix(self):
        """Обновляет отображение матрицы расстояний."""
        self.matrix_text.config(state=tk.NORMAL)
        self.matrix_text.delete("1.0", tk.END)

        vertices = list(self.vertices.keys())
        if not vertices:
            self.matrix_text.config(state=tk.DISABLED)
            return

        max_vertex_length = max(len(v) for v in vertices)
        header = " " * (max_vertex_length + 1) + "".join([f"{v:<4}" for v in vertices]) + "\n"
        self.matrix_text.insert("1.0", header)

        reversed_vertices = list(reversed(vertices))

        for v1 in reversed_vertices:
            row = f"{v1:<{max_vertex_length}} "
            for v2 in vertices:
                try:
                    weight = self.graph[v1][v2]['weight']
                    row += f"{weight:<4}"
                except KeyError:
                    row += "i   "
            row += "\n"
            self.matrix_text.insert("2.0", row)

        self.matrix_text.config(state=tk.DISABLED)

    def create_control_frame(self, master):
        control_frame = ttk.Frame(master)
        control_frame.pack(padx=10, pady=5, fill=tk.X)

        general_frame = ttk.LabelFrame(control_frame, text="Основные команды")
        general_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        run_button = ttk.Button(general_frame, text="Найти кратчайший путь", command=self.run_ant_colony)
        run_button.grid(row=0, column=0, padx=5, pady=2)

        clear_button = ttk.Button(general_frame, text="Очистить поле", command=self.clear_graph)
        clear_button.grid(row=0, column=1, padx=5, pady=2)

        delete_vertex_button = ttk.Button(general_frame, text="Удалить последнюю вершину", command=self.delete_last_vertex)
        delete_vertex_button.grid(row=1, column=0, padx=5, pady=2)

        delete_path_button = ttk.Button(general_frame, text="Удалить последнее ребро", command=self.delete_last_edge)
        delete_path_button.grid(row=1, column=1, padx=5, pady=2)

        start_vertex_frame = ttk.LabelFrame(control_frame, text="Начальная вершина")
        start_vertex_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        self.start_vertex_label = ttk.Label(start_vertex_frame, text="Начальная вершина:")
        self.start_vertex_label.grid(row=0, column=0, padx=5, pady=2)

        self.start_vertex_entry = ttk.Entry(start_vertex_frame, width=5)
        self.start_vertex_entry.grid(row=0, column=1, padx=5, pady=2)

        self.set_start_vertex_button = ttk.Button(start_vertex_frame, text="Установить", command=self.set_start_vertex)
        self.set_start_vertex_button.grid(row=0, column=2, padx=5, pady=2)

        load_matrix_frame = ttk.LabelFrame(control_frame, text="Загрузка матрицы")
        load_matrix_frame.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        load_matrix_button = ttk.Button(load_matrix_frame, text="Загрузить матрицу из таблицы", command=self.load_matrix_from_table)
        load_matrix_button.grid(row=0, column=0, padx=5, pady=2)

        edge_weight_frame = ttk.LabelFrame(control_frame, text="Вес ребра")
        edge_weight_frame.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

        ttk.Label(edge_weight_frame, text="Ребро:").grid(row=0, column=0, padx=5, pady=2)
        self.edge_entry = ttk.Entry(edge_weight_frame, width=5)
        self.edge_entry.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(edge_weight_frame, text="Вес ребра:").grid(row=1, column=0, padx=5, pady=2)
        self.edge_weight_entry = ttk.Entry(edge_weight_frame, width=5)
        self.edge_weight_entry.grid(row=1, column=1, padx=5, pady=2)

        self.update_edge_weight_button = ttk.Button(edge_weight_frame, text="Изменить", command=self.update_edge_weight)
        self.update_edge_weight_button.grid(row=1, column=2, padx=5, pady=2)

        # Параметры муравьиного алгоритма
        aco_params_frame = ttk.LabelFrame(control_frame, text="Параметры муравьиного алгоритма")
        aco_params_frame.grid(row=0, column=1, rowspan=4, padx=5, pady=5, sticky="nsew")

        ttk.Label(aco_params_frame, text="Количество муравьев:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.E)
        self.n_ants_entry = ttk.Entry(aco_params_frame, width=5)
        self.n_ants_entry.insert(0, str(self.n_ants))
        self.n_ants_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(aco_params_frame, text="Количество итераций:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.E)
        self.n_iterations_entry = ttk.Entry(aco_params_frame, width=5)
        self.n_iterations_entry.insert(0, str(self.n_iterations))
        self.n_iterations_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(aco_params_frame, text="Феромоны:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.E)
        self.alpha_entry = ttk.Entry(aco_params_frame, width=5)
        self.alpha_entry.insert(0, str(self.alpha))
        self.alpha_entry.grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(aco_params_frame, text="Эвристика:").grid(row=3, column=0, padx=5, pady=2, sticky=tk.E)
        self.beta_entry = ttk.Entry(aco_params_frame, width=5)
        self.beta_entry.insert(0, str(self.beta))
        self.beta_entry.grid(row=3, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(aco_params_frame, text="Испарение:").grid(row=4, column=0, padx=5, pady=2, sticky=tk.E)
        self.evaporation_entry = ttk.Entry(aco_params_frame, width=5)
        self.evaporation_entry.insert(0, str(self.evaporation_rate))
        self.evaporation_entry.grid(row=4, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(aco_params_frame, text="Константа для феромонов:").grid(row=5, column=0, padx=5, pady=2, sticky=tk.E)
        self.q_entry = ttk.Entry(aco_params_frame, width=5)
        self.q_entry.insert(0, str(self.q))
        self.q_entry.grid(row=5, column=1, padx=5, pady=2, sticky=tk.W)

        self.modification_var = tk.BooleanVar()
        modification_check = ttk.Checkbutton(aco_params_frame, text="Использовать шаблоны", variable=self.modification_var)
        modification_check.grid(row=6, column=0, columnspan=2, padx=5, pady=2)

        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)

        aco_params_frame.columnconfigure(0, weight=1)
        aco_params_frame.columnconfigure(1, weight=1)

        general_frame.columnconfigure(0, weight=1)
        general_frame.columnconfigure(1, weight=1)

        start_vertex_frame.columnconfigure(0, weight=1)
        start_vertex_frame.columnconfigure(1, weight=1)
        start_vertex_frame.columnconfigure(2, weight=1)

        load_matrix_frame.columnconfigure(0, weight=1)

        edge_weight_frame.columnconfigure(0, weight=1)
        edge_weight_frame.columnconfigure(1, weight=1)
        edge_weight_frame.columnconfigure(2, weight=1)


    def create_result_frame(self, master):
        result_frame = ttk.LabelFrame(master, text="Результат")
        result_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        ttk.Label(result_frame, text="Кратчайший путь:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.path_text = tk.Text(result_frame, height=3, width=50)
        self.path_text.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        self.path_text.config(state=tk.DISABLED)

        ttk.Label(result_frame, text="Общая длина:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.length_label = ttk.Label(result_frame, text="")
        self.length_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(result_frame, text="Время расчета:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.time_label = ttk.Label(result_frame, text="")
        self.time_label.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        result_frame.columnconfigure(1, weight=1)

    def on_canvas_click(self, event):
        x, y = event.x, event.y
        closest_vertex = self.find_closest_vertex(x, y)

        if closest_vertex:
            self.selected_vertices.append(closest_vertex)

            if len(self.selected_vertices) == 2:
                v1, v2 = self.selected_vertices[0], self.selected_vertices[1]
                self.selected_vertices = []

                if v1 == v2:
                    messagebox.showinfo("Информация", "Нельзя создать ребро между одной и той же вершиной.")
                else:
                    weight = self.ask_for_weight()
                    if weight is not None:
                        if (v1, v2) not in self.edges:
                            self.edges[(v1, v2)] = []
                            self.edge_objects[(v1,v2)] = []
                            self.edge_text_objects[(v1, v2)] = []
                        self.graph.add_edge(v1, v2, weight=weight)
                        self.edges[(v1, v2)].append(weight)
                        edge_id, text_id = self.draw_edge(v1, v2, weight, len(self.edges[(v1, v2)])-1)
                        self.edge_objects[(v1,v2)].append(edge_id)
                        self.edge_text_objects[(v1, v2)].append(text_id)
                        edge = (v1, v2)
                        self.edge_history.append(edge)
                        self.update_distance_matrix()
        else:
            self.add_vertex_event(event)

    def add_vertex_event(self, event):
        x, y = event.x, event.y
        vertex_name = chr(65 + len(self.vertices))

        x1, y1 = x - self.vertex_radius, y - self.vertex_radius
        x2, y2 = x + self.vertex_radius, y + self.vertex_radius
        vertex_id = self.canvas.create_oval(x1, y1, x2, y2, fill=self.vertex_color, outline="black")
        text_id = self.canvas.create_text(x, y, text=vertex_name, fill="black", font=('Arial', 10, 'bold'))

        self.vertices[vertex_name] = {"x": x, "y": y, "oval_id": vertex_id, "text_id": text_id}
        self.graph.add_node(vertex_name)
        self.update_distance_matrix()

    def get_midpoint_coordinates(self, x1, y1, ctrl_x1, ctrl_y1, ctrl_x2, ctrl_y2, x2, y2):
        """Вычисляет координаты середины линии."""
        mid_x = (x1 + x2 + ctrl_x1 + ctrl_x2) / 4
        mid_y = (y1 + y2 + ctrl_y1 + ctrl_y2) / 4
        return mid_x, mid_y

    def draw_arrow(self, x, y, angle, color):
        """Рисует стрелку-треугольник."""
        arrow_size = self.arrow_size
        points = [
            x + arrow_size * math.cos(angle), y + arrow_size * math.sin(angle),
            x + arrow_size/2 * math.cos(angle + 2 * math.pi / 3), y + arrow_size/2 * math.sin(angle + 2 * math.pi / 3),
            x + arrow_size/2 * math.cos(angle + 4 * math.pi / 3), y + arrow_size/2 * math.sin(angle + 4 * math.pi / 3)
        ]
        arrow_id = self.canvas.create_polygon(points, fill=color, outline=color)
        return arrow_id

    def draw_edge(self, v1, v2, weight, edge_index):
      """Рисует ребро между двумя вершинами."""
      x1, y1 = self.vertices[v1]["x"], self.vertices[v1]["y"]
      x2, y2 = self.vertices[v2]["x"], self.vertices[v2]["y"]

      dx = x2 - x1
      dy = y2 - y1
      distance = math.sqrt(dx**2 + dy**2)
      if distance == 0:
          return

      dx /= distance
      dy /= distance

      offset_x = dy * self.curve_offset * (edge_index - 1)
      offset_y = -dx * self.curve_offset * (edge_index - 1)

      ctrl_x1 = x1 + offset_x
      ctrl_y1 = y1 + offset_y
      ctrl_x2 = x2 + offset_x
      ctrl_y2 = y2 + offset_y

      edge_id = self.canvas.create_line(x1, y1, ctrl_x1, ctrl_y1, ctrl_x2, ctrl_y2, x2, y2,
                                           smooth=True,
                                           fill=self.edge_color, width=self.edge_line_width)
      mid_x, mid_y = self.get_midpoint_coordinates(x1, y1, ctrl_x1, ctrl_y1, ctrl_x2, ctrl_y2, x2, y2)

      angle = math.atan2(y2 - y1, x2 - x1)
      self.draw_arrow(mid_x, mid_y, angle, self.edge_color)

      text_id = self.canvas.create_text(mid_x , mid_y - 10, text=str(weight), fill="black", font=('Arial', 8))

      return edge_id, text_id

    def find_closest_vertex(self, x, y, max_distance=20):
        closest_vertex = None
        min_distance = max_distance

        for vertex, data in self.vertices.items():
            vx, vy = data["x"], data["y"]
            distance = ((x - vx)**2 + (y - vy)**2)**0.5
            if distance < min_distance:
                closest_vertex = vertex
                min_distance = distance

        return closest_vertex

    def ask_for_weight(self, ):
        weight = None
        while weight is None:
            try:
                weight_str = tk.simpledialog.askstring("Ввод веса", "Введите вес ребра:")
                if weight_str is None:
                    return None
                weight = int(weight_str)
                if weight <= 0:
                    messagebox.showerror("Ошибка ввода", "Вес должен быть положительным числом.")
                    weight = None
            except ValueError:
                messagebox.showerror("Ошибка ввода", "Вес должен быть целым числом.")
                weight = None
        return weight

    def set_start_vertex(self):
        start_vertex = self.start_vertex_entry.get()
        if start_vertex in self.vertices:
            self.start_vertex = start_vertex
            messagebox.showinfo("Информация", f"Начальная вершина установлена в {start_vertex}")
            self.draw_graph()
        else:
            messagebox.showerror("Ошибка", "Неверная начальная вершина.  Выберите существующую вершину.")
            self.start_vertex = None

    def delete_last_vertex(self):
        if self.vertices:
            vertex_to_delete = list(self.vertices.keys())[-1]

            edges_to_delete = []
            for (v1, v2) in self.edges:
                if v1 == vertex_to_delete or v2 == vertex_to_delete:
                    edges_to_delete.append((v1, v2))

            for v1, v2 in edges_to_delete:
                if (v1, v2) in self.edge_objects:
                    for edge_id in self.edge_objects[(v1, v2)]:
                        self.canvas.delete(edge_id)
                    del self.edge_objects[(v1, v2)]
                if (v1, v2) in self.edge_text_objects:
                  for text_id in self.edge_text_objects[(v1, v2)]:
                    self.canvas.delete(text_id)
                  del self.edge_text_objects[(v1, v2)]
                if (v1, v2) in self.edges:
                    del self.edges[(v1, v2)]
                if self.graph.has_edge(v1, v2):
                    self.graph.remove_edge(v1, v2)
            self.edge_history = [(v1, v2) for v1, v2 in self.edge_history if v1 != vertex_to_delete and v2 != vertex_to_delete]
            self.canvas.delete(self.vertices[vertex_to_delete]["oval_id"])
            self.canvas.delete(self.vertices[vertex_to_delete]["text_id"])

            del self.vertices[vertex_to_delete]
            self.graph.remove_node(vertex_to_delete)
            self.update_distance_matrix()
            self.draw_graph()

            if self.start_vertex == vertex_to_delete:
                self.start_vertex = None

    def draw_graph(self):
        self.canvas.delete("all")

        for (v1, v2), weights in self.edges.items():
            for i, weight in enumerate(weights):
                edge_id, text_id = self.draw_edge(v1,v2, weight, i)
                self.edge_objects[(v1,v2)].append(edge_id)
                self.edge_text_objects[(v1, v2)].append(text_id)

        for vertex, data in self.vertices.items():
            x, y = data["x"], data["y"]
            x1, y1 = x - self.vertex_radius, y - self.vertex_radius
            x2, y2 = x + self.vertex_radius, y + self.vertex_radius
            oval_id = self.canvas.create_oval(x1, y1, x2, y2, fill=self.vertex_color, outline="black")
            text_id = self.canvas.create_text(x, y, text=vertex, fill="black", font=('Arial', 10, 'bold'))
            self.vertices[vertex]["oval_id"] = oval_id
            self.vertices[vertex]["text_id"] = text_id

    def clear_graph(self):
        self.graph.clear()
        self.vertices.clear()
        self.edges.clear()
        self.edge_objects.clear()
        self.edge_text_objects.clear()
        self.start_vertex = None
        self.path = None
        self.path_text.config(state=tk.NORMAL)
        self.path_text.delete("1.0", tk.END)
        self.path_text.config(state=tk.DISABLED)
        self.length_label.config(text="")
        self.time_label.config(text="")
        self.canvas.delete("all")
        self.update_distance_matrix()

    def get_aco_parameters(self):
        """Получает параметры муравьиного алгоритма из полей ввода."""
        try:
            self.n_ants = int(self.n_ants_entry.get())
            self.n_iterations = int(self.n_iterations_entry.get())
            self.alpha = float(self.alpha_entry.get())
            self.beta = float(self.beta_entry.get())
            self.evaporation_rate = float(self.evaporation_entry.get())
            self.q = float(self.q_entry.get())

            if (self.n_ants <= 0 or self.n_iterations <= 0 or self.alpha <= 0 or 
                self.beta <= 0 or self.evaporation_rate <= 0 or self.evaporation_rate >= 1 or 
                self.q <= 0):
                messagebox.showerror("Ошибка", "Все параметры должны быть положительными числами. Испарение должно быть между 0 и 1.")
                return False

            return True
        except ValueError:
            messagebox.showerror("Ошибка", "Пожалуйста, введите корректные числовые значения для параметров.")
            return False

    def run_ant_colony(self):
        """Запускает муравьиный алгоритм для решения задачи коммивояжера."""
        if not self.get_aco_parameters():
            return

        if not self.graph.nodes():
            messagebox.showerror("Ошибка", "Пожалуйста, создайте граф.")
            return

        nodes = list(self.graph.nodes())
        if self.start_vertex is None or self.start_vertex not in nodes:
            self.start_vertex = random.choice(nodes)

        self.clear_temporary_edge()
        self.delete_last_path()

        start_time = time.time()
        best_path, best_cost = ant_colony(
            self.graph, 
            n_ants=self.n_ants,
            n_iterations=self.n_iterations,
            alpha=self.alpha,
            beta=self.beta,
            evaporation_rate=self.evaporation_rate,
            q=self.q,
            use_templates=self.modification_var.get()
        )
        end_time = time.time()
        calculation_time = end_time - start_time

        if best_path:
            self.best_path = best_path
            self.best_cost = best_cost

            self.path_text.config(state=tk.NORMAL)
            self.path_text.delete("1.0", tk.END)
            self.path_text.insert(tk.END, " -> ".join(best_path))
            self.path_text.config(state=tk.DISABLED)

            # Изменяем форматирование вывода длины пути
            if best_cost.is_integer():
                self.length_label.config(text=str(int(best_cost)))
            else:
                self.length_label.config(text=str(best_cost))
            
            self.time_label.config(text=f"{calculation_time:.4f} секунд")

            self.draw_graph_with_path()
        else:
            messagebox.showerror("Ошибка", "Решение не найдено.")

    def calculate_total_cost(self, graph, path):
        """Calculates the total cost of a given path."""
        total_cost = 0
        for i in range(len(path) - 1):
            try:
                total_cost += graph[path[i]][path[i + 1]]['weight']
            except KeyError:
                return float('inf')
        return total_cost

    def run_all_nearest_neighbor(self):
        """(Мод) Перебирает все вершины как стартовые и находит лучший путь."""
        start_time = time.time()

        graph_copy = self.graph.copy()
        edges_copy = self.edges.copy()
        edge_objects_copy = self.edge_objects.copy()
        edge_text_objects_copy = self.edge_text_objects.copy()

        best_path = None
        min_cost = float('inf')
        best_start_node = None

        if self.start_vertex is None and not self.modification_var.get():
            messagebox.showerror("Ошибка", "Пожалуйста, установите начальную вершину.")
            return

        if self.modification_var.get():
            start_nodes = self.vertices.keys()
        else:
            start_nodes = [self.start_vertex]

        for start_node in start_nodes:
            path, total_cost = nearest_neighbor(graph_copy, start_node)

            if path and total_cost < min_cost:
                min_cost = total_cost
                best_path = path
                best_start_node = start_node

        end_time = time.time()
        calculation_time = end_time - start_time

        self.graph = graph_copy
        self.edges = edges_copy
        self.edge_objects = edge_objects_copy
        self.edge_text_objects = edge_text_objects_copy
        self.draw_graph()
        self.update_distance_matrix()

        if best_path:
            self.best_path = best_path
            self.best_cost = min_cost

            self.path_text.config(state=tk.NORMAL)
            self.path_text.delete("1.0", tk.END)
            self.path_text.insert(tk.END, " -> ".join(best_path))
            self.path_text.config(state=tk.DISABLED)

            self.length_label.config(text=str(min_cost))
            self.time_label.config(text=f"{calculation_time:.4f} секунд") # Форматированный вывод

            self.draw_graph_with_path()
        else:
            messagebox.showerror("Ошибка", "Некорректный граф.")

    def draw_graph_with_path(self):
        """Отображает граф с найденным путем."""
        if self.graph is None or self.best_path is None:
            return

        self.path_objects = []
        for i in range(len(self.best_path) - 1):
            v1, v2 = self.best_path[i], self.best_path[i + 1]
            x1, y1 = self.vertices[v1]["x"], self.vertices[v1]["y"]
            x2, y2 = self.vertices[v2]["x"], self.vertices[v2]["y"]

            line_id = self.canvas.create_line(x1, y1, x2, y2, fill="green", width=3)
            self.path_objects.append(line_id)

            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            angle = math.atan2(y2 - y1, x2 - x1)
            arrow_id = self.draw_arrow(mid_x, mid_y, angle, "green")
            self.path_objects.append(arrow_id)

    def delete_last_edge(self):
        """Удаляет последнее добавленное ребро."""
        if self.edge_history:
            v1, v2 = self.edge_history.pop()

            if (v1, v2) in self.edge_objects:
                if self.edge_objects[(v1, v2)]:
                  edge_id = self.edge_objects[(v1, v2)].pop()
                  self.canvas.delete(edge_id)
                if not self.edge_objects[(v1, v2)]:
                  del self.edge_objects[(v1, v2)]
            if (v1, v2) in self.edge_text_objects:
                for text_id in self.edge_text_objects[(v1,v2)]:
                    self.canvas.delete(text_id)
                del self.edge_text_objects[(v1,v2)]
            if (v1, v2) in self.edges:
                del self.edges[(v1, v2)]
            if self.graph.has_edge(v1, v2):
               self.graph.remove_edge(v1,v2)

            self.update_distance_matrix()
            self.draw_graph()
            if not self.edge_history:
                self.last_edge = None

    def delete_last_path(self):
        """Удаляет с холста последний нарисованный путь и восстанавливает матрицу."""
        for obj_id in self.path_objects:
            self.canvas.delete(obj_id)
        self.path_objects = []

        self.best_path = None
        self.best_cost = None

        self.path_text.config(state=tk.NORMAL)
        self.path_text.delete("1.0", tk.END)
        self.path_text.config(state=tk.DISABLED)
        self.length_label.config(text="")
        self.time_label.config(text="")
        self.draw_graph()

    def clear_temporary_edge(self):
        self.selected_vertices = []

    def load_matrix_from_table(self):
        """Загружает матрицу расстояний из таблицы CSV."""
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            try:
                df = pd.read_csv(filename, index_col=0)
                vertices = list(df.index)
                distances = df.to_dict()
                self.load_graph_from_matrix(vertices, distances)

            except FileNotFoundError:
                messagebox.showerror("Ошибка", "Файл не найден.")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить матрицу из таблицы. Проверьте формат. Ошибка: {e}")

    def load_graph_from_matrix(self, vertices, distances):
        """Загружает граф из матрицы расстояний, игнорируя значения <= 0 как отсутствие пути."""
        self.clear_graph()

        for vertex in vertices:
            x = random.randint(50, self.canvas_width - 50)
            y = random.randint(50, self.canvas_height - 50)
            self.add_vertex_from_matrix(vertex, x, y)

        for v1 in vertices:
            for v2 in vertices:
                if v1 != v2:
                    weight = distances[v2][v1]  # Access the distance correctly

                    # Check if the weight is zero or negative, indicating no path
                    if weight <= 0:
                        continue

                    # Also check if it is not NaN if you are getting the value from a pandas dataframe
                    if not isinstance(weight, str) and not (isinstance(weight, float) and math.isnan(weight)):
                        if (v1, v2) not in self.edges:
                            self.edges[(v1, v2)] = []
                            self.edge_objects[(v1,v2)] = []
                            self.edge_text_objects[(v1, v2)] = []
                        self.graph.add_edge(v1, v2, weight=weight)
                        self.edges[(v1, v2)].append(weight)
                        edge_id, text_id = self.draw_edge(v1, v2, weight, len(self.edges[(v1, v2)])-1)
                        self.edge_objects[(v1,v2)].append(edge_id)
                        self.edge_text_objects[(v1, v2)].append(text_id)
                        edge = (v1, v2)
                        self.edge_history.append(edge)

        self.update_distance_matrix()
        self.draw_graph()

    def add_vertex_from_matrix(self, vertex_name, x, y):
        """Добавляет вершину из матрицы."""
        x1, y1 = x - self.vertex_radius, y - self.vertex_radius
        x2, y2 = x + self.vertex_radius, y + self.vertex_radius
        vertex_id = self.canvas.create_oval(x1, y1, x2, y2, fill=self.vertex_color, outline="black")
        text_id = self.canvas.create_text(x, y, text=vertex_name, fill="black", font=('Arial', 10, 'bold'))

        self.vertices[vertex_name] = {"x": x, "y": y, "oval_id": vertex_id, "text_id": text_id}
        self.graph.add_node(vertex_name)

    def update_edge_weight(self):
        """Обновляет вес выбранного ребра."""
        edge_str = self.edge_entry.get().upper()
        try:
            v1, v2 = edge_str[0], edge_str[1]
            if v1 not in self.vertices or v2 not in self.vertices:
                raise ValueError("Указанная вершина не существует.")

            if (v1, v2) not in self.edges:
                raise ValueError("Указанное ребро не существует.")

            try:
                new_weight = int(self.edge_weight_entry.get())
                if new_weight <= 0:
                    messagebox.showerror("Ошибка", "Вес должен быть положительным числом.")
                    return
            except ValueError:
                messagebox.showerror("Ошибка", "Пожалуйста, введите целое число для веса.")
                return

            self.graph[v1][v2]['weight'] = new_weight
            self.edges[(v1, v2)][0] = new_weight
            self.update_distance_matrix()

            if (v1, v2) in self.edge_text_objects:
                for text_id in self.edge_text_objects[(v1, v2)]:
                    self.canvas.delete(text_id)

            edge_id, text_id = self.draw_edge(v1, v2, new_weight, 0)

            self.edge_objects[(v1, v2)] = [edge_id]
            self.edge_text_objects[(v1, v2)] = [text_id]

            self.draw_graph()
            messagebox.showinfo("Информация", "Вес ребра успешно изменен.")

        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))