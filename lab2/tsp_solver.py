import random
import math

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

def simulated_annealing(graph, start_node, initial_temperature, cooling_rate, num_iterations, use_boltzmann, max_attempts=20):
    """Implements simulated annealing that handles missing edges gracefully."""
    nodes = list(graph.nodes())
    if not nodes:
        return None, 0

    if start_node not in nodes:
        start_node = random.choice(nodes)

    for attempt in range(max_attempts):
        current_solution = nodes.copy()
        random.shuffle(current_solution)
        if start_node in current_solution:
            current_solution.remove(start_node)
        current_solution.insert(0, start_node)

        current_cost = calculate_total_cost(graph, current_solution)
        if current_cost is None:
            continue

        best_solution = current_solution.copy()
        best_cost = current_cost
        temperature = initial_temperature

        for i in range(num_iterations):
            if len(nodes) > 2:
                idx1, idx2 = random.sample(range(1, len(nodes)), 2)
                neighbor_solution = current_solution.copy()
                neighbor_solution[idx1], neighbor_solution[idx2] = neighbor_solution[idx2], neighbor_solution[idx1]
                
                neighbor_cost = calculate_total_cost(graph, neighbor_solution)
                if neighbor_cost is None:
                    continue

                cost_diff = neighbor_cost - current_cost
                
                if cost_diff < 0 or (use_boltzmann and random.random() < math.exp(-cost_diff / temperature)):
                    current_solution = neighbor_solution
                    current_cost = neighbor_cost
                    
                    if current_cost < best_cost:
                        best_solution = current_solution.copy()
                        best_cost = current_cost

            if use_boltzmann:
                temperature = initial_temperature / math.log(i + 2)
            else:
                temperature *= (1 - cooling_rate)

        if best_cost is not None:
            return best_solution + [best_solution[0]], best_cost

    return None, 0

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