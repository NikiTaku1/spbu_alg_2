def nearest_neighbor(graph, start_node):
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