import numpy as np
import heapq

def read_adjacency_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    adjacency_matrix = []
    for line in lines:
        row = [int(val) for val in line.strip().split(':')]
        adjacency_matrix.append(row)

    return np.array(adjacency_matrix)

def read_bandwidth_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    bandwidth_matrix = []
    for line in lines:
        if line.strip():  
            row = [float(val) for val in line.strip().split(':')]
            bandwidth_matrix.append(row)

    return np.array(bandwidth_matrix)


def dijkstra(adjacency_matrix, source, destination):
    n = len(adjacency_matrix)
    visited = [False] * n
    distance = [float('inf')] * n
    distance[source] = 0
    previous = [None] * n

    priority_queue = [(0, source)]

    while priority_queue:
        current_dist, current_node = heapq.heappop(priority_queue)

        if visited[current_node]:
            continue

        visited[current_node] = True

        for neighbor, weight in enumerate(adjacency_matrix[current_node]):
            if not visited[neighbor] and weight > 0:
                new_distance = distance[current_node] + weight
                if new_distance < distance[neighbor]:
                    distance[neighbor] = new_distance
                    previous[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_distance, neighbor))

    path = []
    current = destination
    while current is not None:
        path.insert(0, current)
        current = previous[current]

    return path, distance[destination]

def bellman_ford(adjacency_matrix, source, destination):
    n = len(adjacency_matrix)
    distance = [float('inf')] * n
    distance[source] = 0
    previous = [None] * n

    for _ in range(n - 1):
        for u in range(n):
            for v, weight in enumerate(adjacency_matrix[u]):
                if weight > 0 and distance[u] + weight < distance[v]:
                    distance[v] = distance[u] + weight
                    previous[v] = u

    
    for u in range(n):
        for v, weight in enumerate(adjacency_matrix[u]):
            if weight > 0 and distance[u] + weight < distance[v]:
                raise ValueError("Graph contains a negative cycle")

    path = []
    current = destination
    while current is not None:
        path.insert(0, current)
        current = previous[current]

    return path, distance[destination]

def astar(adjacency_matrix, source, destination, heuristic):
    n = len(adjacency_matrix)
    visited = [False] * n
    distance = [float('inf')] * n
    distance[source] = 0
    previous = [None] * n

    priority_queue = [(heuristic(source, destination), 0, source)]

    while priority_queue:
        _, current_dist, current_node = heapq.heappop(priority_queue)

        if visited[current_node]:
            continue

        visited[current_node] = True

        for neighbor, weight in enumerate(adjacency_matrix[current_node]):
            if not visited[neighbor] and weight > 0:
                new_distance = distance[current_node] + weight
                if new_distance < distance[neighbor]:
                    distance[neighbor] = new_distance
                    previous[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_distance + heuristic(neighbor, destination), new_distance, neighbor))

    path = []
    current = destination
    while current is not None:
        path.insert(0, current)
        current = previous[current]

    return path, distance[destination]

def heuristic(node, destination):
    
    return np.linalg.norm(np.array(node) - np.array(destination))

def main():
    usnet_adjacency_path = "USNET_AjdMatrix.txt"
    usnet_bandwidth_path = "USNET.txt"

    source_node = 0
    destination_node = 5

   
    adjacency_matrix = read_adjacency_matrix(usnet_adjacency_path)
    bandwidth_matrix = read_bandwidth_matrix(usnet_bandwidth_path)

    shortest_path_dijkstra, distance_dijkstra = dijkstra(adjacency_matrix, source_node, destination_node)
    print("Dijkstra's Shortest Path:", shortest_path_dijkstra)
    print("Dijkstra's Full Path:", " -> ".join(map(str, shortest_path_dijkstra)))
    print("Dijkstra's Distance:", distance_dijkstra)

    shortest_path_bellman_ford, distance_bellman_ford = bellman_ford(adjacency_matrix, source_node, destination_node)
    print("Bellman-Ford Shortest Path:", shortest_path_bellman_ford)
    print("Bellman-Ford Full Path:", " -> ".join(map(str, shortest_path_bellman_ford)))
    print("Bellman-Ford Distance:", distance_bellman_ford)

    shortest_path_astar, distance_astar = astar(adjacency_matrix, source_node, destination_node, heuristic)
    print("A* Shortest Path:", shortest_path_astar)
    print("A* Full Path:", " -> ".join(map(str, shortest_path_astar)))
    print("A* Distance:", distance_astar)

if __name__ == "__main__":
    main()