import matplotlib.pyplot as plt
import networkx as nx
from pprint import pprint
from random import choice
from math import sqrt
from operator import itemgetter


class Maze:

    def __init__(self, labels):
        self.graph = labels
        self.current_room = "A"
        self.end_room = "N"
        self.all_cost = 0

    def __str__(self):
        return "Trenutna soba: " + self.current_room + "\n" + "Cilj: " + self.end_room + "\n" + "Uk. cijena: " + str(self.all_cost)

    def show_graph(self):
        pprint(self.graph)

    def neighbours(self, node='A'):
        neighbours = []
        for room in self.graph:
            if node == room[0]:
                neighbours.append(room[1])
            elif node == room[1]:
                neighbours.append(room[0])

        return neighbours

    def room_cost(self, room):
        move = (self.current_room, room)

        if self.current_room == room:
            return 0
        if move not in self.graph:
            cost = self.graph[(move[1], move[0])]
        else:
            cost = self.graph[move]

        return cost


def read_coordinates(filename):
    coordinates_dict = {}
    with open(filename) as f:
        next(f)
        for line in f:
            if "edges" in line:
                break
            room = line.split()
            # Dodaj koordinate u dict u obliku soba: (x, y)
            coordinates_dict[room[1]] = (int(room[2]), int(room[3]))

    return coordinates_dict


def heuristic_coords_dist(coords_dict, room, end_room):
    (x1, y1) = coords_dict[room]
    (x2, y2) = coords_dict[end_room]

    # Euklidova udaljenost između 2 točke
    distance = round(sqrt(pow(x1-y1, 2) + pow(x2-y2, 2)), 4)

    return distance


def AStar(game, coords, conn_sort=False):
    rooms_visited = {}
    heap = [[game.current_room, heuristic_coords_dist(
        coords, game.current_room, game.end_room), 0, None, len(game.neighbours())]]

    while game.current_room != game.end_room:
        heap.sort(key=itemgetter(1))
        if conn_sort:
            # Preferenca čvorova s više veza pa onda po udaljenosti
            heap.sort(key=itemgetter(4), reverse=True)
        print("Heap:", heap)
        node, heur, cost, parent, n_conn = heap.pop(0)
        while node in rooms_visited:
            node, heur, cost, parent = heap.pop(0)
        rooms_visited[node] = (parent, cost)
        game.current_room = node
        game.all_cost = cost
        print("currently visiting:", node)
        neighbours = game.neighbours(node)
        print("Rooms visited:", rooms_visited)
        print("Neighbours:", neighbours)
        print()
        for neighbour in neighbours:
            coords_distance = heuristic_coords_dist(coords, neighbour, game.end_room)
            n_connected = len(game.neighbours(neighbour))
            in_heap = False
            for room in heap:
                if neighbour == room[0]:
                    in_heap = True
                    same_r = room
            if neighbour in rooms_visited:
                # Ako postoji jeftiniji put do sobe (gleda se samo cijena, ne heuristika), zamijeni ga
                if (game.all_cost + game.room_cost(neighbour)) < rooms_visited[neighbour][1]:
                    heap.append([neighbour, coords_distance, game.all_cost +
                                 game.room_cost(neighbour), node, n_connected])
                else:
                    continue
            elif in_heap:
                index = heap.index(same_r)
                room = heap[index]
                if (game.all_cost + game.room_cost(neighbour)) < room[2]:
                    heap[index] = [room[0], coords_distance, game.all_cost +
                                   game.room_cost(neighbour), node, n_connected]
            else:
                heap.append([neighbour, coords_distance, game.all_cost + game.room_cost(neighbour), node, n_connected])

    return rooms_visited


def print_path(rooms_visited, end_room):
    path = []
    parent = end_room
    path_cost = rooms_visited[parent][1]
    while parent != None:
        path.append(parent)
        p = rooms_visited[parent]
        parent = p[0]

    path.reverse()
    print("Path:", path)
    print("Path cost:", path_cost)


def main():
    graph = nx.read_pajek("rooms.net")
    graph = nx.Graph(graph)
    labels = nx.get_edge_attributes(graph, 'weight')
    for label in labels:
        labels[label] = int(labels[label])

    game = Maze(labels)
    coords = read_coordinates("rooms.net")
    visited = AStar(game, coords, True)
    print_path(visited, game.end_room)


if __name__ == "__main__":
    main()


'''
graph = nx.read_pajek("rooms.net")
graph = nx.Graph(graph)
layout = nx.spring_layout(graph)
nx.draw(graph, pos=layout, with_labels=True)
labels = nx.get_edge_attributes(graph, 'weight')
for label in labels:
    labels[label] = int(labels[label])
nx.draw_networkx_edge_labels(graph, pos=layout, edge_labels=labels)
plt.savefig("rooms.png")
plt.show()
'''
