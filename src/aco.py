import numpy as np
import pandas as pd

class ACO:
    def __init__(self, distances, num_ants, num_iterations, alpha=1.0, beta=2.0, rho=0.1, q=100):
        self.distances = distances
        self.num_nodes = distances.shape[0]
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Heuristic importance
        self.rho = rho      # Pheromone evaporation rate
        self.q = q          # Pheromone deposit constant
        self.pheromones = np.ones((self.num_nodes, self.num_nodes)) / self.num_nodes
        self.best_tour = None
        self.best_distance = float("inf")

    def run(self):
        for iteration in range(self.num_iterations):
            tours = []
            tour_distances = []
            for ant in range(self.num_ants):
                tour = self.construct_tour()
                distance = self.calculate_distance(tour)
                tours.append(tour)
                tour_distances.append(distance)
                if distance < self.best_distance:
                    self.best_tour = tour
                    self.best_distance = distance
            self.update_pheromones(tours, tour_distances)
        return self.best_tour, self.best_distance

    def construct_tour(self):
        tour = [np.random.randint(self.num_nodes)]
        unvisited = set(range(self.num_nodes)) - {tour[0]}
        while unvisited:
            next_node = self.select_next_node(tour[-1], unvisited)
            tour.append(next_node)
            unvisited.remove(next_node)
        return tour

    def select_next_node(self, current, unvisited):
        probabilities = []
        total = 0
        for node in unvisited:
            pheromone = self.pheromones[current, node] ** self.alpha
            heuristic = (1.0 / self.distances[current, node]) ** self.beta
            prob = pheromone * heuristic
            probabilities.append(prob)
            total += prob
        probabilities = [p / total for p in probabilities]
        return np.random.choice(list(unvisited), p=probabilities)

    def calculate_distance(self, tour):
        distance = 0
        for i in range(len(tour)):
            distance += self.distances[tour[i], tour[(i + 1) % len(tour)]]
        return distance

    def update_pheromones(self, tours, tour_distances):
        self.pheromones *= (1 - self.rho)  # Evaporation
        for tour, distance in zip(tours, tour_distances):
            for i in range(len(tour)):
                u = tour[i]
                v = tour[(i + 1) % len(tour)]
                self.pheromones[u, v] += self.q / distance
                self.pheromones[v, u] += self.q / distance

def main():
    # Load distance matrix
    distances = np.loadtxt("data/distance_matrix.csv", delimiter=",")
    # Simulate traffic
    traffic_distances = simulate_traffic(distances)
    # Run ACO
    aco = ACO(traffic_distances, num_ants=10, num_iterations=100)
    best_tour, best_distance = aco.run()
    print(f"Best Tour: {best_tour}")
    print(f"Best Distance: {best_distance}")

if __name__ == "__main__":
    main()