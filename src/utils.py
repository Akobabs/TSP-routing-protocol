import numpy as np
import pandas as pd
import osmnx as ox
import os

def generate_tsp_dataset(num_nodes=10, grid_size=100, output_file="data/tsp_data.csv"):
    np.random.seed(42)
    nodes = np.random.rand(num_nodes, 2) * grid_size
    data = {
        "node_id": range(num_nodes),
        "x": nodes[:, 0],
        "y": nodes[:, 1]
    }
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")

    # Generate distance matrix
    distances = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distances[i, j] = np.sqrt(
                    (nodes[i, 0] - nodes[j, 0])**2 + (nodes[i, 1] - nodes[j, 1])**2
                )
    np.savetxt("data/distance_matrix.csv", distances, delimiter=",")
    print("Distance matrix saved to data/distance_matrix.csv")

def simulate_traffic(distances, congestion_prob=0.2, congestion_factor=2.0):
    traffic_matrix = distances.copy()
    num_nodes = distances.shape[0]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and np.random.rand() < congestion_prob:
                traffic_matrix[i, j] *= congestion_factor
    return traffic_matrix

def download_osm_network(city="downtown, Los Angeles, USA", output_file="data/osm_network.graphml"):
    G = ox.graph_from_place(city, network_type="drive")
    ox.save_graphml(G, output_file)
    print(f"OSM network saved to {output_file}")

if __name__ == "__main__":
    download_osm_network()
    os.makedirs("data", exist_ok=True)
    generate_tsp_dataset()