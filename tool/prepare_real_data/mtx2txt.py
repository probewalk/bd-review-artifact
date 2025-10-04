import os

input_file = "./data/real-large-graph/asia_osm/asia_osm/asia_osm.mtx"
output_file = "./data/real-large-graph/asia_osm/asia_osm/asia_osm.txt"

num_nodes = 0
edges = []

with open(input_file, "r") as fin:
    for line in fin:
        if line.startswith("%"):
            continue
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        u, v = int(parts[0]) - 1, int(parts[1]) - 1  # convert to 0-based
        edges.append((u, v))
        num_nodes = max(num_nodes, u, v)

num_nodes += 1  # number of nodes = max index + 1

with open(output_file, "w") as fout:
    fout.write(f"{num_nodes}\n")
    for u, v in edges:
        fout.write(f"{u} {v}\n")

print(f"Done: nodes {num_nodes}, edges {len(edges)}, output file {output_file}")
