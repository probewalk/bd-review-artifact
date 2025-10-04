import sys

path = "./data/real-data/asia_osm.txt"

with open(path, "r") as f:
    first_line = f.readline().strip()
    try:
        num_nodes = int(first_line)
    except ValueError:
        print("❌ The first line is not an integer")
        sys.exit(1)

    print(f"✅ Declared number of nodes: {num_nodes}")

    line_num = 1
    valid = True
    for line in f:
        line_num += 1
        parts = line.strip().split()
        if len(parts) != 2:
            print(f"❌ Line {line_num} does not contain two integers: {line.strip()}")
            valid = False
            break
        try:
            u, v = map(int, parts)
        except ValueError:   
            print(f"❌ Line {line_num} contains non-integer values: {line.strip()}")
            valid = False
            break
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            print(f"⚠️ Line {line_num} contains out-of-range nodes: {u}, {v}")
            valid = False
            break

    if valid:
        print("✅ File format check passed: the first line is the number of nodes, followed by valid edges.")
