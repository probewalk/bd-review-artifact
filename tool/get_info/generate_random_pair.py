#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, random

INFO_PATH = "./data/real-data/info.json"

def main():
    # Read original info.json
    with open(INFO_PATH, "r") as f:
        info = json.load(f)

    random.seed(0)  # fix random seed for reproducibility

    for fname, data in info.items():
        # if fname!="twitter-2010.txt":
        #     continue
        n = data.get("n", 0)
        pairs = data.get("pairs", [])
        if n > 1:
            for _ in range(100):
                u, v = random.sample(range(n), 2)
                pairs.append([u, v])
        data["pairs"] = pairs  # append pairs

    # Write back to JSON
    with open(INFO_PATH, "w") as f:
        json.dump(info, f, ensure_ascii=False, separators=(",", ":"))

    print(f"[DONE] Updated {INFO_PATH}, added 100 random node pairs to each graph.")

if __name__ == "__main__":
    main()
