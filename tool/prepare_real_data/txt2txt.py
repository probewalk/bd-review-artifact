#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remap the SNAP Friendster edge list into consecutive IDs 0..N-1, 
and output in the specified format:
The 1st line is the number of nodes; 
each following line is one edge with u < v (normalized for undirected graph).
Note: this script does NOT deduplicate edges. 
If the original file contains the same undirected edge multiple times, 
the normalized output will also contain duplicates 
(typically SNAP contains two lines: u v and v u).
If deduplication is needed, you can post-process the output; 
I can provide another script for that.
"""

import os, sys, time
from array import array

# ---------- Hard-coded paths ----------
IN_PATH  = "./data/real-large-graph/twitter-2010/twitter-2010.txt"
OUT_PATH = "./data/real-large-graph/twitter-2010/twitter-2010-new.txt"
PRINT_EVERY = 10_000_000  # Report progress every N lines
BUFFER = 16 * 1024 * 1024 # 16MB line buffer

def pass0_check():
    if not os.path.exists(IN_PATH):
        print(f"❌ Input file not found: {IN_PATH}", file=sys.stderr); sys.exit(1)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

def pass1_scan_max_and_lines():
    """First pass: scan the file to get max_id and total number of lines, 
    in order to allocate mapping array and display progress later."""
    print("[Pass1] Scanning to get max_id and total lines ...")
    st = time.time()
    max_id = -1
    total = 0
    with open(IN_PATH, "r", buffering=BUFFER) as fin:
        for line in fin:
            parts = line.strip().split(" ")
            if len(parts) != 2: 
                continue
            try:
                u = int(parts[0]); v = int(parts[1])
            except ValueError:
                print("ValueError: invalid line")
                continue
            if u > max_id: max_id = u
            if v > max_id: max_id = v
            total += 1
            if total % PRINT_EVERY == 0:
                elapsed = time.time() - st
                print(f"[Pass1] {total//1_000_000}M lines, max_id={max_id}, {elapsed:.1f}s", flush=True)
    if max_id < 0:
        print("❌ Input is empty or invalid format", file=sys.stderr); sys.exit(1)
    print(f"[Pass1] Done: total lines={total:,}, max_id={max_id}, elapsed {time.time()-st:.1f}s")
    return max_id, total

def pass2_build_mapping(max_id, total_lines):
    """
    Second pass: build old_id -> new_id mapping to ensure node IDs are consecutive 0..N-1.
    Use 4-byte int array, initialized with -1. Memory ~ (max_id+1)*4 bytes.
    """
    print("[Pass2] Allocating mapping array and building ID remapping (old -> new) ...")
    st = time.time()
    try:
        mapping = array('i', [-1]) * (max_id + 1)  # sentinel -1
    except MemoryError:
        print("❌ Not enough memory: failed to allocate mapping array", file=sys.stderr); sys.exit(1)

    next_id = 0
    seen_lines = 0
    with open(IN_PATH, "r", buffering=BUFFER) as fin:
        for line in fin:
            parts = line.strip().split(" ") 
            if len(parts) != 2:
                continue
            try:
                u = int(parts[0]); v = int(parts[1])
            except ValueError:
                continue

            if mapping[u] == -1:
                mapping[u] = next_id
                next_id += 1
            if mapping[v] == -1:
                mapping[v] = next_id
                next_id += 1

            seen_lines += 1
            if seen_lines % PRINT_EVERY == 0:
                elapsed = time.time() - st
                print(f"[Pass2] {seen_lines//1_000_000}M/{total_lines//1_000_000}M lines, unique_nodes={next_id:,}, {elapsed:.1f}s", flush=True)

    print(f"[Pass2] Done: unique_nodes={next_id:,} (expected ~65,608,366), elapsed {time.time()-st:.1f}s")
    return mapping, next_id

def pass3_write_output(mapping, num_nodes, total_lines):
    """
    Third pass: write the final output file.
    - The first line is num_nodes
    - Each following line is an edge remapped to [0..num_nodes-1], normalized so that u < v.
    - Self-loops are discarded.
    Note: no deduplication. If needed, you can post-process with external deduplication.
    """
    print("[Pass3] Writing normalized edge file ...")
    st = time.time()
    written = 0
    processed = 0

    # Write header (number of nodes)
    with open(OUT_PATH, "w", buffering=BUFFER) as fout:
        fout.write(f"{num_nodes}\n")

    # Write edges
    with open(IN_PATH, "r", buffering=BUFFER) as fin, \
         open(OUT_PATH, "a", buffering=BUFFER) as fout:
        for line in fin:
            parts = line.strip().split(" ") 
            if len(parts) != 2:
                continue
            try:
                u0 = int(parts[0]); v0 = int(parts[1])
            except ValueError:
                continue
            u = mapping[u0]; v = mapping[v0]
            # Safety check
            if u < 0 or v < 0:
                # Unmapped (should not happen)
                continue
            if u == v:
                # Discard self-loops
                processed += 1
                if processed % PRINT_EVERY == 0:
                    elapsed = time.time() - st
                    pct = processed / max(1, total_lines) * 100.0
                    print(f"[Pass3] {processed//1_000_000}M/{total_lines//1_000_000}M lines, wrote {written//1_000_000}M edges, {pct:.1f}%, {elapsed:.1f}s", flush=True)
                continue
            if u > v:
                u, v = v, u
            fout.write(f"{u} {v}\n")
            written += 1
            processed += 1

            if processed % PRINT_EVERY == 0:
                elapsed = time.time() - st
                pct = processed / max(1, total_lines) * 100.0
                print(f"[Pass3] {processed//1_000_000}M/{total_lines//1_000_000}M lines, wrote {written//1_000_000}M edges, {pct:.1f}%, {elapsed:.1f}s", flush=True)

    print(f"[Pass3] Done: edges written={written:,}, elapsed {time.time()-st:.1f}s")
    print(f"[DONE] Output file: {OUT_PATH}\n        Num. nodes: {num_nodes:,}\n        Num. edges: {written:,}")
    return written

def main():
    pass0_check()
    max_id, total_lines = pass1_scan_max_and_lines()
    mapping, num_nodes = pass2_build_mapping(max_id, total_lines)
    pass3_write_output(mapping, num_nodes, total_lines)

if __name__ == "__main__":
    main()
