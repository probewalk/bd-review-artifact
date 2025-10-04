# -*- coding: utf-8 -*-
"""
core.py - Core tools of experiment framework (general scheduler)
=================================================
Functions:
1. Unique run_id generation (for checkpoint continuation and deduplication of results)
2. JSONL / CSV / single JSON file result writing
3. Load run_ids of completed experiments
4. A simple experiment executor (sequential execution, avoiding nested parallelism)

Typical usage:
------------
from config.core import ExperimentExecutor, DatasetConf, MethodGrid, ExperimentConf

datasets = [
    DatasetConf(name="g5", graph="/data3/.../graph5.txt", start=2, end=5),
]

methods = [
    MethodGrid(method="bpa", params={"L": 6, "nr": 6, "workers": 32}),
]

cfg = ExperimentConf(
    out_dir="exp_logs/bpa_demo",
    datasets=datasets,
    methods=methods,
    seed=42,          # ✅ single seed
    env_threads=32,
)

executor = ExperimentExecutor(methods_registry={"bpa": BPA()})
executor.run_experiments(cfg)
"""

import os, json, csv, time, hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List
import random


# ----------------- Data class definitions -----------------
@dataclass
class DatasetConf:
    name: str
    graph: str
    start: int
    end: int

@dataclass
class MethodGrid:
    method: str                 # method name, corresponding to the key in methods_registry
    params: Dict[str, Any]      # parameter dictionary (fixed values)

@dataclass
class ExperimentConf:
    out_dir: str
    datasets: List[DatasetConf]
    methods: List[MethodGrid]
    seed: int = 42              # ✅ single seed
    env_threads: int | None = None
    random_pair_num:int=10
    is_random:int=0


# ----------------- Utility functions -----------------
def make_run_id(payload: Dict[str, Any]) -> str:
    """Generate unique ID based on experiment config"""
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def append_jsonl(path: Path, obj: Dict[str, Any]):
    """Append a JSONL record"""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def append_csv(path: Path, obj: Dict[str, Any], header_written: set):
    """Append a CSV record"""
    flat = dict(obj)
    t = flat.pop("timings_ms", {})
    for k, v in t.items(): 
        flat[f"time_{k}"] = v
    fieldnames = list(flat.keys())
    write_header = path not in header_written
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
            header_written.add(path)
        w.writerow(flat)

def load_done_ids(jsonl_path: Path) -> set:
    """Read completed run_id set"""
    if not jsonl_path.exists(): 
        return set()
    done = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if "run_id" in rec: 
                    done.add(rec["run_id"])
            except Exception:
                continue
    return done


def get_num_nodes(graph_path: str | Path) -> int:
    """First line of the graph file is the number of nodes"""
    with open(graph_path, "r") as f:
        first_line = f.readline().strip()
    return int(first_line)


# ----------------- Core executor class -----------------



class ExperimentExecutor_gold:
    def __init__(self, methods_registry: Dict[str, Any]):
        self.methods_registry = methods_registry

    def run_experiments(self, cfg: ExperimentConf) -> None:
        # Read info.json to get pairs for each graph
        info_json_path = "./data/real-data/info.json"

        with open(info_json_path, "r") as f:
            graph_info = json.load(f)
        
        dataset_base="./data/real-data"
        datasets=["facebook.txt","dblp.txt","youtube.txt","orkut.txt","lj.txt","as-skitter.txt"]
        methods_for_dataset=["push","push","push","push","push","push"]
        # datasets=["youtube.txt","orkut.txt","lj.txt"]
        # methods_for_dataset=["push","push","push"]
        # datasets=["as-skitter.txt"]
        # methods_for_dataset=["push"]
        params_for_dataset={
            "as-skitter.txt":{ "L": 1000 },
            "facebook.txt":{ "L": 1000 },
            "dblp.txt":{ "L": 1000 },        
            "youtube.txt":{ "L": 1000 },
            "orkut.txt":{ "L": 1000 },
            "lj.txt":{ "L": 1000 }
        }
        
        out_root= Path("./data/paper/gold_log")
        out_root.mkdir(parents=True, exist_ok=True)
        
        for ds in datasets:
            print(f"[INFO] Processing graph {ds} ...")
            path_to_graph=os.path.join(dataset_base,ds)
            pairs=graph_info.get(ds, {}).get("pairs", [])
            
            method_key = methods_for_dataset[datasets.index(ds)]
            method = self.methods_registry[method_key]
            method_dir= out_root / f"{ds}_log.json"
            log = {
                "dataset": ds ,              
                "results": []   
            }
            
            # if len(pairs)!=10:
            #     print(f"❌ Graph {ds} pairs count is not 10")
            #     continue
            
            
            results=[0]*10         
            idx=0
            
            L=params_for_dataset[ds]["L"]
            if method_key=="methoda":
                R=params_for_dataset[ds]["R"]
                G=params_for_dataset[ds]["G"]
                m=params_for_dataset[ds]["m"]
            
            for (s,e) in pairs:
                # if idx!=0:
                #     continue
                
                if method_key=="methoda":
                    res = method.run(
                        path_to_graph,
                        start=s,
                        end=e,
                        L=L,                
                        R=R,
                        G=G,
                        m=m,
                        seed= 42,
                        verbose=False,
                        force_reload=False
                    )
                else:
                    res = method.run(
                        path_to_graph,
                        start=s,
                        end=e,
                        L=L,         
                        seed= 42,
                        verbose=False,
                        force_reload=False
                    )
                pair_info={
                    "idx":idx,
                    "pair":pairs[idx],
                    "bst_2": res.get("bst_2"),
                    "elapsed_ms": res.get("timings_ms", {}).get("key_algo", 0.0) 
                }
                log["results"].append(pair_info)
                print(f">>>>>>>>>>  Graph {ds} idx={idx} pair ({s},{e})   bst_2={res.get('bst_2'):.20f}, elapsed={res.get('timings_ms', {}).get('key_algo', 0.0) :.1f}ms  ")

                results[idx]=res.get("bst_2")
                idx+=1

            graph_info[ds]["gold_L"]=L
            graph_info[ds]["gold_ans"]=results
            print(f"[INFO] results: {results}")
            print(f"[INFO] Graph {ds} gold_ans has been updated.")
            
            with open(info_json_path, "w", encoding="utf-8") as f:
                json.dump(graph_info, f, indent=2, ensure_ascii=False)
                
            with open(method_dir, "w", encoding="utf-8") as f:
                json.dump(log, f, ensure_ascii=False, indent=2)
