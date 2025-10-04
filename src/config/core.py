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
    random_pair_num:int=10
    pairs_num:int=5


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
    """The first line of the graph file is the number of nodes"""
    with open(graph_path, "r") as f:
        first_line = f.readline().strip()
    return int(first_line)


# ----------------- Core executor class -----------------



class ExperimentExecutor:
    def __init__(self, methods_registry: Dict[str, Any]):
        self.methods_registry = methods_registry

    def run_experiments(self, cfg: ExperimentConf) -> None:
        # Read info.json to get pairs for each graph
        info_json_path = "./data/real-data/info.json"
        if os.path.exists(info_json_path):
            with open(info_json_path, "r") as f:
                graph_info = json.load(f)
        else:
            graph_info = {}

        base_root = Path("./data/paper/test")
        base_root.mkdir(parents=True, exist_ok=True)

        for ds in cfg.datasets:
            # —— Each run processes only one graph (you already guaranteed this), dir: .../test/<dataset_name> ——
            ds_dirname = ds.name if ds.name else Path(ds.graph).stem
            ds_dir = base_root / ds_dirname
            ds_dir.mkdir(parents=True, exist_ok=True)

            n = get_num_nodes(ds.graph)

            base_name = os.path.basename(ds.graph)
            if base_name in graph_info and "pairs" in graph_info[base_name]:
                pairs = graph_info[base_name]["pairs"]
                pairs_source = "info_json"
            else:
                pairs = [(2, 5)]
                pairs_source = "default"



            for mg in cfg.methods:
                method_key = mg.method                       # e.g. "methoda"
                method = self.methods_registry[method_key]   # e.g. METHODA()
                params = dict(mg.params)                     # { "L": 20,"R": 200,"K": 1000,"m":50 ,"re":0.05}
                # re=params.get("re")
                
                # method_json_path = ds_dir / f"{method_key}_{re}.json"


                log = {
                    "dataset": ds_dirname,
                    "n": n,
                    "method_name": getattr(method, "name", method_key),
                    "seed": cfg.seed,
                    "pairs_source": pairs_source,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "params": params, 
                    "avg_time": 0.0,
                    "avg_re": 0.0,                    
                    "results": []   
                }

                    
                pairs_num=cfg.pairs_num
                params_str = ", ".join(f"{k}={v}" for k, v in params.items())
                print(f"[run] {ds_dirname}/{method_key} with params:{params_str}  total pair:{pairs_num}")
                # Run each pair + record
                
                
                idx=0
                time_sum=0.0
                re_sum=0.0
                for (s, e) in pairs[:pairs_num] :
                    record_for_id = {
                        "dataset": ds_dirname,
                        "start": s,
                        "end": e,
                        "method": getattr(method, "name", method_key),
                        "seed": cfg.seed,
                        "params": params,
                    }


                    params_str = ", ".join(f"{k}={v}" for k, v in params.items())
                    print(f">>>>>>>>>>  {ds_dirname}/{method_key} {idx}/{pairs_num} pair ({s},{e})   ")

 
                    res = method.run(
                        ds.graph,
                        start=s, end=e,
                        seed=cfg.seed,
                        verbose=False,
                        **params,
                    )

                    pair_result={
                        "idx":idx,
                        "pair":pairs[idx],
                        "bst_2": res.get("bst_2"),
                        "elapsed_ms": res.get("timings_ms", {}).get("key_algo", 0.0) 
                    }

                    ds_txt=ds_dirname+".txt"
                    gold_bst=graph_info[ds_txt]["gold_ans"][idx]
                    bst=res.get('bst_2')
                    re=abs(bst-gold_bst)/gold_bst 
                    
                    log["results"].append(pair_result)
                    
                    print(f">>>>>>>>>>   bst_2={bst:.20f},gold={gold_bst},re={re} elapsed={res.get('timings_ms', {}).get('key_algo', 0.0):.1f}ms")

                    time_sum+=res.get('timings_ms', {}).get('key_algo', 0.0)
                    re_sum+=re
                    idx+=1
                    
                    
                avg_time=time_sum/pairs_num
                avg_re=re_sum/pairs_num
                log["avg_time"]=avg_time
                log["avg_re"]=avg_re
                method_json_path = ds_dir / f"{method_key}_{avg_re:.6f}_{avg_time:.1f}.json"
                with open(method_json_path, "w", encoding="utf-8") as f:
                    json.dump(log, f, ensure_ascii=False, indent=2)
