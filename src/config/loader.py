# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
from src.config.core import ExperimentConf, DatasetConf, MethodGrid



def load_config(json_path: str | Path) -> ExperimentConf:
    p = Path(json_path)
    data: Dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))

    out_dir     = data["out_dir"]
    seed        = data.get("seed", 42)
    random_pair_num= data.get("random_pair_num", 2)
    pairs_num    = data.get("pairs_num", 5)

    datasets = [
        DatasetConf(
            name=d["name"],
            graph=d["graph"],
            start=None,
            end=None,
        ) for d in data["datasets"]
    ]

    methods = [
        MethodGrid(method=m["method"], params=m.get("params", {}))
        for m in data["methods"]
    ]

    return ExperimentConf(
        out_dir=out_dir,
        datasets=datasets,
        methods=methods,
        seed=seed,
        random_pair_num=random_pair_num,
        pairs_num=pairs_num
    )
