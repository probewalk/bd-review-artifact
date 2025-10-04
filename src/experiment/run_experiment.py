from pathlib import Path
from src.config.loader import load_config
from src.config.core import ExperimentExecutor
from src.config.gold import ExperimentExecutor_gold
# from src.scripts.stw import STW
# from src.scripts.swf import SWF
# from src.scripts.push import PUSH

from src.scripts.probewalk import probewalk

def main():
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.json"
    cfg = load_config(cfg_path)

    methods_registry = {
        # "stw":STW(),
        # "swf":SWF(),
        # "push":PUSH(),
        "probewalk":probewalk()
    }

    executor = ExperimentExecutor(methods_registry)
    # executor = ExperimentExecutor_gold(methods_registry)
    executor.run_experiments(cfg)

if __name__ == "__main__":
    main()
  