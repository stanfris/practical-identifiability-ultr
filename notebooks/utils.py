import pandas as pd
import re
from pathlib import Path


def load(experiment, file, base_dir=Path("results")):
    dfs = []

    experiment_dir = base_dir / experiment
    paths = [p / file for p in experiment_dir.iterdir() if p.is_dir() and (p / file).exists()]

    for run_dir in experiment_dir.iterdir():
        path = (run_dir / file)
        
        if not path.exists():
            continue

        df = pd.read_csv(path)

        for kv in str(run_dir).split(","):
            key, val = kv.split("=")
            df[key] = parse_value(val)

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def parse_value(raw: str):
    low = raw.lower()
    
    if low == 'true':
        return True
    elif low == 'false':
        return False
    elif re.fullmatch(r'-?\d+', raw):
        return int(raw)
    elif re.fullmatch(r'-?\d+\.\d*', raw):
        return float(raw)
    return raw
