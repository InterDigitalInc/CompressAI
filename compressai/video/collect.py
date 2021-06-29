import argparse
import functools
import json
import re
import sys

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def collect(dirpath: Path) -> Dict[str, Any]:
    # collect for all sequences
    paths = Path(dirpath).glob("*_qp*.json")
    results: Dict[int, Any] = defaultdict(functools.partial(defaultdict, list))
    for p in paths:
        qp = int(re.findall(r".*_qp([0-9]+)", p.stem)[0])
        data = json.load(p.open("r"))
        for k, v in data.items():
            results[qp][k].append(v)

    # aggregate data
    qps = sorted(results.keys(), reverse=True)
    out: Dict[str, List[Any]] = defaultdict(list)
    out["qp"] = qps
    for qp in qps:
        for k, v in results[qp].items():
            out[k].append(np.mean(v))
    return out


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirpath", type=str, help="results directory")
    return parser


def main(args: Any = None) -> None:
    if args is None:
        args = sys.argv[1:]
    parser = create_parser()
    args = parser.parse_args(args)

    results = collect(args.dirpath)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
