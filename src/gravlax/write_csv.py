import csv
import functools as ft
from pathlib import Path
from typing import Any


def initialise_csv(fn):

    init = False
    def _init_csv(csv_path, csv_dict):

        with open(csv_path, 'w+', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(list(csv_dict))

    @ft.wraps(fn)
    def _fn(csv_path, csv_dict):

        nonlocal init

        if not init:
            init = True
            _init_csv(csv_path, csv_dict)

        return fn(csv_path, csv_dict)

    return _fn


@initialise_csv
def write_dict_to_csv(csv_path: Path, csv_dict: dict[str, Any]) -> None:

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(list(csv_dict.values()))
