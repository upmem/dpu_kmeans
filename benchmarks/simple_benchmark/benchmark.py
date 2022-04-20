#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from copy import deepcopy
from itertools import product
from typing import Tuple

import pandas as pd
import yaml


def unroll_dict(dic: dict) -> "list[dict]":
    """
    Unrolls a dictionary with lists values into a list of dictionaries with scalar values.
    """
    keys, values = zip(*dic.items())
    values = [v if isinstance(v, list) else [v] for v in values]
    return [dict(zip(keys, v)) for v in product(*values)]


def unpack(dic: dict) -> "list[dict]":
    """
    Unpacks the input parameters into a list of experiments.
    """
    return unroll_dict(
        {k: v if not isinstance(v, dict) else unpack(v) for k, v in dic.items()}
    )


def get_experiments() -> Tuple[dict, "list[dict]", "list[dict]"]:
    """
    Loads the experiments from the params.yaml file.
    """

    script_dir = os.path.dirname(__file__)
    params_file = os.path.join(script_dir, "params.yaml")

    with open(params_file, "r") as f:
        # params = yaml.load(f, Loader=yaml.FullLoader)
        params = yaml.safe_load(f)

    dpu_experiments_list = unpack(params)
    dpu_experiments_list = [deepcopy(experiment) for experiment in dpu_experiments_list]

    for experiment in dpu_experiments_list:
        experiment["data"]["n_points"] = (
            int(experiment["data"]["n_points"] * experiment["dimm"]["n_dpu"])
            if experiment["data"]["scaling"] == True
            else int(experiment["data"]["n_points"])
        )

    datasets = set(yaml.dump(experiment["data"]) for experiment in dpu_experiments_list)
    datasets = [yaml.safe_load(d) for d in datasets]

    train_parameters_set = set(
        yaml.dump(experiment["train"]) for experiment in dpu_experiments_list
    )
    train_parameters_set = [yaml.safe_load(e) for e in train_parameters_set]

    return params, dpu_experiments_list, train_parameters_set, datasets


def run_benchmark():
    """
    Runs the benchmark.
    """
    params, dpu_experiments_list, train_parameters_set, datasets = get_experiments()

    # making sure that there is either one dataset, or one per experiment
    assert len(datasets) == 1 or len(datasets) == len(dpu_experiments_list)

    print(datasets)
    for experiment in dpu_experiments_list:
        print(experiment)
    for experiment in cpu_experiments_list:
        print(experiment)


# if __name__ == "__main__":
script_dir = os.path.dirname(__file__)
params_file = os.path.join(script_dir, "params.yaml")

with open(params_file, "r") as f:
    # params = yaml.load(f, Loader=yaml.FullLoader)
    params = yaml.safe_load(f)

datasets = unpack(params["data"])

print(datasets)

train_parameters_set = unpack(params["train"])
dimm_parameters_set = unpack(params["dimm"])

print(train_parameters_set)
print(dimm_parameters_set)

datasets_scaled = []
for dataset in datasets:
    if dataset["scaling"] == True:
        for dimm_param in dimm_parameters_set:
            dataset_scaled = deepcopy(dataset)
            dataset_scaled["n_points"] = int(dataset["n_points"] * dimm_param["n_dpu"])
            dataset_scaled["n_dpu"] = [dimm_param["n_dpu"]]
            datasets_scaled.append(dataset_scaled)
    else:
        dataset_scaled = deepcopy(dataset)
        dataset_scaled["n_points"] = int(dataset["n_points"])
        dataset_scaled["train"]["dimm"]["n_dpu"] = [
            dimm_param["n_dpu"] for dimm_param in dimm_parameters_set
        ]
        datasets_scaled.append(dataset)

print(datasets_scaled)


# pd.json_normalize(params)
df = pd.DataFrame.from_dict(params, orient="index").stack().to_frame().transpose()
for col in list(df.columns):
    df = df.explode(col)
df["data", "n_points"] = df.apply(
    lambda row: row["data", "n_points"] * row["dimm"]["n_dpu"]
    if row["data"]["scaling"]
    else row["data", "n_points"],
    axis=1,
)
