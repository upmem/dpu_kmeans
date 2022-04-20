#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from time import sleep

import pandas as pd
import yaml
from tqdm import tqdm


def get_experiments() -> pd.DataFrame:
    """
    Loads the experiments from the params.yaml file.
    """
    script_dir = os.path.dirname(__file__)
    params_file = os.path.join(script_dir, "params.yaml")

    with open(params_file, "r") as f:
        params = yaml.safe_load(f)

    df = pd.DataFrame.from_dict(params, orient="index").stack().to_frame().transpose()
    for col in list(df.columns):
        df = df.explode(col)
    df["data", "n_points"] = df.apply(
        lambda row: row["data", "n_points"] * row["dimm", "n_dpu"]
        if row["data", "scaling"]
        else row["data", "n_points"],
        axis=1,
    )
    return df


def run_benchmark():
    """
    Runs the benchmark.
    """
    df = get_experiments()

    datasets = df.data.drop_duplicates()
    nonconstant_data = datasets.columns[datasets.nunique() > 1]
    for _, dataset in (pbar_data := tqdm(datasets.iterrows(), total=datasets.shape[0])):
        desc = ", ".join([p + ": " + str(dataset[p]) for p in nonconstant_data])
        pbar_data.set_description(f"generating dataset: ({desc})     ")
        sleep(2)
        pbar_data.set_description(f"generating dataset: ({desc}) done")
        dataset_df = df[(df.data == dataset).all(axis=1)]
        trains = dataset_df.train.drop_duplicates()
        for _, train_param in (
            pbar_train := tqdm(trains.iterrows(), total=trains.shape[0], leave=False)
        ):
            pbar_train.set_description(
                f"with train parameters:{train_param.n_clusters}, running CPU     "
            )
            sleep(2)
            pbar_train.set_description(
                f"with train parameters:{train_param.n_clusters}, running CPU done"
            )

            train_param_df = dataset_df[(dataset_df.train == train_param).all(axis=1)]
            dimms = train_param_df.dimm.drop_duplicates()
            for _, dimm_param in (
                pbar_dimm := tqdm(dimms.iterrows(), total=dimms.shape[0], leave=False)
            ):
                pbar_dimm.set_description(
                    f"with dimm parameters:{dimm_param.n_dpu}     "
                )
                sleep(2)
                pbar_dimm.set_description(
                    f"with dimm parameters:{dimm_param.n_dpu} done"
                )


if __name__ == "__main__":
    run_benchmark()
