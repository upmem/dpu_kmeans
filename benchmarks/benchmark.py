#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from collections.abc import Sequence
from random import random
from time import perf_counter

import numpy as np
import pandas as pd
import yaml
from dropin_patch._kmeans_w_perf import KMeans as CPUKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score
from tqdm import tqdm

from dpu_kmeans import KMeans as DPUKMeans
from dpu_kmeans import _dimm


def get_int_keys(d: dict, prefix=()) -> list:
    """
    Recursively returns the keys of a dictionary that hold integers.
    """
    keys = []
    for k, v in d.items():
        if (
            isinstance(v, int)
            and not isinstance(v, bool)
            or isinstance(v, list)
            and all(isinstance(n, int) for n in v)
        ):
            rec_key = prefix + (k,)
            keys.append(rec_key)
        elif isinstance(v, dict):
            keys.extend(get_int_keys(v, prefix + (k,)))
    return keys


def get_available_dpus() -> int:
    """
    Returns the number of available DPUs
    """
    machine_file = "machine.yaml"
    try:
        with open(machine_file, "r") as f:
            machine = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        return np.inf
    return machine["nr_dpus"]


def get_experiments() -> pd.DataFrame:
    """
    Loads the experiments from the parameters yaml file.
    """
    # load the params.yaml file as a dictionary
    params_file = "params.yaml"

    with open(params_file, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # get the experiment name and remove it from parameters
    exp_name = params["exp_name"]
    del params["exp_name"]

    # convert the dictionary to a pandas DataFrame and explode the experiments
    df = pd.DataFrame.from_dict(params, orient="index").stack().to_frame().transpose()
    for col in list(df.columns):
        df = df.explode(col, ignore_index=True)

    # adjust number of points if we want it to scale with DPUs
    if "n_points_per_dpu" in params["data"]:
        df["data", "n_points"] = df.apply(
            lambda row: int(row["data", "n_points_per_dpu"] * row["dimm", "n_dpu"]),
            axis=1,
        )
    elif "n_points" in params["data"]:
        df["data", "n_points"] = df["data", "n_points"].astype(int)

    # adjust random number generator seed if we want it to be identical with data generation
    if any(df["train", "random_state"] == "like_data"):
        df["train", "random_state"] = df.apply(
            lambda row: int(row["data", "random_state"])
            if row["train", "random_state"] == "like_data"
            else int(row["train", "random_state"]),
            axis=1,
        )

    # convert integer columns back to int type as this was lost in the dataframe creation
    integer_columns = get_int_keys(params)
    for column in df:
        if column in integer_columns:
            df[column] = df[column].astype(int)

    # adding a layer to the multi-index
    df = pd.concat([df], axis=1, keys=["inputs"])

    return df, exp_name


def get_desc(nonconstants: Sequence, params: pd.DataFrame) -> str:
    """
    Returns the description of a set of parameters.
    """
    return (
        "(" + ", ".join([p + ": " + str(params[p]) for p in nonconstants]) + ")"
        if len(nonconstants) > 0
        else ""
    )


def generate_dataset(**kwargs) -> np.ndarray:
    """
    Generates a dataset
    """
    data, _ = make_blobs(
        n_samples=kwargs["n_points"],
        n_features=kwargs["n_dim"],
        centers=kwargs["centers"],
        random_state=kwargs["random_state"],
    )
    return data


def load_dataset(**kwargs) -> np.ndarray:
    """
    Loads a dataset
    """
    # script_dir = os.path.dirname(__file__)
    dataset_name = kwargs["name"]
    dataset_file = os.path.join("data", dataset_name + ".pq")
    df = pd.read_parquet(dataset_file)

    data = np.require(
        df.iloc[:, 1:].to_numpy(dtype=np.float32), requirements=["C", "A", "O"]
    )

    return data


def get_dataset(**kwargs) -> np.ndarray:
    """
    Generates or load a dataset
    """
    if kwargs["synthetic"]:
        return generate_dataset(**kwargs)
    else:
        return load_dataset(**kwargs)


def experiment_outputs(df: pd.DataFrame, exp_name: str) -> None:
    """
    Outputs the results of the experiment
    """
    # output the entire benchmarks table
    df.to_csv("benchmarks.csv", index=False)

    # output the important results table
    important_input_columns = df.inputs.columns[df.inputs.nunique() > 1]
    # if (
    #     ("data", "n_points_per_dpu") in important_input_columns
    #     and ("data", "n_points") in important_input_columns
    # ):
    #     important_input_columns = important_input_columns.drop(("data", "n_points"))
    if important_input_columns.empty:
        important_input_columns = ("dimm", "n_dpu")
    important_output_columns = [(c[1:]) for c in df.columns if c[2] in ("train_times",)]

    df_readable = pd.concat(
        (df.inputs[important_input_columns], df.results[important_output_columns]),
        axis=1,
    )
    # df_readable.set_index(important_input_columns.to_list(), inplace=True)
    df_readable.columns = ["_".join(col) for col in df_readable.columns.values]
    # param_index = "--".join(["_".join(name) for name in df_readable.index.names])
    # df_readable.index = df_readable.index.to_flat_index()
    # df_readable.index.rename(param_index, inplace=True)
    df_readable = df_readable.dropna()
    df_readable.to_csv("plots.csv", index=False)

    df_readable = df_readable.set_index(df_readable.columns[0])
    dict_readable = df_readable.to_dict(orient="index")
    # add a performance ratio
    for key in dict_readable:
        if (
            "cpu_train_times" in dict_readable[key]
            and "dpu_train_times" in dict_readable[key]
        ):
            dict_readable[key]["speedup"] = (
                dict_readable[key]["cpu_train_times"]
                / dict_readable[key]["dpu_train_times"]
            )
    # add best speedup
    dict_readable["best_speedup"] = max(
        exp["speedup"] for exp in dict_readable.values()
    )
    # add a top level index to let dvc know these are separate experiments
    dict_readable = {exp_name: dict_readable}
    with open("metrics.json", "w") as f:
        json.dump(dict_readable, f, indent=2)
        f.write("\n")  # add a newline because Py JSON does not
    # df_readable.to_json("metrics.json", orient="index")


def log_cpu(
    df: pd.DataFrame,
    dataset: pd.DataFrame,
    train_param: pd.DataFrame,
    CPU_kmeans: CPUKMeans,
    data: np.ndarray,
    cpu_time: float,
) -> None:
    """
    Logs the CPU results in the experiment
    """
    # logging the results
    cpu_index = (df.inputs.data == dataset).all(axis=1) & (
        df.inputs.train == train_param
    ).all(axis=1)
    df.loc[cpu_index, ("results", "cpu", "times")] = cpu_time
    df.loc[cpu_index, ("results", "cpu", "train_times")] = CPU_kmeans.train_time_
    df.loc[
        cpu_index, ("results", "cpu", "preprocessing_times")
    ] = CPU_kmeans.preprocessing_timer_
    df.loc[
        cpu_index, ("results", "cpu", "single_kmeans_times")
    ] = CPU_kmeans.main_loop_timer_
    df.loc[cpu_index, ("results", "cpu", "iterations")] = CPU_kmeans.n_iter_
    df.loc[cpu_index, ("results", "cpu", "times_one_iter")] = (
        CPU_kmeans.main_loop_timer_ / CPU_kmeans.n_iter_
    )

    # computing score
    df.loc[cpu_index, ("results", "cpu", "score")] = calinski_harabasz_score(
        data, CPU_kmeans.labels_
    )


def log_dpu(
    df: pd.DataFrame,
    dataset: pd.DataFrame,
    train_param: pd.DataFrame,
    dimm_param: pd.DataFrame,
    DPU_kmeans: DPUKMeans,
    CPU_kmeans: CPUKMeans,
    data: np.ndarray,
    dpu_time: float,
    DPU_init_time: float,
) -> None:
    """
    Logs the DPU results in the experiment
    """
    # logging the results
    dimm_index = (
        (df.inputs.data == dataset).all(axis=1)
        & (df.inputs.train == train_param).all(axis=1)
        & (df.inputs.dimm == dimm_param).all(axis=1)
    )
    df.loc[dimm_index, ("results", "dpu", "times")] = dpu_time
    df.loc[dimm_index, ("results", "dpu", "train_times")] = DPU_kmeans.train_time_
    df.loc[dimm_index, ("results", "dpu", "init_times")] = DPU_init_time
    df.loc[
        dimm_index, ("results", "dpu", "preprocessing_times")
    ] = DPU_kmeans.preprocessing_timer_
    df.loc[dimm_index, ("results", "dpu", "cpu_pim_times")] = DPU_kmeans.cpu_pim_time_
    df.loc[dimm_index, ("results", "dpu", "pim_cpu_times")] = DPU_kmeans.pim_cpu_time_
    df.loc[dimm_index, ("results", "dpu", "inertia_times")] = DPU_kmeans.inertia_timer_
    df.loc[
        dimm_index, ("results", "dpu", "reallocate_times")
    ] = DPU_kmeans.reallocate_timer_
    df.loc[
        dimm_index, ("results", "dpu", "single_kmeans_times")
    ] = DPU_kmeans.main_loop_timer_
    df.loc[dimm_index, ("results", "dpu", "kernel_runtime")] = DPU_kmeans.dpu_run_time_
    df.loc[dimm_index, ("results", "dpu", "inter_pim_core_times")] = (
        DPU_kmeans.main_loop_timer_ - DPU_kmeans.dpu_run_time_
    )
    df.loc[dimm_index, ("results", "dpu", "iterations")] = DPU_kmeans.n_iter_
    df.loc[dimm_index, ("results", "dpu", "times_one_iter")] = (
        DPU_kmeans.main_loop_timer_ / DPU_kmeans.n_iter_
    )

    # computing score
    df.loc[dimm_index, ("results", "dpu", "score")] = calinski_harabasz_score(
        data, DPU_kmeans.labels_
    )
    df.loc[dimm_index, ("results", "dpu", "cross_score")] = adjusted_rand_score(
        CPU_kmeans.labels_, DPU_kmeans.labels_
    )

    # logging real number of DPUs
    if dimm_param["n_dpu"] == 0:
        df.loc[dimm_index, ("inputs", "dimm", "n_dpu")] = _dimm.ctr.get_nr_dpus()


def run_benchmark(verbose: bool = False) -> None:
    """
    Runs the benchmark.
    """
    # check number of available DPUs
    n_available_dpu = get_available_dpus()

    # load the experiments
    df, exp_name = get_experiments()

    # run the experiments

    # get unique dataset parameters
    datasets = df.inputs.data.drop_duplicates()
    nonconstant_data = datasets.columns[datasets.nunique() > 1]
    for _, dataset in (pbar_data := tqdm(datasets.iterrows(), total=datasets.shape[0])):
        desc = get_desc(nonconstant_data, dataset)
        pbar_data.set_description(f"getting dataset {desc}")

        ##################################################
        #                   DATA GEN                     #
        ##################################################

        data = get_dataset(**dataset)

        pbar_data.set_description(f"with dataset    {desc}")
        dataset_df = df[(df.inputs.data == dataset).all(axis=1)]
        trains = dataset_df.inputs.train.drop_duplicates()
        nonconstant_train = trains.columns[trains.nunique() > 1]
        for _, train_param in (
            pbar_train := tqdm(trains.iterrows(), total=trains.shape[0], leave=False)
        ):
            desc = get_desc(nonconstant_train, train_param)
            pbar_train.set_description(f"running CPU {desc}")

            ##################################################
            #                   CPU PERF                     #
            ##################################################

            # perform the clustering on CPU
            tic = perf_counter()
            CPU_kmeans = CPUKMeans(
                init="random",
                verbose=verbose,
                copy_x=False,
                algorithm="full",
                **train_param,
            )
            CPU_kmeans.fit(data)
            toc = perf_counter()
            cpu_time = toc - tic

            pbar_train.set_description(f"scoring CPU {desc}")

            # logging the results
            log_cpu(df, dataset, train_param, CPU_kmeans, data, cpu_time)

            pbar_train.set_description(f"running DPU {desc}")
            train_param_df = dataset_df[
                (dataset_df.inputs.train == train_param).all(axis=1)
            ]
            dimms = train_param_df.inputs.dimm.drop_duplicates()
            nonconstant_dimm = df.inputs.dimm.columns[df.inputs.dimm.nunique() > 1]
            for _, dimm_param in (
                pbar_dimm := tqdm(dimms.iterrows(), total=dimms.shape[0], leave=False)
            ):
                desc = get_desc(nonconstant_dimm, dimm_param)
                pbar_dimm.set_description(f"on dimm {desc}")

                ##################################################
                #                   DPU PERF                     #
                ##################################################

                # check that we have enough available DPUs for this experiment
                if dimm_param["n_dpu"] > n_available_dpu:
                    continue

                # load the DPUS
                _dimm.free_dpus()
                tic = perf_counter()
                _dimm.set_n_dpu(dimm_param["n_dpu"])
                _dimm.load_kernel("kmeans", verbose)
                toc = perf_counter()
                DPU_init_time = toc - tic

                # perform the clustering on DPU
                tic = perf_counter()
                DPU_kmeans = DPUKMeans(
                    init="random",
                    verbose=verbose,
                    copy_x=False,
                    reload_data=True,
                    **train_param,
                )
                DPU_kmeans.fit(data)
                toc = perf_counter()
                dpu_time = toc - tic

                pbar_dimm.set_description(f"scoring {desc}")

                # logging the results
                log_dpu(
                    df,
                    dataset,
                    train_param,
                    dimm_param,
                    DPU_kmeans,
                    CPU_kmeans,
                    data,
                    dpu_time,
                    DPU_init_time,
                )

                # writing outputs at every iteration in case we crash early
                experiment_outputs(df, exp_name)

    return df


if __name__ == "__main__":
    df = run_benchmark()
