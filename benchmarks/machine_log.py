# -*- coding: utf-8 -*-
"""
Script to get the current machine name, number of DPUs, and diagnostic information.
"""

import re
import socket
import subprocess

import yaml

if __name__ == "__main__":
    # get host name
    hostname = socket.gethostname()

    # get dpu diagnostics
    dpu_diagnostics = subprocess.run(["dpu-diag"], capture_output=True).stdout.decode(
        "utf-8"
    )

    # write to file
    with open("dpu_diag.txt", "w") as f:
        f.write(dpu_diagnostics)

    # get dpu count
    nr_dpus = re.findall(r"\[dpu frequency] (\d+)", dpu_diagnostics)[0]
    nr_dpus = int(nr_dpus)

    # get dpu frequency
    dpu_frequency = re.findall(r"\[dpu frequency\] \d+ DPUs @ (\d+)", dpu_diagnostics)[
        0
    ]
    dpu_frequency = int(dpu_frequency)

    machine = {"hostname": hostname, "nr_dpus": nr_dpus, "frequency": dpu_frequency}
    with open("machine.yaml", "w") as f:
        yaml.dump(machine, f)
