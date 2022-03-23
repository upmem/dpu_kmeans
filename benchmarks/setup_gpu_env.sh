if command -v mamba &> /dev/null
then
    mamba create -n rapids-22.02 -c rapidsai -c nvidia -c conda-forge \
        cuml=22.02 python=3.9 cudatoolkit=11.5 \
        hurry.filesize scikit-learn tqdm
elif command -v conda &> /dev/null
then
    conda create -n rapids-22.02 -c rapidsai -c nvidia -c conda-forge \
        cuml=22.02 python=3.9 cudatoolkit=11.5 \
        hurry.filesize scikit-learn tqdm
else
    echo "conda could not be found, please install conda or mamba"
    exit
fi
