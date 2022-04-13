for n_tasklets in $(seq 1 32)
do
    cd ../../
    sed -i "s/DNR_TASKLETS=[0-9]\+/DNR_TASKLETS=$n_tasklets/g" setup.py
    pip install --no-build-isolation -e .
    cd -
    python CPU+DPU.py $n_tasklets
done
