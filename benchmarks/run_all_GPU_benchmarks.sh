mkdir -p gpu_results
cd weak_scaling || exit
echo "Running weak scaling benchmark"
python GPU.py
mv weak_scaling_GPU_results.* ../gpu_results/
cd ../strong_scaling || exit
echo "Running strong scaling benchmark"
python GPU.py
mv strong_scaling_GPU_results.* ../gpu_results/
cd ../higgs || exit
if [[ ! -f "./data/higgs.pq" ]]; then
    echo "Downloading Higgs dataset"
    sh ./download_dataset.sh
fi
echo "Running Higgs benchmark"
python GPU.py
mv higgs_GPU_results.* ../gpu_results/
