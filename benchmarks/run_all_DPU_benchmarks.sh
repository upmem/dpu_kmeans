cd weak_scaling || exit
echo "Running weak scaling benchmark"
python CPU+DPU.py
cd ../strong_scaling || exit
echo "Running strong scaling benchmark"
python CPU+DPU.py
cd ../higgs || exit
if [[ ! -f "./data/higgs.pq" ]]; then
    echo "Downloading Higgs dataset"
    sh ./download_dataset.sh
fi
echo "Running Higgs benchmark"
python CPU+DPU.py
cd ../dimension || exit
echo "Running dimension benchmark"
python CPU+DPU.py
cd ../tasklets || exit
echo "Running tasklets benchmark"
./run_bench.sh
