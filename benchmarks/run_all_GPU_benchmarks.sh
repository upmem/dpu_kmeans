cd weak_scaling || exit
echo "Running weak scaling benchmark"
python GPU.py
cd ../strong_scaling || exit
echo "Running strong scaling benchmark"
python GPU.py
cd ../higgs || exit
if [[ ! -f "./data/higgs.pq" ]]; then
    echo "Downloading Higgs dataset"
    sh ./download_dataset.sh
fi
echo "Running Higgs benchmark"
python GPU.py
