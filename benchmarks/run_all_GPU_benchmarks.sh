cd weak_scaling || exit
python GPU.py
cd ../strong_scaling || exit
python GPU.py
cd ../higgs || exit
if [[ ! -f "./data/higgs.pq" ]]; then
    sh ./download_dataset.sh
fi
python GPU.py
