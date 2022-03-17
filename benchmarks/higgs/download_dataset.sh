mkdir -p data
cd data || exit
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
gunzip HIGGS.csv
cd ..
python converte_to_pq.py
