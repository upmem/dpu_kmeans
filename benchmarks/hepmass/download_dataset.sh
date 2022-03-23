mkdir -p data
cd data || exit
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00347/all_test.csv.gz
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00347/all_train.csv.gz
gunzip 'all_t*.csv.gz'
cd ..
python converte_to_pq.py
