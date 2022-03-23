mkdir -p data
cd data || exit
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
echo "unzipping data"
gunzip HIGGS.csv
cd ..
echo "converting to parquet format"
python convert_to_pq.py
