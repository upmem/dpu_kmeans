mkdir -p data
cd data || exit
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00344/Activity%20recognition%20exp.zip
unzip Activity\ recognition\ exp.zip
mv Activity\ recognition\ exp/* ./
rm Activity\ recognition\ exp.zip
rm -r Activity\ recognition\ exp
cd ..
python converte_to_pq.py
