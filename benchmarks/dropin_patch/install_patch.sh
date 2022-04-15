#!/usr/bin/env sh

SKLEARN_PATH = $(/usr/bin/env python3 -c "import sklearn; print(sklearn.__path__[0])")
cp _kmeans.py $SKLEARN_PATH/cluster/_kmeans.py
