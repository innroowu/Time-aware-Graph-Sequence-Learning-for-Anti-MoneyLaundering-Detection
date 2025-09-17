# Time-aware-Graph-Sequence-Learning-for-Anti-MoneyLaundering-Detection


Dataset: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
Download and extract data files to ./data folder.
    
## Dataset Details
```
..                                  SMALL           MEDIUM           LARGE
..                                  HI     LI        HI      LI       HI       LI
.. Date Range HI + LI (2022)         Sep 1-10         Sep 1-16        Aug 1 - Nov 5
.. # of Days Spanned                 10     10        16      16       97       97
.. # of Bank Accounts               515K   705K     2077K   2028K    2116K    2064K
.. # of Transactions                  5M     7M       32M     31M      180M    176M
.. # of Laundering Transactions     5.1K   4.0K       35K     16K      223K    100K
.. Laundering Rate (1 per N Trans)  981   1942       905    1948       807     1750
```
## Requirements:
Python==3.9
torch-2.5.1+cu118
torch-geometric
scikit-learn
scipy
numpy

## Usage
Modify the file's location that calls data in pattern_analyzer.py/evaluate.py/data_loader.py 
(You can directly press Ctrl+F to search for "HI-" or "LI-")

python main.py --data HI-Small --prediction_window 2 --length 32


