# Demographic Decipher

A notebook that predicts income level from the UCI Adult (US Census) dataset. I used it to compare classifiers side by side.

## What is in here

- `us_census_data_analysis.ipynb`: data loading, preprocessing, training and evaluation.
- `a9a`: the dataset in LIBSVM format (the Adult data after one-hot encoding).

## Models

- SVM with different kernels
- Decision tree
- Naive Bayes

Each is trained on a train/test split and compared on accuracy.

## Running it

```bash
pip install scikit-learn numpy matplotlib jupyter
jupyter notebook us_census_data_analysis.ipynb
```

Dataset source: [UCI Adult](https://archive.ics.uci.edu/ml/datasets/adult).
