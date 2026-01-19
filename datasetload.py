import numpy as np
import pandas as pd

data = pd.read_csv("sql_injection_dataset.csv")

queries = data["query"].values
labels = data["label"].values

