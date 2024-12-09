import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use("WebAgg")

show_chart = False
folder = "files/census/"
label_encoders = {}

# load test and training data
with open(f"{folder}census.pkl", mode="rb") as f:
    x_census_training, y_census_training, x_census_test, y_census_test = pickle.load(f)

# training naive bayes
census_data = GaussianNB()
census_data.fit(x_census_training, y_census_training)

with open(f'{folder}census.trained.pkl', 'wb') as f:
    pickle.dump(census_data,f)
