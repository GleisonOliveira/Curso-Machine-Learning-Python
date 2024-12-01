import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd

matplotlib.use("WebAgg")

show_chart = False
folder = "files/census/"
label_encoders = {}

# dic with new client data
data_dict = {
    "age": [39],
    "workclass": ["State-gov"],
    "final-weight": [77516],
    "education": ["Bachelors"],
    "education-num": [13],
    "marital-status": ["Never-married"],
    "occupation": ["Adm-clerical"],
    "relationship": ["Not-in-family"],
    "race": ["White"],
    "sex": ["Male"],
    "capital-gain": [2174],
    "capital-loss": [0],
    "hours-per-week": [40],
    "native-country": ["United-States"],
}

# Criar DataFrame
df = pd.DataFrame(data_dict)

base = pd.read_csv(f"{folder}census_data.csv")

# client data
clients_data = df.iloc[:, 0:14].values

# load test and training data
with open(f"{folder}census.pkl", mode="rb") as f:
    x_census_training, y_census_training, x_census_test, y_census_test = pickle.load(f)

# load the label encoders
with open(f"{folder}label_encoders/label_encoders.pkl", mode="rb") as f:
    label_encoders = pickle.load(f)

# load the onehot encoder
with open(f"{folder}one_hot_encoders/onehotencoder_census.pkl", mode="rb") as f:
    onehotencoder_census = pickle.load(f)

# load o StandardScaler
with open(f"{folder}scalers/scaler_census.pkl", "rb") as f:
    scaler_census = pickle.load(f)

# training naive bayes
census_data = GaussianNB()
census_data.fit(x_census_training, y_census_training)

# make a prevision of test data
prevision = census_data.predict(x_census_test)

if show_chart == True:
    cm = ConfusionMatrix(census_data)
    cm.fit(x_census_training, y_census_training)
    cm.score(x_census_test, y_census_test)

    plt.show()

# create the client to test
columns = [1, 3, 5, 6, 7, 8, 9, 13]
for index, value in enumerate(columns):
    clients_data[:, value] = label_encoders[value].transform(clients_data[:, value])

# apply the OneHotEncoder
clients_data = onehotencoder_census.transform(clients_data).toarray()

# apply the StandardScaler
clients_data = scaler_census.transform(clients_data)

# get client prevision
client_prevision = census_data.predict(clients_data)

print('Previsào do cliente', client_prevision[0])

print('Previsões', prevision)
print('Percentual de acertos: ', accuracy_score(y_census_test, prevision))
print('Matrix de acertos e erros: ', confusion_matrix(y_census_test, prevision))
print('Classificação dos acertos', classification_report(y_census_test, prevision))


