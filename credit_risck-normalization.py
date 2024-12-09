import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

folder = 'files/credit_risk/'

# read the base
base_credit_risk = pd.read_csv(f'{folder}risco_credito.csv')

# separate the previsors and class
x_credit_risk = base_credit_risk.iloc[:, 0:4].values
y_credit_risk = base_credit_risk.iloc[:, 4].values

# columns to apply label encoder to transform the categorical data into numerical data (one hot encoder does not will be applied to tests purpouse)
columns = [0,1,2,3]
label_encoders = {}

for i in columns:
    le = LabelEncoder()
    x_credit_risk[:, i] = le.fit_transform(x_credit_risk[:, i])
    label_encoders[i] = le
  
# save the LabelEncoder
with open(f'{folder}label_encoders/label_encoders.pkl', 'wb') as file: pickle.dump(label_encoders, file)
    
# because base is so small we will not divide in tests and training
with open(f'{folder}credit_risk.pkl', 'wb') as f:
    pickle.dump([x_credit_risk, y_credit_risk], f)
