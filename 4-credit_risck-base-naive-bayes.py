from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.naive_bayes import GaussianNB
import numpy as np
    
folder = 'files/credit_risk/'

# because base is so small we will not divide in tests and training
with open(f'{folder}credit_risk.pkl', 'rb') as f:
    x_credit_risk, y_credit_risk = pickle.load(f)

# insert the data into model
naive_credit_risk = GaussianNB()
naive_credit_risk.fit(x_credit_risk, y_credit_risk)

# read te label encoders
label_encoders = {} 
with open(f'{folder}/label_encoders/label_encoders.pkl', 'rb') as file: 
    label_encoders = pickle.load(file)

# create a list of clients
clients_test_data = np.array([
    ['boa', 'alta', 'nenhuma', 'acima_35'],
    ['ruim', 'alta', 'adequada', '0_15'],
    ['desconhecida', 'baixa', 'adequada', 'acima_35'],
])

# iterate over clients and apply the transforms
for i in range(0, 4):
    clients_test_data[:, i] = label_encoders[i].transform(clients_test_data[:, i])

prevision = naive_credit_risk.predict(np.array(clients_test_data, dtype=np.integer))

# show prevision
print('Riscos: ', prevision)

# show all available classes
print('Classes: ', naive_credit_risk.classes_)

#show the count of recors in each class
print('Contagem: ', naive_credit_risk.class_count_)

# show the class prior in total of percentage
print('Apriori: ', naive_credit_risk.class_prior_)

