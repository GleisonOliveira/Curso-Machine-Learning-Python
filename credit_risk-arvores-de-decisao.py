import pickle
import numpy as np

folder = "files/credit_risk/"

with open(f'{folder}tree_credit_risk.trained.pkl', 'rb') as f:
    tree_credit_risk = pickle.load(f)

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
    
previsions = tree_credit_risk.predict(np.array(clients_test_data, dtype=np.integer))
print(previsions)