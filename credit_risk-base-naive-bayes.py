import pickle
import numpy as np
    
folder = 'files/credit_risk/'

# open the trained model
with open(f'{folder}credit_risk.trained.pkl', 'rb') as f:
    naive_credit_risk = pickle.load(f)
    
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
