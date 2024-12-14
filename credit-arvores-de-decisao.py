import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("WebAgg")

show_chart = True
folder = "files/credit/"

# load the trained model
with open(f'{folder}tree_credit.trained.pkl', 'rb') as f:
    tree_credit = pickle.load(f)

# load the training and test data
with open(f'{folder}credit.pkl', 'rb') as f:
    x_credit_trainging, y_credit_trainging, x_credit_test, y_credit_test = pickle.load(f)

# read te label encoders
with open(f'{folder}scalers/scaler_credit.pkl', 'rb') as file: 
    scaler_credit = pickle.load(file)

# create a list of clients
clients_test_data = np.array([
    ['66155.9250950813', '59.017015066929204', '8106.53213128514'],
    ['66952.68884534019', '18.5843359269202', '8770.09923520439'],
])

# Apply the transformers
clients_test_data = scaler_credit.transform(clients_test_data)
    
if show_chart == True:
    cm = ConfusionMatrix(tree_credit)
    cm.fit(x_credit_trainging, y_credit_trainging)
    cm.score(x_credit_test, y_credit_test)

    plt.show()
    
# predict the result
previsions = tree_credit.predict(x_credit_test)
clients_previsions = tree_credit.predict(clients_test_data)

print('Previsões', previsions)
print('Previsões', clients_previsions)

print('Percentual de acertos: ', accuracy_score(y_credit_test, previsions))
print('Matrix de acertos e erros: ', confusion_matrix(y_credit_test, previsions))
print('Classificação dos acertos', classification_report(y_credit_test, previsions))