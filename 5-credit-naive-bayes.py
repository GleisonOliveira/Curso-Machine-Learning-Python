import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('WebAgg')

folder = 'files/credit/'

# open the file with data
with open(f'{folder}credit.pkl', 'rb') as f:
    x_credit_training, y_credit_training, x_credit_test, y_credit_test = pickle.load(f)
    
# training the naive bayes
naive_credit_data = GaussianNB()
naive_credit_data.fit(x_credit_training, y_credit_training)

# make previsions of test data
previsions = naive_credit_data.predict(x_credit_test)
prevision_client = naive_credit_data.predict([[70000,80.017015066929204,10000]])

cm = ConfusionMatrix(naive_credit_data)
cm.fit(x_credit_training, y_credit_training)
cm.score(x_credit_test, y_credit_test)

print(prevision_client)
print('Percentual de acertos: ', accuracy_score(y_credit_test, previsions))
print('Matrix de acertos e erros: ', confusion_matrix(y_credit_test, previsions))
print('Classificação dos acertos', classification_report(y_credit_test, previsions))

plt.show()