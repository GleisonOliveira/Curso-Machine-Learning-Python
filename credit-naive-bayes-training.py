import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('WebAgg')

folder = 'files/credit/'

# open the file with data
with open(f'{folder}credit.pkl', 'rb') as f:
    x_credit_training, y_credit_training, x_credit_test, y_credit_test = pickle.load(f)
    
# training the naive bayes
gaussian_nb = GaussianNB()
naive_credit_data = gaussian_nb
naive_credit_data.fit(x_credit_training, y_credit_training)

# export the trained data
with open(f'{folder}credit.trained.pkl', 'wb') as f:
    pickle.dump(gaussian_nb, f)