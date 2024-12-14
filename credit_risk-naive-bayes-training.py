import pickle
from sklearn.naive_bayes import GaussianNB
    
folder = 'files/credit_risk/'

# because base is so small we will not divide in tests and training
with open(f'{folder}credit_risk.pkl', 'rb') as f:
    x_credit_risk, y_credit_risk = pickle.load(f)

# insert the data into model
naive_credit_risk = GaussianNB()
naive_credit_risk.fit(x_credit_risk, y_credit_risk)

with open(f'{folder}credit_risk.trained.pkl', 'wb') as f:
    pickle.dump(naive_credit_risk, f)