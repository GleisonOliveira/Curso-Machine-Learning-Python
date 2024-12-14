from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pickle
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("WebAgg")


folder = "files/credit/"
show_graphics = True

# load the processed data
with open(f'{folder}credit.pkl', 'rb') as f:
    x_credit_training, y_credit_training, x_credit_test, y_credit_test = pickle.load(f)
    
# create the model training
tree_credit_risk = DecisionTreeClassifier(criterion='entropy', random_state=0)
tree_credit_risk.fit(x_credit_training, y_credit_training)

# export the trained model
with open(f'{folder}tree_credit.trained.pkl', 'wb') as f:
    pickle.dump(tree_credit_risk, f)
 
if show_graphics == True: 
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,20))
    previsors = ['income', 'age', 'loan']
    
    print('Ganhos de informação', tree_credit_risk.feature_importances_)
    tree.plot_tree(tree_credit_risk, filled=True, feature_names=previsors, class_names=['0', '1'])

    fig.savefig('credit-arvore-de-decisão.png')
    
    plt.show()