from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pickle
import matplotlib.pyplot as plt

folder = "files/credit_risk/"
previsors = ['história', 'dívida', 'garantias', 'renda']
show_graphics = False

with open(f'{folder}credit_risk.pkl', 'rb') as f:
    x_credit_risk, y_credit_risk = pickle.load(f)
    
tree_credit_risk = DecisionTreeClassifier(criterion='entropy')
tree_credit_risk.fit(x_credit_risk, y_credit_risk)

with open(f'{folder}tree_credit_risk.trained.pkl', 'wb') as f:
    pickle.dump(tree_credit_risk, f)
 
if show_graphics == True: 
    fig, axes = plt.subplots(nrows=1, ncols=1)
    
    print('Ganhos de informação', tree_credit_risk.feature_importances_)
    tree.plot_tree(tree_credit_risk, filled=True, feature_names=previsors, class_names=tree_credit_risk.classes_)

    plt.show()