import pandas as pd
import numpy as np
import matplotlib.pyplot as plt,mpld3
import seaborn as sns
import plotly.express as px
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
matplotlib.use('WebAgg')

folder = 'files/credit/'

# normalize values to remove inconsistenses
def normalize_age(base_credit):
    # get the mean age to normalize table
    mean_age = base_credit['age'][base_credit['age'] > 0].mean()

    # update the invalid age with mean
    base_credit.loc[base_credit['age'] < 0, 'age'] = mean_age

    # update the null age with mean
    base_credit['age'].fillna(mean_age, inplace=True)
    
    return base_credit

# escale the values and transform in same scale applying standartization of data
def escale_x_values(x_credit):
    scaler_credit = StandardScaler()
    x_credit = scaler_credit.fit_transform(x_credit)
    
    return x_credit

# fix missing iteritems in pandas 2
pd.DataFrame.iteritems = pd.DataFrame.items

show_graphics = False
base_credit = pd.read_csv(f'{folder}credit_data.csv')
base_credit = normalize_age(base_credit)

# get the previsors, the columns 1 to loan and convert then to numpy values
x_credit = base_credit.iloc[:, 1:4].values

# get the classes, the columns default
y_credit = base_credit.iloc[:, 4].values
x_credit = escale_x_values(x_credit)

# sum all records null
# base_credit.isnull().sum()

# return all records with null age
# base_credit.loc[pd.isnull(base_credit['age'])]

# remove all records in a column (age)
# base_credit2 = base_credit.drop('age', axis=1)

# remove all itens with invalid age, filtering and getting the filtered itens index
# base_credit3 = base_credit.drop(base_credit[base_credit['age'] < 0].index)

# Read header of data
# base_credit.head()

# describe the data
# base_credit.describe()

# filter = print(base_credit[base_credit['age'] >= 60])

# use numpy to filter unique values, and return de quantity of each one
# print(np.unique(base_credit['default'], return_counts=True))


# here we separate the data to test and training
x_credit_treinamento, x_credit_test, y_credit_treinamento, y_credit_test = train_test_split(x_credit, y_credit, test_size=0.25, random_state=0)

# here we export the database
with open(f'{folder}credit.pkl', mode='wb') as f:
    pickle.dump([x_credit_treinamento, y_credit_treinamento, x_credit_test, y_credit_test], f)


if show_graphics == True:
    fig1 = plt.figure(figsize=(10, 6))

    # initiate the plot graph, will plot de defaults quantity in bar chart
    plt.subplot(2, 2, 1)
    chart = sns.countplot(x='default', hue="default", data=base_credit)
    chart.set_xlabel('Payed')
    chart.set_title('Payed x Not Payed')
    chart.legend(title='Payeds')

    # initiate the plot of ages
    plt.subplot(2, 2, 2)
    plt.hist(x=base_credit['age'])

    # initiate the plot of incomes
    plt.subplot(2, 2, 3)
    plt.hist(x=base_credit['income'])

    # initiate the plot of loans
    plt.subplot(2, 2, 4)
    plt.hist(x=base_credit['loan'])

    general_chart = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default')
    general_chart.show()

    mpld3.show()