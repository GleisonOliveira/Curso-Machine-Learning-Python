import pandas as pd
import numpy as np
import matplotlib.pyplot as plt,mpld3
import seaborn as sns
import plotly.express as px
import plotly.tools as tls
import matplotlib
matplotlib.use('WebAgg')

# fix missing iteritems in pandas 2
pd.DataFrame.iteritems = pd.DataFrame.items

base_credit = pd.read_csv('files/credit_data.csv')

# Read header of data
base_credit.head()

# describe the data
base_credit.describe()

# filter = print(base_credit[base_credit['age'] >= 60])

# use numpy to filter unique values, and return de quantity of each one
print(np.unique(base_credit['default'], return_counts=True))

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