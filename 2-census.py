import pandas as pd
import numpy as np
import matplotlib.pyplot as plt,mpld3
import seaborn as sns
import plotly.express as px
import matplotlib
from sklearn.preprocessing import StandardScaler
matplotlib.use('WebAgg')

# fix missing iteritems in pandas 2
pd.DataFrame.iteritems = pd.DataFrame.items

base_census = pd.read_csv('files/census.csv')

fig1 = plt.figure(figsize=(10, 6))

# initiate the plot graph, will plot de defaults quantity in bar chart
plt.subplot(2, 2, 1)
sns.countplot(x=base_census['income'], data=base_census, hue='income')

# initiate the plot graph, will plot de hist age chart
plt.subplot(2, 2, 2)
plt.hist(x=base_census['age'])

# initiate the plot graph, will plot de hist education chart
plt.subplot(2, 2, 3)
plt.hist(x=base_census['education-num'])

# initiate the plot graph, will plot de hist hour per week chart
plt.subplot(2, 2, 4)
plt.hist(x=base_census['hour-per-week'])

chart = px.scatter_matrix(base_census, dimensions=['age', 'education-num', 'hour-per-week'], color='education-num')
chart.show()

workclass_chart = px.treemap(base_census, path=['workclass', 'age'])
workclass_chart.show()

occupation_chart = px.treemap(base_census, path=['occupation', 'relationship', 'age'])
occupation_chart.show()

relation_chart = px.parallel_categories(base_census, dimensions=['education','income'])
relation_chart.show()

plt.show()

x_census = base_census.iloc[:, 14].values
y_census = base_census.iloc[14].values
