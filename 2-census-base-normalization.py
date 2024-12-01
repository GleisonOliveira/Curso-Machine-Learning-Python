import pandas as pd
import numpy as np
import matplotlib.pyplot as plt,mpld3
import seaborn as sns
import plotly.express as px
import matplotlib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle
matplotlib.use('WebAgg')

folder = 'files/census/'

# fix missing iteritems in pandas 2
pd.DataFrame.iteritems = pd.DataFrame.items
show_graphics = False
base_census = pd.read_csv(f'{folder}census.csv')

if show_graphics == True:
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

# here we will start the data preparation, separating previsors and class
x_census = base_census.iloc[:, 0:14].values
y_census = base_census.iloc[:, 14].values

# here we will transform the categorical (words) data into numerical data
columns = [1,3,5,6,7,8,9,13]
label_encoders = {}

for i in columns:
    le = LabelEncoder()
    x_census[:, i] = le.fit_transform(x_census[:, i])
    label_encoders[i] = le

with open(f'{folder}label_encoders/label_encoders.pkl', mode='wb') as f: pickle.dump(label_encoders, f)

# here we configure the column transformer to equalize each categorical column in census to prevent one converted value to be considered as more important ( the passthrough is to maintain the other values unchanged)
onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), columns)], remainder="passthrough")

# here we apply the onehotencoder
x_census = onehotencoder_census.fit_transform(x_census).toarray()

# here we apply the scaler to adjust the values
scaler_sensus = StandardScaler()
x_census = scaler_sensus.fit_transform(x_census)

# here we separate the data to test and training
x_census_treinamento, x_census_test, y_census_treinamento, y_census_test = train_test_split(x_census, y_census, test_size=0.15, random_state=0)

with open(f'{folder}census.pkl', mode='wb') as f:
    pickle.dump([x_census_treinamento, y_census_treinamento, x_census_test, y_census_test], f)