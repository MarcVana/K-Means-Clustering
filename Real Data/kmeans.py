"""
Created on Sun Oct  4 12:47:45 2020

@author: Marc
"""
# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the data
data = pd.read_csv('College_Data.csv')

# Plot (just for visualization)
sns.lmplot(x = 'Outstate', y = 'F.Undergrad', data = data, hue = 'Private',
           fit_reg = False)
plt.savefig('outstate_f-undergrad.png')

# Outstate VS Private Visualization
sns.set_style('darkgrid')
g = sns.FacetGrid(data, hue = 'Private', palette = 'coolwarm')
g.map(plt.hist, 'Outstate', bins = 20, alpha = 0.7)
plt.savefig('outstate_private.png')

# Grad. Rate VS Private Visualization
g = sns.FacetGrid(data, hue = 'Private', palette = 'coolwarm')
g.map(plt.hist, 'Grad.Rate', bins = 20, alpha = 0.7)
plt.savefig('grad-rate_private.png')

# In the plot we saw there was a school with a grad. rate > 100
school = data[data['Grad.Rate'] > 100]
# We need to change it at 100 (so it makes sense)
data['Grad.Rate']['Cazenovia College'] = 100

# K MEANS
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2)
kmeans.fit(data.drop(['Private', 'Unnamed: 0'], axis = 1))

# New dataframe (only for visualization)
d = {'Name': data['Unnamed: 0'],
     'Private': data['Private'],
     'Cluster': kmeans.labels_}
new_df = pd.DataFrame(d)
def check(df):
    if df['Private'] == 'Yes' and df['Cluster'] == 1:
        return 'YES'
    if df['Private'] == 'No' and df['Cluster'] == 0:
        return 'YES'
    return 'NO'
new_df['Same'] = new_df.apply(check, axis = 1)

# Final plot (only for visualization)
sns.countplot(x = 'Same', data = new_df)
plt.xlabel('K Means Clusters & Real Clusters')
plt.savefig('kmeans_results.png')

# Evaluation
# We can do this because we had labeled data
# In a real world project, you will use K Means Algorithm when
# you won't have labeled data
def conversion(x):
    if x == 'Yes':
        return 1
    return 0
from sklearn.metrics import confusion_matrix,classification_report
data['Cluster'] = data['Private'].apply(conversion)
print('------------------------------------------------------')
print('-----CONFUSION MATRIX---------------------------------')
print(confusion_matrix(data['Cluster'], kmeans.labels_))
print('------------------------------------------------------')
print('-----CLASSIFICATION REPORT----------------------------')
print(classification_report(data['Cluster'], kmeans.labels_))


