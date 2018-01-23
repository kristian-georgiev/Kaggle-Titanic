#Test project
import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn import svm
#Load the data
train_df = pd.read_csv('train.csv', header=0)
test_df = pd.read_csv('test.csv', header=0)

train_df = train_df.fillna(0)
test_df = test_df.fillna(0)

train_df['Sex'].replace(['female','male'],[0,1],inplace=True)
test_df['Sex'].replace(['female','male'],[0,1],inplace=True)


features_train = train_df[['Age','Sex']]
features_test = test_df[['Age','Sex']]
results_train = train_df['Survived']
features_test = test_df

print(features_train)

clf = svm.SVC()
clf.fit(features_train, results_train)
results_test = clf.predict(features_test)
print(results_test)
#Visualize things
sns.stripplot(x='Age', y='Sex', hue='Survived', data=train_df.loc[train_df['Survived'] == False], jitter=True)
plt.show()
