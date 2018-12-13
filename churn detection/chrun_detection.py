import numpy as np
from sklearn import preprocessing, cross_validation , neighbors
import pandas as pd
import pylab as plot
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
#from sklearn.cross_validation import StratifiedKFold
#from sklearn.cross_validation import GridSearchCV
#from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

df_train = pd.read_csv("Train_data.csv")
df_test = pd.read_csv("Test_data.csv")



#df_test.isnull().values.ravel().sum()

df_train['Churn'] =  df_train['Churn'].map(lambda x: 0 if x == " False." else 1)
df_test['Churn'] =  df_test['Churn'].map(lambda x: 0 if x == " False." else 1)
#df_train['not_churn'] = 1 - df_train['Churn']

#df_train.groupby('international plan').agg('sum')[['Churn', 'not_churn']].plot(kind='bar', figsize=(25, 7),
#                                                          stacked=True, color=['g', 'r'])
#plot.show()



train_targets = df_train.Churn
test_targets = df_test.Churn
df_train.drop(['Churn'] ,1 , inplace=True)
df_test.drop(['Churn'] ,1 , inplace=True)
combined = df_train.append(df_test)


#print df_train.shape,df_test.shape,combined.shape

state_dummy = pd.get_dummies(combined['state'] , prefix='state_')
combined = pd.concat([combined,state_dummy] , axis=1)
combined.drop(['state'] , axis=1 , inplace=True)

combined.drop(['account length'] , axis=1 , inplace=True)
combined.drop(['phone number'] , axis=1 , inplace=True)

areacode_dummy = pd.get_dummies(combined['area code'] , prefix='areacode_')
combined = pd.concat([combined,areacode_dummy] , axis=1)
combined.drop(['area code'] , axis=1 , inplace=True)

combined['international plan'] = combined['international plan'].map(lambda x: 0.0 if x == " no" else 1.0)
combined['voice mail plan'] = combined['voice mail plan'].map(lambda x: 0.0 if x == " no" else 1.0)

combined.drop(['total day minutes'] , axis=1 , inplace=True)
combined.drop(['total eve minutes'] , axis=1 , inplace=True)
combined.drop(['total night minutes'] , axis=1 , inplace=True)
combined.drop(['total intl minutes'] , axis=1 , inplace=True)

train = combined[:3333]
test = combined[3333:]


#clf = neighbors.KNeighborsClassifier()

#clf.fit(train,train_targets)

#accuracy = clf.score(test, test_targets)
#0.882423515297
#print(accuracy)

#print train.shape,test.shape

clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train,train_targets)

#accuracy = clf.score(test, test_targets)
#0.94601079784
#print(accuracy)

features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by='importance' , ascending=True , inplace=True)
features.set_index('feature', inplace=True)

features.plot(kind='barh',figsize=(25,25))
plot.show()

model = SelectFromModel(clf,prefit=True)
train_reduced = model.transform(train)
test_reduced = model.transform(test)


#print train.shape,train_reduced.shape

knn = neighbors.KNeighborsClassifier()
logreg = LogisticRegression()
logreg_cv = LogisticRegressionCV()
rf = RandomForestClassifier()
gboost = GradientBoostingClassifier()

models = [knn,logreg, logreg_cv, rf, gboost]

for model in models:
	clf = model.fit(train_reduced,train_targets)
	score = clf.score(test_reduced, test_targets)
	print model.__class__,'accuracy = {0}'.format(score)
	print '****'


