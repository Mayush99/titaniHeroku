import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('train.csv')

def new_age(row):
    age, pcl = row[0], row[1]
    if pd.isnull(age):
        if pcl == 1:
            return 37
        elif pcl == 2:
            return 29
        else:
            return 24
    else:
        return age

train['Age'] = train[['Age','Pclass']].apply(new_age, axis=1)

embarked = {'S':0, 'C':1, 'Q':2}
gen = {'male':0, 'female':1}
train['Sex'] = train['Sex'].map(gen)
train['Embarked'] = train['Embarked'].map(embarked)
train.dropna(axis=0, subset=['Embarked'], inplace=True)

X = train[['Pclass','Sex','Age','SibSp','Embarked','Parch']]
y = train['Survived']

# print(train.head())
model = RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(X, y)
#
# #saving model to disk
pickle.dump(model, open('titanicModel.pkl', 'wb'))