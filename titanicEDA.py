# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:01.792877Z","iopub.execute_input":"2021-06-13T16:49:01.793315Z","iopub.status.idle":"2021-06-13T16:49:02.615466Z","shell.execute_reply.started":"2021-06-13T16:49:01.793181Z","shell.execute_reply":"2021-06-13T16:49:02.614458Z"}}

import pandas as pd
import numpy as np    
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:04.874403Z","iopub.execute_input":"2021-06-13T16:49:04.874724Z","iopub.status.idle":"2021-06-13T16:49:04.925125Z","shell.execute_reply.started":"2021-06-13T16:49:04.874696Z","shell.execute_reply":"2021-06-13T16:49:04.924274Z"}}
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:05.242588Z","iopub.execute_input":"2021-06-13T16:49:05.242908Z","iopub.status.idle":"2021-06-13T16:49:05.251924Z","shell.execute_reply.started":"2021-06-13T16:49:05.242879Z","shell.execute_reply":"2021-06-13T16:49:05.250965Z"}}
train.isnull().any()

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:05.553673Z","iopub.execute_input":"2021-06-13T16:49:05.553975Z","iopub.status.idle":"2021-06-13T16:49:05.562404Z","shell.execute_reply.started":"2021-06-13T16:49:05.553948Z","shell.execute_reply":"2021-06-13T16:49:05.561574Z"}}
train.isna().sum()

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:05.835603Z","iopub.execute_input":"2021-06-13T16:49:05.836065Z","iopub.status.idle":"2021-06-13T16:49:05.855063Z","shell.execute_reply.started":"2021-06-13T16:49:05.836034Z","shell.execute_reply":"2021-06-13T16:49:05.854407Z"}}
train.info()

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:06.035006Z","iopub.execute_input":"2021-06-13T16:49:06.035361Z","iopub.status.idle":"2021-06-13T16:49:06.042107Z","shell.execute_reply.started":"2021-06-13T16:49:06.035333Z","shell.execute_reply":"2021-06-13T16:49:06.041071Z"}}
train.shape

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:06.204550Z","iopub.execute_input":"2021-06-13T16:49:06.204903Z","iopub.status.idle":"2021-06-13T16:49:06.237415Z","shell.execute_reply.started":"2021-06-13T16:49:06.204872Z","shell.execute_reply":"2021-06-13T16:49:06.236753Z"}}
train.describe()

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:06.366028Z","iopub.execute_input":"2021-06-13T16:49:06.366529Z","iopub.status.idle":"2021-06-13T16:49:06.632973Z","shell.execute_reply.started":"2021-06-13T16:49:06.366496Z","shell.execute_reply":"2021-06-13T16:49:06.632020Z"}}
sns.heatmap(train.isna(), cbar=False, cmap = 'plasma')
#cbar = color bar shown just right of the heatmap
#cmap = color map 

# %% [markdown]
# **Too much values are missing in the cabin data. We might drop the column later or change it to another feature.**

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:06.678882Z","iopub.execute_input":"2021-06-13T16:49:06.679226Z","iopub.status.idle":"2021-06-13T16:49:06.787100Z","shell.execute_reply.started":"2021-06-13T16:49:06.679180Z","shell.execute_reply":"2021-06-13T16:49:06.786247Z"}}
sns.set_style('whitegrid')
sns.countplot(x = 'Survived', data = train)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:06.842716Z","iopub.execute_input":"2021-06-13T16:49:06.843240Z","iopub.status.idle":"2021-06-13T16:49:06.997931Z","shell.execute_reply.started":"2021-06-13T16:49:06.843171Z","shell.execute_reply":"2021-06-13T16:49:06.997138Z"}}
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Sex', data=train,  palette='hls')  

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:07.012501Z","iopub.execute_input":"2021-06-13T16:49:07.013044Z","iopub.status.idle":"2021-06-13T16:49:07.176582Z","shell.execute_reply.started":"2021-06-13T16:49:07.013001Z","shell.execute_reply":"2021-06-13T16:49:07.175565Z"}}
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Pclass', data=train,  palette='hls') 

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:39.732594Z","iopub.execute_input":"2021-06-13T16:49:39.733088Z","iopub.status.idle":"2021-06-13T16:49:39.974836Z","shell.execute_reply.started":"2021-06-13T16:49:39.733057Z","shell.execute_reply":"2021-06-13T16:49:39.974257Z"}}
sns.distplot(train['Age'].dropna(),kde=False)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:07.378096Z","iopub.execute_input":"2021-06-13T16:49:07.378360Z","iopub.status.idle":"2021-06-13T16:49:07.524667Z","shell.execute_reply.started":"2021-06-13T16:49:07.378335Z","shell.execute_reply":"2021-06-13T16:49:07.523780Z"}}
sns.countplot(x='SibSp', data=train)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:07.576654Z","iopub.execute_input":"2021-06-13T16:49:07.576980Z","iopub.status.idle":"2021-06-13T16:49:07.878260Z","shell.execute_reply.started":"2021-06-13T16:49:07.576949Z","shell.execute_reply":"2021-06-13T16:49:07.877304Z"}}
sns.countplot(x='SibSp', hue='Survived', data=train)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:07.879754Z","iopub.execute_input":"2021-06-13T16:49:07.880025Z","iopub.status.idle":"2021-06-13T16:49:08.031948Z","shell.execute_reply.started":"2021-06-13T16:49:07.879998Z","shell.execute_reply":"2021-06-13T16:49:08.031008Z"}}
sns.countplot(x='Embarked', hue='Survived', data=train)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:08.041477Z","iopub.execute_input":"2021-06-13T16:49:08.041793Z","iopub.status.idle":"2021-06-13T16:49:08.226053Z","shell.execute_reply.started":"2021-06-13T16:49:08.041767Z","shell.execute_reply":"2021-06-13T16:49:08.225219Z"}}
sns.countplot(x='Parch', hue='Survived', data=train)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:33.844361Z","iopub.execute_input":"2021-06-13T16:49:33.844730Z","iopub.status.idle":"2021-06-13T16:49:34.072706Z","shell.execute_reply.started":"2021-06-13T16:49:33.844695Z","shell.execute_reply":"2021-06-13T16:49:34.072049Z"}}
sns.distplot(train['Fare'], kde=False)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:08.811259Z","iopub.execute_input":"2021-06-13T16:49:08.811750Z","iopub.status.idle":"2021-06-13T16:49:08.976368Z","shell.execute_reply.started":"2021-06-13T16:49:08.811705Z","shell.execute_reply":"2021-06-13T16:49:08.975343Z"}}
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:09.214747Z","iopub.execute_input":"2021-06-13T16:49:09.215110Z","iopub.status.idle":"2021-06-13T16:49:09.235275Z","shell.execute_reply.started":"2021-06-13T16:49:09.215068Z","shell.execute_reply":"2021-06-13T16:49:09.234296Z"}}
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
#the line in box represents the average age in each Pclass
#from boxplot we can take average of pclass1,2,3 as 37, 29, 24 respectively

pcl = train['Pclass'].unique()
train['Age'] = train[['Age','Pclass']].apply(new_age, axis=1)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:09.663442Z","iopub.execute_input":"2021-06-13T16:49:09.663780Z","iopub.status.idle":"2021-06-13T16:49:09.830258Z","shell.execute_reply.started":"2021-06-13T16:49:09.663745Z","shell.execute_reply":"2021-06-13T16:49:09.829268Z"}}
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:11.189512Z","iopub.execute_input":"2021-06-13T16:49:11.189869Z","iopub.status.idle":"2021-06-13T16:49:11.201115Z","shell.execute_reply.started":"2021-06-13T16:49:11.189837Z","shell.execute_reply":"2021-06-13T16:49:11.200343Z"}}
train.dropna(axis=0, subset=['Embarked'], inplace=True)
embarked = {'S':0, 'C':1, 'Q':2}
gen = {'male':0, 'female':1}
train['Sex'] = train['Sex'].map(gen)
train['Embarked'] = train['Embarked'].map(embarked)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:12.504061Z","iopub.execute_input":"2021-06-13T16:49:12.504617Z","iopub.status.idle":"2021-06-13T16:49:12.524739Z","shell.execute_reply.started":"2021-06-13T16:49:12.504584Z","shell.execute_reply":"2021-06-13T16:49:12.524091Z"}}
corelation = train.corr().round(3)
corelation

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:13.300132Z","iopub.execute_input":"2021-06-13T16:49:13.300684Z","iopub.status.idle":"2021-06-13T16:49:13.885528Z","shell.execute_reply.started":"2021-06-13T16:49:13.300650Z","shell.execute_reply":"2021-06-13T16:49:13.884320Z"}}
plt.figure(figsize=(15,15))
sns.heatmap(corelation, annot = True)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:13.886847Z","iopub.execute_input":"2021-06-13T16:49:13.887108Z","iopub.status.idle":"2021-06-13T16:49:13.893363Z","shell.execute_reply.started":"2021-06-13T16:49:13.887083Z","shell.execute_reply":"2021-06-13T16:49:13.892425Z"}}
X = train[['Pclass','Sex','Age','SibSp','Embarked','Parch']]
y = train['Survived']

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:13.940748Z","iopub.execute_input":"2021-06-13T16:49:13.941078Z","iopub.status.idle":"2021-06-13T16:49:14.134021Z","shell.execute_reply.started":"2021-06-13T16:49:13.941046Z","shell.execute_reply":"2021-06-13T16:49:14.133297Z"}}
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X,y, test_size=0.3, random_state=42)
X.isnull().any()

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:14.135054Z","iopub.execute_input":"2021-06-13T16:49:14.135430Z","iopub.status.idle":"2021-06-13T16:49:14.254756Z","shell.execute_reply.started":"2021-06-13T16:49:14.135401Z","shell.execute_reply":"2021-06-13T16:49:14.254078Z"}}
from sklearn.linear_model import LogisticRegression

modellr = LogisticRegression()
modellr.fit(train_X, train_y)
pred = modellr.predict(val_X)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:14.255793Z","iopub.execute_input":"2021-06-13T16:49:14.256153Z","iopub.status.idle":"2021-06-13T16:49:14.261120Z","shell.execute_reply.started":"2021-06-13T16:49:14.256105Z","shell.execute_reply":"2021-06-13T16:49:14.260528Z"}}
from sklearn.metrics import accuracy_score
accuracy_score(val_y,pred)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:14.408720Z","iopub.execute_input":"2021-06-13T16:49:14.409164Z","iopub.status.idle":"2021-06-13T16:49:14.525315Z","shell.execute_reply.started":"2021-06-13T16:49:14.409135Z","shell.execute_reply":"2021-06-13T16:49:14.524419Z"}}
from sklearn import tree
modeldt = tree.DecisionTreeClassifier()
modeldt.fit(train_X, train_y)
preddt = modeldt.predict(val_X)
accuracy_score(val_y,preddt)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:14.566945Z","iopub.execute_input":"2021-06-13T16:49:14.567278Z","iopub.status.idle":"2021-06-13T16:49:14.580949Z","shell.execute_reply.started":"2021-06-13T16:49:14.567247Z","shell.execute_reply":"2021-06-13T16:49:14.580281Z"}}
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(train_X,train_y).predict(val_X)
accuracy_score(val_y,y_pred)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:49:15.019246Z","iopub.execute_input":"2021-06-13T16:49:15.019720Z","iopub.status.idle":"2021-06-13T16:49:29.890226Z","shell.execute_reply.started":"2021-06-13T16:49:15.019690Z","shell.execute_reply":"2021-06-13T16:49:29.889280Z"}}
from sklearn.ensemble import RandomForestClassifier
estimators = [100,200,500,750,1000]
depth = [3,5,7,9]
for i in estimators:
    for j in depth:
        model = RandomForestClassifier(n_estimators=i, max_depth=j, random_state=42)
        model.fit(train_X, train_y)
        predictions = model.predict(val_X)
        print(i,j,accuracy_score(val_y,predictions))

# %% [markdown]
# **n_estimators = 100 and max_depth = 5 giving best accuracy so we can use these parameter values in Random Forest Classifier**

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:58:44.364040Z","iopub.execute_input":"2021-06-13T16:58:44.364557Z","iopub.status.idle":"2021-06-13T16:58:44.376853Z","shell.execute_reply.started":"2021-06-13T16:58:44.364524Z","shell.execute_reply":"2021-06-13T16:58:44.376065Z"}}
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


pcl = test['Pclass'].unique()
test['Age'] = test[['Age','Pclass']].apply(new_age, axis=1)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-13T16:58:47.137107Z","iopub.execute_input":"2021-06-13T16:58:47.137679Z","iopub.status.idle":"2021-06-13T16:58:47.320118Z","shell.execute_reply.started":"2021-06-13T16:58:47.137635Z","shell.execute_reply":"2021-06-13T16:58:47.318014Z"}}
embarked = {'S':0, 'C':1, 'Q':2}
gen = {'male':0, 'female':1}
test['Sex'] = test['Sex'].map(gen)
test['Embarked'] = test['Embarked'].map(embarked)
features = ['Pclass','Sex','Age','SibSp','Embarked','Parch']
X = pd.get_dummies(train[features])
y = train['Survived']
X_test = pd.get_dummies(test[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('submissionRFC.csv', index=False)

# %% [code]
