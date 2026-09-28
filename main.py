from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import csv
import os
from sklearn.model_selection import StratifiedKFold
from datasets import load_dataset

'''
dataset.set_format('pandas')
text = dataset['train']
text.to_csv("dataset.csv",index=False)
'''


os.chdir(os.path.dirname(os.path.abspath(__file__)))
clf = RandomForestClassifier(random_state=0)
reader = csv.reader(open('update.csv','r', encoding='utf-8'))
reader=list(reader)
y=[]
X=[]
count=0
for row in reader[1:]:
    X.append(list(map(float, row[2:])))
    y.append(int(row[1]))
#clf.fit(X,Y)
trainX, testX, trainy, testy = train_test_split(X, y,shuffle=True, test_size=0.2, random_state=0)
clf.fit(trainX, trainy)
y_pred = clf.predict(testX)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
score = cross_val_score(clf, X, y, cv=cv)
accuracy = accuracy_score(testy, y_pred)
print(f'Ai text:{y.count(0)}, Human text:{y.count(1)}')
print(accuracy)
print(score)
