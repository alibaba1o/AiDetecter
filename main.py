from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from datasets import load_dataset

'''
dataset.set_format('pandas')
text = dataset['train']
text.to_csv("dataset.csv",index=False)
'''

import csv
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
clf = RandomForestClassifier(random_state=0)
reader = csv.reader(open('update.csv','r', encoding='utf-8'))
reader=list(reader)
Y=[]
X=[]
count=0
for row in reader[1:]:
    X.append(list(map(float, row[2:])))
    Y.append(int(row[1]))
#clf.fit(X,Y)
trainX, testX, trainY, testY = train_test_split(X, Y,shuffle=True, test_size=0.2, random_state=0)
clf.fit(trainX, trainY)
y_pred = clf.predict(testX)
accuracy = accuracy_score(testY, y_pred)
print(f'Ai text:{Y.count(0)}, Human text:{Y.count(1)}')
print(accuracy)
