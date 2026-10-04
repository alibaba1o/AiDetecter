from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import csv
import os
from sklearn.model_selection import StratifiedKFold
from datasets import load_dataset  # сейчас не используется (датасет уже лежит в update.csv), можно убрать

'''
dataset.set_format('pandas')
text = dataset['train']
text.to_csv("dataset.csv",index=False)
'''

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Создаём модель Random Forest. random_state=0 фиксирует случайность,
# чтобы результат был воспроизводимым при повторном запуске.
clf = RandomForestClassifier(random_state=0)

def processingForTraining(dataset):
    # Читаем update.csv целиком в список строк.
    reader = csv.reader(open(dataset, 'r', encoding='utf-8'))
    reader = list(reader)

    y = []  # метки классов: 0 = ИИ, 1 = человек
    X = []  # признаки каждого текста (числа из столбцов 2..11)
    count = 0
    for row in reader[1:]:  # [1:] — пропускаем заголовок
        # row[0] — сам текст, row[1] — метка, row[2:] — вычисленные признаки.
        # В X кладём только признаки (без текста и без метки), переводим их в float.
        X.append(list(map(float, row[2:])))
        # row[1] — метка класса (строка '0' или '1') -> переводим в int.
        y.append(int(row[1]))
    return X,y
X,y = processingForTraining('update.csv')

# Делим данные на обучающую (80%) и тестовую (20%) выборки.
# shuffle=True — строки перемешиваются перед разбиением (иначе первые 480 строк,
# то есть все ИИ-тексты, уйдут отдельно от человеческих).
# random_state=0 — разбиение одно и то же при каждом запуске.
trainX, testX, trainy, testy = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=0)

# Обучаем модель только на train-части.
clf.fit(trainX, trainy)

# Предсказываем метки для test-части (та, что модель не видела при обучении).
y_pred = clf.predict(testX)

# Кросс-валидация: вместо одного разбиения 80/20 данные 5 раз по-разному
# делятся на train/test (shuffle=True — с перемешиванием, иначе первый
# фолд захватит однородный кусок данных и даст заниженную точность).
# StratifiedKFold следит, чтобы в каждом из 5 фолдов сохранялась пропорция
# классов (примерно поровну ИИ и человек), а не только порядок строк.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# cross_val_score обучает и проверяет модель 5 раз (по одному разу на каждый
# фолд в роли теста) и возвращает 5 значений accuracy — по одному на фолд.
score = cross_val_score(clf, X, y, cv=cv)

# Accuracy на одном конкретном разбиении (train_test_split выше) —
# доля верно угаданных меток среди 192 тестовых текстов.
accuracy = accuracy_score(testy, y_pred)

# Проверка баланса классов: должно быть 480 и 480.
print(f'Ai text:{y.count(0)}, Human text:{y.count(1)}')
print(accuracy)   # одно число — точность на одном разбиении
print(score)      # пять чисел — точность на каждом из 5 фолдов кросс-валидации
X_Pharap, y_Pharap = processingForTraining('paraphrased_update.csv')
y_PharapPred = clf.predict(X_Pharap)
#score_pharap = cross_val_score(clf, X_Pharap, y_Pharap, cv=cv)
accuracy_pharap = accuracy_score(y_Pharap, y_PharapPred)
print(f'Ai text:{y_Pharap.count(0)}, Human text:{y_Pharap.count(1)}')
#print(score_pharap)
print(accuracy_pharap)
