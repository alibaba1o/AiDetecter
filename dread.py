import json
import csv

for i in ['generated_essays.json','original_essay.json']:
    with open(i,'r',encoding='utf-8') as f:
        data = json.load(f)
        for d in data:
            text = d['text']
            text = text.replace('#','')
            ai = d['source']
            print([text,ai])
            with open('dataset.csv','a',encoding='utf-8', newline='') as file:
                 writer = csv.writer(file)

                 writer.writerow([text,ai])
