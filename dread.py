import json
import csv

for i in ['generated_essays.json','original_essay.json','paraphrased_essays.json']:
    with open(i,'r',encoding='utf-8') as f:
        data = json.load(f)
        for d in data:
            text = d['text']
            text = text.replace('#','')
            ai = d['source']
            file_name = ''
            if i == 'paraphrased_essays.json':
                file_name = 'paraphrased_data.csv'
            else:
                file_name = 'dataset.csv'
            with open(file_name,'a',encoding='utf-8', newline='') as file:
                 writer = csv.writer(file)

                 writer.writerow([text,ai])
