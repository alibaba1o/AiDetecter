import csv
from datasets import load_dataset
#Моя песнь был а красива!
def supersplit(text,label):
    text = text.replace('—','-')
    texts = text.split(' ')
    syllable_count = syllables(texts)
    uppercase_ratio = uppercase(texts)
    sentence_count = text.replace('...','.').replace('?!','.').replace('.','.').replace('!','.').replace('?','.').count('.')
    if sentence_count == 0:
        sentence_count = 1
    char_count = len(text.replace(',', '').replace('.', '').replace('!', '').replace('?', '').replace(' ',''))
    texts = [word.lower() for word in texts]
    set_text = set(texts)
    word_count = len(texts)
    syllable_ratio = syllable_count / word_count
    types_count = len(set_text)
    punctuation_count = text.count('.')+text.count('!')+text.count('?')+text.count(',')+text.count('-')
    avg_word_length = char_count / word_count
    avg_sentence_length = word_count / sentence_count
    unique_word_ratio = types_count / word_count
    avg_syllable_length = syllable_count / word_count
    readabillity = 206.835 - (1.3 * avg_sentence_length) - (60.1 * avg_syllable_length)

    return [text,label,word_count,char_count,sentence_count,avg_word_length,avg_sentence_length,unique_word_ratio,uppercase_ratio,syllable_ratio,punctuation_count,readabillity]
def uppercase(text):
    count = 0
    for word in text:
        if word.capitalize() == word:
            count += 1

    return count/len(text)

def syllables(text):
    consonants = "бвгджзйклмнпрстфхцчшщ"
    vowels = "аеёиоуыэюя"
    numbers = "0123456789"
    syllable_count = 0
    for word in text:
        for i in word:
            if i in vowels:
                syllable_count += 1
            elif len(word)==1 or word[len(word)//2] in numbers:
                syllable_count += 1
    return syllable_count


with open('dataset.csv','r',encoding='utf-8') as file:
    reader = csv.reader(file)
    header = next(reader)
    for row in reader:
        with open('update.csv','a',encoding='utf-8',newline='') as f:
            writer = csv.writer(f)
            text = row[0]
            label = row[1]
            if label.lower() != 'human':
                label = 0
            else:
                label = 1
            writer.writerow(supersplit(text,label))
