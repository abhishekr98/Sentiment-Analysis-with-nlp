#importing required pkgs
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer as wnl

df = pd.read_excel('F:/Python DS/Input.xlsx') #importing input data

#importing emotion dictionary
master_lib = pd.read_csv('F:/Python DS/nlp/LoughranMcDonald_MasterDictionary_2020.csv')

pos_words = master_lib['Word'][master_lib.Positive != 0].array
neg_words =  master_lib['Word'][master_lib.Negative != 0].array
complex_words = master_lib['Word'][master_lib.Complexity != 0].array

urls = df[['URL_ID', 'URL']] #using only required input

size = len(urls)
url = ""
headers = {"User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.2.8) Gecko/20100722 Firefox/3.6.8 GTB7.1 (.NET CLR 3.5.30729)", "Referer": "http://example.com"}

#scrapping data
for i in range(size):
    url = urls['URL'][i]
    r = requests.get(url, headers= headers, timeout = 90)
    soup = BeautifulSoup(r.content, 'html5lib')
    text = soup.find_all('p', 'class' == "td-post-content")
    np.savetxt('F:/Python DS/text/' + str(urls["URL_ID"][i]) + '.txt', text, fmt='%s', encoding="utf-8")
    
#creating output dataframe    
op = urls
op = urls.reindex(columns = urls.columns.tolist() + ['POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE','SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'])

# doing analysis and filling the dataframe
for a in range(size):
    with open('F:/Python DS/text/'+ str(urls["URL_ID"][a]) +'.txt', encoding="utf8") as f:
        file = f.read().replace("<p>","").replace("</p>","")
        positive_score, negative_score, complex_word_count, polarity_score, subjective_score = 0, 0, 0, 0, 0
        avg_sentence_len, per_complex_words, fog_index, avg_no_of_words_per_sen, syllable_count =0, 0, 0, 0, 0
        per_pro = 0
        filtered_sentence = []

        upper_case = file.upper()
        text_no_punct = upper_case.translate(str.maketrans('', '', string.punctuation)).replace("’","")

        stop_words = set(stopwords.words('english'))

        sentences = sent_tokenize(upper_case, 'english')
        word_tokens = word_tokenize(text_no_punct, "english")

        for w in word_tokens:
            if w not in stop_words:
                w = wnl().lemmatize(w, pos='v')
                filtered_sentence.append(w)

        for w in filtered_sentence:
            if w in pos_words:
                positive_score+=1
            elif w in neg_words:
                negative_score+=1
            if w in complex_words:
                complex_word_count+=1

        for w in filtered_sentence:
            for l in w.lower():
                if l in ['a','e','i','o','u','y']:
                    syllable_count+=1     

        file_cl = file.translate(str.maketrans('', '', string.punctuation)).replace("’","").replace('—',' ')
        file_split = file_cl.split()
        regex = r"(\b(s?he|S?HE|She|He|it|It|us|Us|we|We|my|My|ours|Ours|I)\b)"

        for w in file_split:
            matches = re.search(regex,w)
            if matches:
                per_pro +=1

        letters = []
        for w in file_split:
            for i in w:
                letters.append(i)

        word_count = len(filtered_sentence)
        total_words = len(upper_case.split())
        total_sen  = len(sentences)


        polarity_score = (positive_score - negative_score)/ ((positive_score + negative_score) + 0.000001)
        subjective_score = (positive_score + negative_score)/ ((word_count) + 0.000001)
        #analysis of readability
        avg_sentence_len = word_count/total_sen
        per_complex_words = complex_word_count/total_words
        fog_index = 0.4*(avg_sentence_len + per_complex_words)

        avg_no_of_words_per_sen = total_words/total_sen
        avg_word_len = len(letters) /total_words


        op.at[a, 'POSITIVE SCORE'] = positive_score
        op.at[a, 'NEGATIVE SCORE'] = negative_score
        op.at[a, 'POLARITY SCORE'] = polarity_score
        op.at[a, 'SUBJECTIVITY SCORE'] = subjective_score
        op.at[a, 'AVG SENTENCE LENGTH'] = avg_sentence_len
        op.at[a, 'PERCENTAGE OF COMPLEX WORDS'] = per_complex_words
        op.at[a, 'FOG INDEX'] =fog_index
        op.at[a, 'AVG NUMBER OF WORDS PER SENTENCE'] = avg_no_of_words_per_sen
        op.at[a, 'COMPLEX WORD COUNT'] = complex_word_count
        op.at[a, 'WORD COUNT'] = word_count
        op.at[a, 'SYLLABLE PER WORD'] = syllable_count
        op.at[a, 'PERSONAL PRONOUNS'] = per_pro
        op.at[a, 'AVG WORD LENGTH'] = avg_word_len
    f.close()
    
op.to_csv('F:/Python DS/Output Data.csv') #exporting output file
print('Completed')