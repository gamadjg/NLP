import nltk
import numpy as np
import gensim
from nltk.tokenize import word_tokenize, sent_tokenize

text = 'This is the space. This is our planet. This is the Mars.'

file_docs = []

# Sentence tokenize the text
tokens = sent_tokenize(text)
for line in tokens:
    file_docs.append(line)

# word tokenize each sentence
gen_docs = [[w.lower() for w in word_tokenize(text)] for text in file_docs]

# Create Dictionary of unique ID's for each word
dictionary = gensim.corpora.Dictionary(gen_docs)

# Create an an array that contatins the frequency of each word
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

# Apply weights to each word based on how often the word appears in the text
tf_idf = gensim.models.TfidfModel(corpus)
for doc in tf_idf[corpus]:
    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])


'''
Output: 
[['space', 0.94], ['the', 0.35]]
[['our', 0.71], ['planet', 0.71]]
[['the', 0.35], ['mars', 0.94]]

'''