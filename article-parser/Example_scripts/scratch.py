import nltk
import numpy as np
import gensim
from nltk.tokenize import word_tokenize, sent_tokenize

data = "Mars is approximately half the diameter of Earth."
# print(word_tokenize(data))
data = "Mars is a cold desert world. It is half the size of Earth. "
# print(sent_tokenize(data))

#-----------------------------------------------------------------------------
# Open the txt file, read contents, tokenize each sentence, store each token in an array
file_docs = []
with open('article-1.txt') as f:
    tokens = sent_tokenize(f.read())
    for line in tokens:
        file_docs.append(line)

#print("Number of documents:", len(file_docs))

#-----------------------------------------------------------------------------------
# Get the text from each tokenized sentence in file_docs, convert the sentence token into word tokens, store it all
#	into an array of arrays
gen_docs = [[w.lower() for w in word_tokenize(text)] for text in file_docs]

# print(gen_docs)

#---------------------------------------------------------------------------------------
# convert every word token into a dictionary value with a unique id
dictionary = gensim.corpora.Dictionary(gen_docs)
print(dictionary.token2id)


#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
# Creating a bag of words
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

print(corpus)
