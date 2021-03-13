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
with open('article-parser/Articles/article-1.txt') as f:
    tokens = sent_tokenize(f.read())
    for line in tokens:
        file_docs.append(line)

#print("Number of documents:", len(file_docs))


# Convert the sentence tokens into word tokens
gen_docs = [[w.lower() for w in word_tokenize(text)] for text in file_docs]
# print(gen_docs)

# Create Dictionary of unique ID's for each word
dictionary = gensim.corpora.Dictionary(gen_docs)
#print(dictionary.token2id)

# Creating a bag of words
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
#print(corpus)

# Create an an array that contatins the frequency of each word
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

# Apply weights to each word based on how often the word appears in the text
tf_idf = gensim.models.TfidfModel(corpus)
for doc in tf_idf[corpus]:
    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])


#----------------------------------------------------------------------------------
# Create Similarity measure object
sims = gensim.similarities.Similarity('similarity_index', tf_idf[corpus],num_features=len(dictionary))


# Tokenize a second document 
file2_docs = []

with open ('article-parser/Articles/article-2.txt') as f:
    tokens = sent_tokenize(f.read())
    for line in tokens:
        file2_docs.append(line)

print("Number of documents:",len(file2_docs))  
for line in file2_docs:
    query_doc = [w.lower() for w in word_tokenize(line)]
    query_doc_bow = dictionary.doc2bow(query_doc) #update an existing dictionary and create bag of words