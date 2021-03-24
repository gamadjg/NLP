# Article Followed: https://dev.to/coderasha/compare-documents-similarity-using-python-nlp-4odp

import nltk
import numpy as np
import gensim
from nltk.tokenize import word_tokenize, sent_tokenize

data_1 = "Mars is the fourth planet in our solar system. It is second-smallest planet in the Solar System after Mercury. Saturn is yellow planet."
data_2 = "Saturn is the sixth planet from the Sun."

#-----------------------------------------------------------------------------
# Open the txt file, read contents, tokenize each sentence, store each token in an array
sent_token_array = []
sent_tokens = sent_tokenize(data_1)
for line in sent_tokens:
    sent_token_array.append(line)
print("Number of documents:", len(sent_token_array))

# Convert the sentence tokens into word tokens
word_tokens = [[w.lower() for w in word_tokenize(text)] for text in sent_token_array]
print('\nWord tokens:')
print(word_tokens)

# Create Dictionary of unique ID's for each word
dictionary = gensim.corpora.Dictionary(word_tokens)
print('\nID Dictionary:')
print(dictionary.token2id)

# Create a bag of words, an array that contatins the frequency of each word
freq_corpus = [dictionary.doc2bow(word_tokens) for word_tokens in word_tokens]
print('\nFrequency Corpus:')
print(freq_corpus)

# Apply weights to each word based on how often the word appears in the text
print('\nWord weights:')
tf_idf = gensim.models.TfidfModel(freq_corpus)
for doc in tf_idf[freq_corpus]:
    #[[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc]
    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])

#----------------------------------------------------------------------------------
# Create Similarity measure object
sims = gensim.similarities.Similarity('similarity_index', tf_idf[freq_corpus], num_features=len(dictionary))

# Tokenize a second document
file2_docs = []

tokens = sent_tokenize(data_2)
for line in tokens:
    file2_docs.append(line)
#print("Number of documents:", len(file2_docs))

for line in file2_docs:
    query_doc = [w.lower() for w in word_tokenize(line)]
    # update an existing dictionary and create bag of words
    query_doc_bow = dictionary.doc2bow(query_doc)

#--------------------------------------------------------------------------------------

# perform a similarity query against the corpus
query_doc_tf_idf = tf_idf[query_doc_bow]

# print(document_number, document_similarity)
print('Comparing Result:', sims[query_doc_tf_idf])
print(sims[query_doc_tf_idf][2])
