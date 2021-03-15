import nltk
import numpy as np
import gensim
from nltk.tokenize import word_tokenize, sent_tokenize


#-----------------------------------------------------------------------------
# Open the txt file, read contents, tokenize each sentence, store each token in an array
sent_token_array = []
with open('../Articles/article-1.txt', 'r') as f:
    sent_tokens = sent_tokenize(f.read())
    for line in sent_tokens:
        sent_token_array.append(line)
#print("Number of documents:", len(sent_token_array))

# Convert the sentence tokens into word tokens
word_tokens = [[w.lower() for w in word_tokenize(text)] for text in sent_token_array]
#print('\nWord tokens:')
# print(word_tokens)

# Create Dictionary of unique ID's for each word
dictionary = gensim.corpora.Dictionary(word_tokens)
print('\nID Dictionary:')
print(dictionary.token2id)

# Create a bag of words, an array that contatins the frequency of each word
freq_corpus = [dictionary.doc2bow(word_tokens) for word_tokens in word_tokens]
#print('\nFrequency Corpus:')
# print(freq_corpus)

# Apply weights to each word based on how often the word appears in the text
#print('\nWord weights:')
tf_idf = gensim.models.TfidfModel(freq_corpus)
# for doc in tf_idf[freq_corpus]:
#    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])

#----------------------------------------------------------------------------------
# Create Similarity measure object
sims = gensim.similarities.Similarity('tacos_similarity_index', tf_idf[freq_corpus], num_features=len(dictionary))

# Tokenize a second document
sent_token_array_2 = []
with open('../Articles/article-2.txt', 'r') as f:
    sent_tokens = sent_tokenize(f.read())
    for line in sent_tokens:
        sent_token_array_2.append(line)
#print("Number of documents:", len(file2_docs))

for line in sent_token_array_2:
    query_doc = [w.lower() for w in word_tokenize(line)]
    # update an existing dictionary and create bag of words
    query_doc_bow = dictionary.doc2bow(query_doc)

print('\nDictionary 2:')
print(query_doc_bow)

#--------------------------------------------------------------------------------------

# perform a similarity query against the corpus
query_doc_tf_idf = tf_idf[query_doc_bow]

# print(document_number, document_similarity)
print('Comparing Result:', sims[query_doc_tf_idf])
# print(len(sims[query_doc_tf_idf]))
