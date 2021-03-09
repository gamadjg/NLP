import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

data = "Mars is approximately half the diameter of Earth."
#print(word_tokenize(data))
data = "Mars is a cold desert world. It is half the size of Earth. "
#print(sent_tokenize(data))

file_docs = []

with open ('demofile.txt') as f:
    tokens = sent_tokenize(f.read())
    for line in tokens:
        file_docs.append(line)

print("Number of documents:",len(file_docs))