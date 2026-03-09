from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def embedding(text):
    result= model.encode(text)
    return result
print(embedding("world"))

def cosine1(a,b):
    return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))

text1=embedding("car")
text2=embedding("fruit")
text3=embedding("apple")
output=cosine1(text1,text2)
output2=cosine1(text2,text3)
print(output)
print(output2)