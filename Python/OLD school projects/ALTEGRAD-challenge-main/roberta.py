import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords 
import re
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

################################################################
##### roBERTa embedding : this script assumes the cleaning #####
##### of the abstract.txt has been accomplished            #####
################################################################
df = pd.read_csv('data//abstracts_processed.txt', delimiter = "\n",header = None)
df = df.rename(columns={0:'A'})
df = pd.DataFrame(df.A.str.split('----',1).tolist(),
                                 columns = ['id','sentences'])

model = SentenceTransformer('paraphrase-distilroberta-base-v1',device='cuda')
f = open("paper_embeddings.txt","w")
for i in tqdm(range(len(list(df.sentences)))):
    f.write(str(df.id[i])+":"+np.array2string(model.encode(df.sentences[i],
                                                           device = "cuda:0"), 
                          formatter={'float_kind':lambda x: "%.8f" % x})+"\n")    
f.close()