
# coding: utf-8

# # Preprocessing, building a Pandas dataframe and saving it as a  .csv file

# In[66]:


import re
import sys
import glob
import string
from pprint import pprint
from collections import Counter, OrderedDict

import spacy
nlp = spacy.load('en',disable=['parser', 'tagger', 'ner'])
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm, tqdm_notebook, tnrange
tqdm.pandas(desc='Progress')
from sklearn.metrics import accuracy_score


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

import warnings
warnings.filterwarnings('ignore')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity='all'

folder_path = "/home/anuja/Desktop/mini_deb/*"
file_name = "/home/anuja/Desktop/EmailRecommendation-master (3)/Preprocessing/dataframe3.csv"
sys.path.insert(0, '/home/anuja/Desktop/EmailRecommendation-master (3)/Preprocessing')

import preprocessing
import read_file
import datetime

def extract_debian(text):
    text = text.split('\n\n\n')
    header = text[0].split('\n')
    body = text[1]
    sender = header[2].split(':')[1].split('<')[0]
#     print('Sender',sender)
#     print('Body \n',body)
    return sender,body

def clean_debian(temp):
    temp = re.sub('\n+','\n',temp)
    temp = re.sub('\n',' ',temp)
    temp = re.sub('\t',' ',temp)
    temp = re.sub(' +',' ',temp)
    return temp

def deb_lemmatize(doc):        
    doc = nlp(doc)
    article, skl_texts = '',''
    for w in doc:
        if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num:
            article += " " + w.lemma_
        if w.text == '\n':                
            skl_texts += " " + article
            article = ''       
    return skl_texts

def deb_toppostremoval(temp):
    strings = temp.splitlines()
    temp = ''
    for st in strings:
        st = st.strip()
        if len(st)>0:
            if st[0]=='>':
                continue
            else:
                temp += '\n' + st
    return temp


# In[113]:


from random import randint
import numpy as np
import torch
from models import InferSent
model_version = 1
MODEL_PATH = "/home/anuja/Desktop/BE project/Models/InferSent/infersent1.pkl"
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
infermodel = InferSent(params_model)
infermodel.load_state_dict(torch.load(MODEL_PATH))
use_cuda = False
infermodel = infermodel.cuda() if use_cuda else infermodel
W2V_PATH = '/home/anuja/Desktop/BE project/glove.6B/glove.840B.300d.txt'
#replace with glove.840B.300d.txt
infermodel.set_w2v_path(W2V_PATH)
infermodel.build_vocab_k_words(K=100000)


# In[114]:


df = pd.DataFrame(columns=['body','replier', 'thread_no','embeddings'])
folder = glob.glob(folder_path)
th_no = 0
obj = preprocessing.preprocess()
cnt = 0
count_file = 0
thread_list=[]
try:
    for fol in tqdm_notebook(folder):
        files = glob.glob(fol+'/*.txt')
        threads = []
        for file in files:
            ob = read_file.file_content(file)
            ob.read_file_content()
            threads.append(ob.mail)
            count_file += 1
        sorted_threads = sorted(threads, key=lambda ke: datetime.datetime.strptime(ke['Date'],'%a, %d %b %Y %H:%M:%S %z'))
        thread_list.append(sorted_threads)
except:
    print(fol)
print(len(thread_list))


# In[ ]:


for thr in thread_list:
    flag = 0
    t = ''
    for mail in thr:
        temp = ''
        count_file += 1
        sender = mail['From']
        temp   = mail['content']
        temp = deb_toppostremoval(temp)
        temp = deb_lemmatize(temp)
        temp = clean_debian(temp)
        if temp == '':
            cnt += 1
            print('NULL')
            continue
        temp = obj.replace_tokens(temp)
        if flag==0:
            t = temp
            flag = 1
            continue
        t=t.strip()
        print(str(t))
        print('---------------------------------------')
        #calculate sentence embedding for body and average it into 4096 sized vector
        if t!='' :
            embedding =infermodel.encode( str(t), bsize=1, tokenize=False, verbose=True)
            sent_vec =[]
            numw = 0
            for w in embedding:
                try:
                    if numw == 0:
                        sent_vec = w
                    else:
                        sent_vec = np.add(sent_vec, w)
                    numw+=1
                except:
                    pass
            v = np.asarray(sent_vec) / numw
            print(v.shape)
            print(v)
            v=np.transpose(v)
            print(v.shape)
            print(v)
       # v=np.zeros(4096)
        df = df.append({'body': str(t),'replier':sender, 'thread_no':th_no,'embeddings':v}, ignore_index=True)
        t = t + temp
    th_no += 1

empty.close()
print(cnt)
print(count_file)
print(len(df['body']))
print(len(df['thread_no'].unique()))
print(len(df['replier'].unique()))
print(len(df['embeddings'][0]))
rep_to_index = {}
index = 0
for rep in df['replier']:
    if rep in rep_to_index:
        continue
    else:
        rep_to_index[rep] = index
        index += 1
pprint(len(rep_to_index))
for rep in df['replier']:
    df.loc[df['replier']==rep,'replier'] = rep_to_index[rep]

# Aggregate according to date / make separate thread list |> P flag
# Split test_train_split 
print(df.head)
df.to_csv(file_name)
unique_users = len(df.replier.unique())


# # Indexing of words in vocab

# In[116]:


words = Counter()
for sent in df.body.values:
    words.update(w.text.lower() for w in nlp(sent))
# print(words)

words = sorted(words, key=words.get, reverse=True)
print(words)
words = ['_PAD','_UNK'] + words

word2idx = {o:i for i,o in enumerate(words)}
idx2word = {i:o for i,o in enumerate(words)}
def indexer(s): return [word2idx[w.text.lower()] for w in nlp(s)]


# # Dataset Loading

# In[106]:


# embedding |> flag
class VectorizeData(Dataset):
    def __init__(self, df_path, maxlen=4096):
        self.df = pd.read_csv(df_path, error_bad_lines=False)
        print(self.df.embeddings)
        self.df['body'] = self.df.body.apply(lambda x: x.strip())
        print('Indexing...')
        self.df['bodyidx'] = self.df.body.apply(indexer)
        print('Calculating lengths')
        self.df['lengths'] = 4096
        self.maxlen = 4096
        print(self.maxlen)
        print('Padding')
        self.df['bodypadded'] = self.df.bodyidx.apply(self.pad_data)
     
        
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        X = self.df.bodypadded[idx]
        lens = self.df.lengths[idx]
        y = self.df.replier[idx]
        e=self.df.embeddings[idx]
        return X,y,lens,e
    
    def pad_data(self, s):
        padded = np.zeros((self.maxlen,), dtype=np.int64)
        if len(s) > self.maxlen: padded[:] = s[:self.maxlen]
        else: padded[:len(s)] = s
        return padded


# In[107]:


ds = VectorizeData(file_name)


# In[108]:


input_size = 4096
# input_size = ds.maxlen
hidden_size = 30
num_classes = unique_users
num_epochs = 1
batch_size = 1
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[109]:


# concatenate user vector
# embeddings |>
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
#         x = torch.FloatTensor(x)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# In[110]:


model = NeuralNet(input_size, hidden_size, num_classes).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)  


# In[112]:


train_dl= DataLoader(ds, batch_size=1)
num_batch = len(train_dl)
for epoch in range(num_epochs):
    y_true_train = list()
    y_pred_train = list()
    total_loss_train = 0
    t = tqdm_notebook(iter(train_dl), leave=False, total=num_batch)
    for X,y, lengths,e in t:
    #     X = X.transpose(0,1)
        X = Variable(X.cpu())
        y = Variable(y.cpu())
        lengths = lengths.numpy()
        print(X,y,lengths,e)    
        opt.zero_grad()
        X = X.float()
        pred = model(e)```````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
        # F.nll_loss can be replaced with criterion
        loss = F.nll_loss(pred, y)
        loss.backward()
        opt.step()

        t.set_postfix(loss=loss.data[0])
        pred_idx = torch.max(pred, dim=1)[1]

        y_true_train += list(y.cpu().data.numpy())
        y_pred_train += list(pred_idx.cpu().data.numpy())
        total_loss_train += loss.data[0]

    train_acc = accuracy_score(y_true_train, y_pred_train)
    train_loss = total_loss_train/len(train_dl)
    print(f' Epoch {epoch}: Train loss: {train_loss} acc: {train_acc}')
#Accuracy
# hyperparameter tuning
#Testing
# architecture testing

