
# coding: utf-8

# # Preprocessing, building a Pandas dataframe and saving it as a  .csv file

# In[1]:


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


folder_path = "/home/niki/Documents/BE_Project/EmailRecommendation/Scraping/debian_dataset/*"
file_name = "/home/niki/Documents/BE_Project/my_EmailRecommmendation/model/dataframe3.csv"
sys.path.insert(0, '/home/niki/Documents/BE_Project/my_EmailRecommmendation/Preprocessing')


import preprocessing
import read_file
import datetime


def get_sender(msg):
    msg = email.message_from_string(msg)
    mfrom = msg['From'].split('<')[0]
    return mfrom

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

df = pd.DataFrame(columns=['body','replier', 'thread_no'])
users = []
folder = glob.glob(folder_path)
th_no = 0
obj = preprocessing.preprocess()
cnt = 0
count_file = 0
thread_list=[]
try:
    for fol in tqdm_notebook(folder):
        files = glob.glob(fol+'/*.txt')
        flag = 0
        t = ''
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
    for mail in thr:
        count_file += 1
        sender = mail['From']
        temp   = mail['content']
        users.append(sender)
        temp = deb_toppostremoval(temp)
        temp = deb_lemmatize(temp)
        temp = clean_debian(temp)
        if temp == '':
            cnt += 1

            continue
        temp = obj.replace_tokens(temp)
        if flag==0:
            t = temp
            flag = 1
            continue
        df = df.append({'body': str(t),'replier':sender, 'thread_no':th_no}, ignore_index=True)
        t = t + temp

#         break
#     break

print(cnt)
print(count_file)
print(len(df['body']))
print(len(df['thread_no'].unique()))
print(len(df['replier'].unique()))
rep_to_index = {}
count = 0
for rep in df['replier']:
    if rep in rep_to_index:
        continue
    else:
        rep_to_index[rep] = count
        count += 1
pprint(len(rep_to_index))
for rep in df['replier']:
    df.loc[df['replier']==rep,'replier'] = rep_to_index[rep]
print(df.head)
df.to_csv(file_name)
unique_users = len(df.replier.unique())


# # Indexing of words in vocab

# In[2]:


words = Counter()
for sent in df.body.values:
    words.update(w.text.lower() for w in nlp(sent))

words = sorted(words, key=words.get, reverse=True)
words = ['_PAD','_UNK'] + words

word2idx = {o:i for i,o in enumerate(words)}
idx2word = {i:o for i,o in enumerate(words)}
def indexer(s): return [word2idx[w.text.lower()] for w in nlp(s)]


# # Dataset Loading

# In[3]:


class VectorizeData(Dataset):
    def __init__(self, df_path, maxlen=10):
        self.df = pd.read_csv(df_path, error_bad_lines=False)
        self.df['body'] = self.df.body.apply(lambda x: x.strip())
        print('Indexing...')
        self.df['bodyidx'] = self.df.body.apply(indexer)
        print('Calculating lengths')
        self.df['lengths'] = self.df.bodyidx.apply(len)
        self.maxlen = max(self.df['lengths'])
        print('Padding')
        self.df['bodypadded'] = self.df.bodyidx.apply(self.pad_data)
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        X = self.df.bodypadded[idx]
        lens = self.df.lengths[idx]
        y = self.df.replier[idx]
        return X,y,lens
    
    def pad_data(self, s):
        padded = np.zeros((self.maxlen,), dtype=np.int64)
        if len(s) > self.maxlen: padded[:] = s[:self.maxlen]
        else: padded[:len(s)] = s
        return padded


# In[4]:


ds = VectorizeData(file_name)


# In[10]:


input_size = ds.maxlen
hidden_size = 30
num_classes = unique_users
num_epochs = 5
batch_size = 1
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[11]:


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


# In[12]:


model = NeuralNet(input_size, hidden_size, num_classes).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)  


# In[ ]:


train_dl= DataLoader(ds, batch_size=1)
num_batch = len(train_dl)
for epoch in range(num_epochs):
    y_true_train = list()
    y_pred_train = list()
    total_loss_train = 0
    t = tqdm_notebook(iter(train_dl), leave=False, total=num_batch)
    for X,y, lengths in t:
    #     X = X.transpose(0,1)
        X = Variable(X.cpu())
        y = Variable(y.cpu())
        lengths = lengths.numpy()

        opt.zero_grad()
        X = X.float()
        pred = model(X)
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

