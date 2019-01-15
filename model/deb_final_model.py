
# coding: utf-8

# # Preprocessing, building a Pandas dataframe and saving it as a  .csv file

# In[271]:


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
from sklearn.preprocessing import LabelEncoder


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

import pickle

import warnings
warnings.filterwarnings('ignore')

BASE_PATH = '/home/niki/Documents/BE_Project/EmailRecommendation'

folder_path = "/home/niki/Documents/BE_Project/my_EmailRecommmendation/Scraping/mini_deb/*"
file_name = BASE_PATH + "/model/dataframe3.csv"
file_name1 = BASE_PATH + "/model/dataframe4.csv"
file_name2 = BASE_PATH + "/model/dataframe5.csv"
sys.path.insert(0, BASE_PATH + '/Preprocessing')
PATH = BASE_PATH + '/model/first_model.pt'

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

df = pd.DataFrame()
folder = glob.glob(folder_path)
obj = preprocessing.preprocess()
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


# In[272]:
df_trn = pd.DataFrame()
df_tst = pd.DataFrame()
split_date = datetime.datetime.strptime('01 Sep 2018 23:01:14 +0000', '%d %b %Y %H:%M:%S %z')

users = []
trn_users = []
tst_users = []
th_no = 0
cnt = 0
for thr in thread_list:
    start_date = ""
    flag = 0
    t = ''
    for mail in thr:
        temp = ''
        sender = mail['From'].split('<')[0].strip()
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
            start_date = datetime.datetime.strptime(mail['Date'],'%a, %d %b %Y %H:%M:%S %z')
            if start_date > split_date:
                df_tst = df_tst.append({'body': str(temp),'replier':sender, 'thread_no':th_no, 'start_date':start_date}, ignore_index=True)
                tst_users.append(sender)
            else:
                trn_users.append(sender)
            t = temp
            flag = 1
            continue


        df = df.append({'body': str(t),'replier':sender, 'thread_no':th_no, 'start_date':start_date}, ignore_index=True)

        if start_date <= split_date:
            df_trn = df_trn.append({'body': str(t),'replier':sender, 'thread_no':th_no, 'start_date':start_date}, ignore_index=True)
            trn_users.append(sender)
            t = t + temp
        else:
            df_tst = df_tst.append({'body': str(temp),'replier':sender, 'thread_no':th_no, 'start_date':start_date}, ignore_index=True)
            tst_users.append(sender)

        
        #t = t + temp
    th_no += 1

print(cnt)
print(count_file)
print(len(df['body']))
print(len(df['thread_no'].unique()))
print(len(df['replier'].unique()))
rep_to_index = {}
index = 0
for rep in users:
    if rep_to_index.get(rep, 0) == 0:
        rep_to_index[rep] = index
        index += 1
pprint(rep_to_index)


for rep in df_trn['replier']:
    df_trn.loc[df_trn['replier']==rep,'int_replier'] = rep_to_index[rep]
#print(df_trn.head)    

for rep in df_tst['replier']:
    df_tst.loc[df_tst['replier']==rep,'int_replier'] = rep_to_index[rep]

for rep in df['replier']:
    df.loc[df['replier']==rep,'int_replier'] = rep_to_index[rep]
    
for rep in df['replier']:
    df.loc[df['replier']==rep,'int_replier'] = rep_to_index[rep]
    

#Offset the replier in test dataframe

df_tst['replier'] = df_tst.groupby('thread_no')['replier'].shift(-1)
df_tst['int_replier'] = df_tst.groupby('thread_no')['int_replier'].shift(-1)

df_tst.dropna(inplace=True)

df_trn.to_csv(file_name)
df_tst.to_csv(file_name1)
df.to_csv(file_name2)

# Aggregate according to date / make separate thread list |> P flag
# Split test_train_split 
#print(df.head)

# unique_users = len(df.replier.unique())


# # Indexing of words in vocab

# In[273]:


words = Counter()
for sent in df_trn.body.values:
    words.update(w.text.lower() for w in nlp(sent))
# print(words)

words = sorted(words, key=words.get, reverse=True)
print(words)
words = ['_PAD','_UNK'] + words

word2idx = {o:i for i,o in enumerate(words)}
idx2word = {i:o for i,o in enumerate(words)}

def indexer(s):
    vec = []
    for wr in nlp(s):
        wr = wr.text.lower()
        if wr in word2idx:
            vec.append(word2idx[wr])
        else:
            vec.append(word2idx['_PAD'])
    return vec



# # User Vector - construction

# In[274]:


np.set_printoptions(threshold = sys.maxsize)
user_indices = []
trn_user_indices = []
tst_user_indices = []

for u in users:
    user_indices.append(rep_to_index[u])

for v in trn_users:
    trn_user_indices.append(rep_to_index[v])

for w in tst_users:
    tst_user_indices.append(rep_to_index[w])



# In[275]:


user_vec_len = max(user_indices) + 1


# In[276]:

indexx=0
weight_list = []
for i in range(0, df_trn.thread_no.shape[0]+1):
    temp_index=indexx
    array  = np.zeros(user_vec_len)
    for j in range(temp_index, temp_index + list(df_trn.thread_no).count(i)):
        array[trn_user_indices[j]] += 1
        weight_list.append(list(array))
        indexx+=1

trn_weights = np.array(weight_list)


# In[169]:


indexx=0
weight_list = []
for i in range(0, df_tst.thread_no.shape[0]+1):
    temp_index=indexx
    array  = np.zeros(user_vec_len)
    for j in range(temp_index, temp_index + list(df_tst.thread_no).count(i)):
        array[tst_user_indices[j]] += 1
        weight_list.append(list(array))
        indexx+=1

tst_weights = np.array(weight_list)



# # Dataset Loading

# In[277]:


# embedding |> flag
class VectorizeData(Dataset):
    def __init__(self, df_path, maxlen=10, calc_maxlen = False):
        self.df = pd.read_csv(df_path, error_bad_lines=False)
        self.df['body'] = self.df.body.apply(lambda x: x.strip())
        print('Indexing...')
        self.df['bodyidx'] = self.df.body.apply(indexer)
        print('Calculating lengths')
        self.df['lengths'] = self.df.bodyidx.apply(len)
        if calc_maxlen == True:
            self.maxlen = max(self.df['lengths'])
        else:
            self.maxlen = maxlen
        print(self.maxlen)
        print('Padding')
        self.df['bodypadded'] = self.df.bodyidx.apply(self.pad_data)
        print(self.df)
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        X = self.df.bodypadded[idx]
        lens = self.df.lengths[idx]
        y = self.df.int_replier[idx]
        return X,y,lens
    
    def pad_data(self, s):
        padded = np.zeros((self.maxlen,), dtype=np.int64)
        if len(s) > self.maxlen: padded[:] = s[:self.maxlen]
        else: padded[:len(s)] = s
        return padded



# In[278]:


ds = VectorizeData(file_name2, calc_maxlen = True)
dtrain = VectorizeData(file_name, maxlen = ds.maxlen)
dtest = VectorizeData(file_name1, maxlen = ds.maxlen)


# # Pytorch Feedforward Neural Network model

# In[279]:


input_size = ds.maxlen
hidden_size = 50
num_classes = user_vec_len
num_epochs = 5
batch_size = 1
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[280]:


# concatenate user vector
# embeddings |>
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size,user_vec_len, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size + user_vec_len, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x,w):
#         x = torch.FloatTensor(x) 
        catt = torch.cat((x,w),1)
        out = self.fc1(catt)
        out = self.relu(out)       
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


# In[281]:
	

model = NeuralNet(input_size, hidden_size,user_vec_len, num_classes).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)  


# In[282]:


train_dl= DataLoader(dtrain, batch_size=1)
num_batch = len(train_dl)
for epoch in range(num_epochs):
    y_true_train = list()
    y_pred_train = list()
    total_loss_train = 0
    t = tqdm_notebook(iter(train_dl), leave=False, total=num_batch)
    for we, w in zip(t,trn_weights):
        X = we[0]
        y = we[1]
        lengths = we[2]
        
        w = w.reshape(-1,1)
        w = w.transpose()
        
        w = Variable(torch.Tensor(w).cpu())
        X = Variable(X.cpu())
        y = Variable(y.cpu())
        lengths = lengths.numpy()

        opt.zero_grad()
        X = X.float()
        w = w.float()
        y = y.long()
        pred = model(X,w)
        # F.nll_loss can be replaced with criterion
        loss = F.nll_loss(pred, y)
        loss.backward()
        opt.step()

        t.set_postfix(loss=loss.item())
        pred_idx = torch.max(pred, dim=1)[1]

        y_true_train += list(y.cpu().data.numpy())
        y_pred_train += list(pred_idx.cpu().data.numpy())
        total_loss_train += loss.item()
        

    train_acc = accuracy_score(y_true_train, y_pred_train)
    train_loss = total_loss_train/len(train_dl)
    print(f' Epoch {epoch}: Train loss: {train_loss} acc: {train_acc}')
torch.save(model.state_dict(),PATH)
# model_to_pickle = 
#Accuracy
# hyperparameter tuning
#Testing
# architecture testing


# # Testing - TODO

# In[247]:
test_dl= DataLoader(dtest, batch_size=1)
num_batches = len(test_dl)
y_true_test1 = list()
y_pred_test1 = list()
total_loss_train = 0
tt = tqdm_notebook(iter(test_dl), leave=False, total=num_batch)
for we, w in zip(tt,tst_weights):
    X = we[0]
    y = we[1]
    lengths = we[2]
    
    w = w.reshape(-1,1)
    w = w.transpose()

    w = Variable(torch.Tensor(w).cpu())
    X = Variable(X.cpu())
    y = Variable(y.cpu())
    lengths = lengths.numpy()

    #opt.zero_grad()
    X = X.float()
    w = w.float()
    y = y.long()
    pred = model(X,w)
    print(pred)
    # F.nll_loss can be replaced with criterion
    loss = F.nll_loss(pred, y)
    #loss.backward()
    #opt.step()

    #t.set_postfix(loss=loss.data[0])
    #pred_idx = torch.max(pred, dim=1)[1]

    y_true_train += list(y.cpu().data.numpy())
    y_pred_train += list(pred_idx.cpu().data.numpy())
    total_loss_train += loss.item()

train_acc = accuracy_score(y_true_train, y_pred_train)
train_loss = total_loss_train/len(train_dl)
print(f' Epoch {epoch}: Train loss: {train_loss} acc: {train_acc}')
#torch.save(model.state_dict(),PATH)
#Accuracy
# hyperparameter tuning
#Testing
# architecture testing
