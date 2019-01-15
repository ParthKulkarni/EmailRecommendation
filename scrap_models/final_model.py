#!/usr/bin/env python
# coding: utf-8

# In[1]:


import email, glob
import tensorflow as tf
import pandas as pd
from flanker import mime
import string
import re
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from pprint import pprint
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def remove_content_in_braces(msg) :
	msg1 = ''
	cnt = 0
	for char in msg :
		if char == '{' :
			cnt += 1
		elif char == '}' :
			cnt -= 1
		elif cnt == 0 :
			msg1 += char
		else :
			continue
	return msg1

def remove_func_and_struct(msg) :
	msg1 = ''
	take_line = True
	msg = msg.splitlines()
	for line in msg :
		take_line = True
		if line == '' :
			continue
		words = line.split(' ')
		if words[0] == "func" :
			take_line = False
		elif words[0] == "type" :
			if len(words)  >= 3  and words[2] == "struct" :
				take_line = False
		if take_line :
			msg1 += (line + '\n')
	return msg1

def remove_other_code_lines(msg) :
	msg1 = ''
	take_line = True
	msg = msg.splitlines()
	i = 0	
	while i < len(msg) :
		if (msg[i] == '') or ("//" in msg[i]) :
			i += 1
			continue
		take_line = True
		line = msg[i]
		if "package" in line :
			words = line.split(' ')
			if len(words) < 4 :
				take_line = False
		elif "import" in line :
			words = line.split(' ')
			if len(words) < 4 :
				take_line = False
				if "(" in line :
					while ')' not in msg[i] :
						i += 1
		elif "const" in line :
			words = line.split(' ')
			if len(words) < 4 :
				take_line = False
				if "(" in line :
					while ')' not in msg[i] :
						i += 1
		if take_line :
			msg1 += line + '\n'
		i += 1
	return msg1


def remove_code(msg) :
	msg = (remove_content_in_braces(msg))
	msg = (remove_func_and_struct(msg))
	msg = (remove_other_code_lines(msg))
	return msg

def get_header(msg):
    msg = email.message_from_string(msg)
    mfrom = msg['From'].split('<')[0]
    return mfrom
    
def flan(msg):
    rt = ''
    msg = mime.from_string(msg)
    if msg.content_type.is_singlepart():
      temp = str(msg.body)
      temp = temp.splitlines()
      for _ in temp:
          if _.startswith('>'):
              continue
          elif _.startswith('On'):
              continue
          else:
              rt+=_+"\n"
    else :
      for part in msg.parts :
          if "(text/plain)" in str(part) :
              temp = str(part.body)
              temp = temp.splitlines()
              for _ in temp :
                  if _.startswith('>') :
                      continue
                  if _.startswith('On'):
                      continue
                  else :
                      rt+=_+"\n"
    return rt


# In[3]:


rt =''
fpath = "/home/anuja/Desktop/BE project/Models/EmailRecommmendation/features/Dataset/*/*.email"
files = glob.glob(fpath)
for file in files :
  f = open(file, "r")
  msg = f.read()
  msg = mime.from_string(msg)
  if msg.content_type.is_singlepart():
      temp = str(msg.body)
      temp = temp.splitlines()
      for _ in temp:
          if _.startswith('>'):
              continue
          elif _.startswith('On'):
              continue
          else:
              rt+=_+"\n"
  else :
      for part in msg.parts :
          if "(text/plain)" in str(part) :
              temp = str(part.body)
              temp = temp.splitlines()
              for _ in temp :
                  if _.startswith('>') :
                      continue
                  if _.startswith('On'):
                      continue
                  else :
                      rt+=_+"\n"
               
rt = remove_code(rt)
# print(rt)
rt = rt.split('\n')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(rt)
total_words = len(tokenizer.word_index) + 1
max_words=total_words


# In[4]:


df = pd.DataFrame(columns=['body','replier', 'thread_no'])
users = []
fpath = "/home/anuja/Desktop/BE project/Models/EmailRecommmendation/features/Dataset/*"
folder = glob.glob(fpath)
th_no = 0

for fol in folder:
    files = glob.glob(fol+'/*.email')
    flag = 0
    t = ''
    for file in files:
        if flag==0:
            data = open(file,'r')
            temp = data.read()
            header = get_header(temp)
            temp = flan(temp)
            temp = remove_code(temp)
            t = temp
            flag = 1
            users.append(header)
            
            continue
        data = open(file,'r')
        temp = data.read()
        header = get_header(temp)
        users.append(header)
        temp = flan(temp)
        temp = remove_code(temp)
        df = df.append({'body': t,'replier':header, 'thread_no':th_no}, ignore_index=True)
        t = t + temp
    th_no += 1


# In[5]:


print(df.head)


# In[6]:


h = df.replier
pprint(len(h))
h=list(h)

thread_no_list = df.thread_no
thread_no_list = list(thread_no_list)
print(thread_no_list)


# In[7]:


np.set_printoptions(threshold = sys.maxsize)
val = np.array(users)
w = open('one_hot.txt','w')

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(val)
print(integer_encoded)
user_indices = integer_encoded

one_hot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
output_shape = one_hot_encoded.shape[1]
w.write(str(one_hot_encoded))
w.close()


# In[8]:


print(user_indices)
input_weights = []
user_size = max(user_indices) + 1
print(user_size)


# In[9]:


onehot_encoded_final = []
for replier in h:
    idx1 = label_encoder.transform([replier])
    onehot = np.zeros(user_size)
    onehot[idx1] = 1
    onehot_encoded_final.append(list(onehot))
one_hot_encoded_final = np.array(onehot_encoded_final)
print(type(one_hot_encoded_final))


# In[10]:


#Binary encoding of users participating in each thread

index=0
weight_list = []
for i in range(0, max(thread_no_list)+1):
#     temp = np.zeros(user_size)
    temp_index=index
#     print(temp)
    array  = np.zeros(user_size)
    for j in range(temp_index, temp_index + thread_no_list.count(i)):
#         print(user_indices[j], end = ",")
        array[user_indices[j]] += 1
        weight_list.append(list(array))
#         temp[user_indices[j]] += 1
        index+=1
#     print("\n")
#     print(temp,"\n")

weights = np.array(weight_list)
pprint(weights.shape) 


# In[11]:


x_train = df.body
# print(x_train)


# In[12]:


def longest(l):
    m=0
    for k in l:
        m = max(len(k),m)
    return m
max_len = longest(x_train)


# In[13]:


# max_words = 1294
# max_len = 3267
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(x_train)
sequences = tok.texts_to_sequences(x_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
print(type(sequences_matrix))
print(len(sequences_matrix))


# In[14]:


b=np.zeros(user_size)
a=tf.convert_to_tensor(b)
#dense_cat = Dense(256, activation='relu')(a)
#flat1 = Dense(32, activation='relu')(dense_cat)


# In[18]:


from keras.layers.merge import concatenate
def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    inputs2 = Input(name='inputs2',shape=[user_size])
    layer2=Dense(256,name='FC2')(inputs2)
    
    merge=concatenate([layer,layer2])
    
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(output_shape,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=[inputs,inputs2],outputs=layer)
    return model


# In[19]:


model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])


# In[20]:


model.fit([sequences_matrix,weights],one_hot_encoded_final,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])


# In[ ]:





# In[ ]:




