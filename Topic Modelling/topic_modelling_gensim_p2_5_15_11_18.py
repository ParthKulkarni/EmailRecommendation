import email, glob
import pandas as pd
from flanker import mime
import string


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
			if words[2] == "struct" :
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
	msg = remove_content_in_braces(msg)
	msg = remove_func_and_struct(msg)
	msg = remove_other_code_lines(msg)
	print(msg)
  
rt =''
fpath = "temp_data/*/*.email"
files = glob.glob(fpath)
for file in files :
  f = open(file, "r")
  msg = f.read()
  msg = remove_code(msg)
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
              rt += _ + "\n"
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
                      rt += _ + "\n"

tt = rt

import matplotlib.pyplot as plt
import gensim
import numpy as np
import spacy

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
import pyLDAvis.gensim
from gensim import models

import os, re, operator, warnings
warnings.filterwarnings('ignore')  # Let's not pay heed to them right now
# %matplotlib inline

import spacy
nlp = spacy.load("en")

my_stop_words = ['gmail','google','github','yahoo','com','org','http','https','golang','www','regards','thanks','html5','tidyhtml5']
for stopword in my_stop_words:
    lexeme = nlp.vocab[stopword]
    lexeme.is_stop = True

ttp1 = tt

doc = nlp(ttp1)
print(doc)

# we add some words to the stop word list
texts, article, skl_texts = [], [], []
for w in doc:
    # if it's not a stop word or punctuation mark, add it to our article!
    if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num:
        # we add the lematized version of the word
        article.append(w.lemma_)
    # if it's a new line, it means we're onto our next document
    if w.text == '\n':
        # print('----hi i am here----\n')
        skl_texts.append(' '.join(article))
        texts.append(article)
        article = []
3+
print(texts)

bigram = gensim.models.Phrases(texts)

texts = [bigram[line] for line in texts]

texts[1][0:10]

dictionary = Dictionary(texts)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)
corpus = tfidf[corpus]

corpus[1][0:10]

ldamodel = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)

ldamodel.show_topics()

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic :", (topic_idx))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(skl_texts)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(skl_texts)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 10

# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

no_top_words = 10
display_topics(nmf, tfidf_feature_names, no_top_words)
display_topics(lda, tf_feature_names, no_top_words)

tfidf_feature_names[2], tf_feature_names[2]

for line in tf:
    print (line)
    break

pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

def evaluate_graph(dictionary, corpus, texts, limit):
    """
    Function to display num_topics - LDA graph using c_v coherence
    
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    limit : topic limit
    
    Returns:
    -------
    lm_list : List of LDA topic models
    c_v : Coherence values corresponding to the LDA model with respective number of topics
    """
    c_v = []
    lm_list = []
    for num_topics in range(1, limit):
        lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        lm_list.append(lm)
        cm = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())
        
    # Show graph
    x = range(1, limit)
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("c_v"), loc='best')
    plt.show()
    
    return lm_list, c_v

"""%%time
lmlist, c_v = evaluate_graph(dictionary=dictionary, corpus=corpus, texts=texts, limit=10)
"""

def ret_top_model():
    """
    Since LDAmodel is a probabilistic model, it comes up different topics each time we run it. To control the
    quality of the topic model we produce, we can see what the interpretability of the best topic is and keep
    evaluating the topic model until this threshold is crossed. 
    
    Returns:
    -------
    lm: Final evaluated topic model
    top_topics: ranked topics in decreasing order. List of tuples
    """
    top_topics = [(0, 0)]
    while top_topics[0][1] < 0.97:
        lm = LdaModel(corpus=corpus, id2word=dictionary)
        coherence_values = {}
        for n, topic in lm.show_topics(num_topics=-1, formatted=False):
            topic = [word for word, _ in topic]
            cm = CoherenceModel(topics=[topic], texts=texts, dictionary=dictionary, window_size=10)
            coherence_values[n] = cm.get_coherence()
        top_topics = sorted(coherence_values.items(), key=operator.itemgetter(1), reverse=True)
    return lm, top_topics

lm, top_topics = ret_top_model()

print(top_topics[:5])

pprint([lm.show_topic(topicid) for topicid, c_v in top_topics[:10]])

ldatopics = [[word for word, prob in topic] for topicid, topic in ldamodel.show_topics(formatted=False)]

lda_coherence = CoherenceModel(topics=ldatopics, texts=texts, dictionary=dictionary, window_size=10).get_coherence()

def evaluate_bar_graph(coherences, indices):
    """
    Function to plot bar graph.
    
    coherences: list of coherence values
    indices: Indices to be used to mark bars. Length of this and coherences should be equal.
    """
    assert len(coherences) == len(indices)
    n = len(coherences)
    x = np.arange(n)
    plt.bar(x, coherences, width=0.2, tick_label=indices, align='center')
    plt.xlabel('Models')
    plt.ylabel('Coherence Value')

evaluate_bar_graph([lsi_coherence, hdp_coherence, lda_coherence],
                   ['LSI', 'HDP', 'LDA'])