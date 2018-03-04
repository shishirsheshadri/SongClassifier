# Reading the lyrics
# # Getting song titles
import os
from os import listdir
from os.path import isfile, join

file_list_song = os.listdir("C:\\Users\\shish_000\\Desktop\\web_scraping\\Training_Set")
song_dir_name = []
for song_file_name in file_list_song:
    exact_string = "C:\\Users\\shish_000\\Desktop\\web_scraping\\Training_Set\\" + str(song_file_name)
    print(exact_string)
    song_dir_name.append(exact_string)
songs_list = []
for song_name_exact in song_dir_name:
    songs_file = open(song_name_exact, "r")
    for line in songs_file:
        songs_list.append(line.replace("\n",""))
    songs_file.close()

len(songs_list)

# # Creating a master list of lyrics
import os
from os import listdir
from os.path import isfile, join
file_list = os.listdir("C:\\Users\\shish_000\\Desktop\\web_scraping\\Final_training_set")
song_dir = []
for song_file in file_list:
    exact_string = "C:\\Users\\shish_000\\Desktop\\web_scraping\\Final_training_set\\" + str(song_file)
    print(exact_string)
    song_dir.append(exact_string)
print(song_dir)

lyrics_list = []
for song_file_exact in song_dir:
    lyrics_file = open(song_file_exact, "r")
    for line in lyrics_file:
        lyrics_list.append(line.replace("\n",""))
    lyrics_file.close()
len(lyrics_list)


# # Tokenize the Document
def tokenize (document):
    return_list = document.split(" ")
    return ((return_list))
document_word_list = []
for document in lyrics_list:
    document_word_list.append(tokenize(document))
len(document_word_list)
from nltk.corpus import stopwords
stop = stopwords.words('english')
document_word_list = list(document_word_list)
corpus = []
for i in document_word_list:
    doc_list = []
    for word in i:
        if word not in stop:
            doc_list.append(word.replace("(","").replace(")","").replace("\"","").replace(",","").replace("in ","ing").replace(" ing ","in").replace("in'","ing").replace("[","").replace("]","").replace("!","").replace("?","").replace(":","").replace(".","") )
    corpus.append(doc_list)    
new_corpus = []
for x in corpus:
    new_corpus.append(list(filter(lambda a: a != '', x)))
corpus = new_corpus
corpus[1]

# # POS Tagging
pos_tagged = []
import nltk
for doc_pos in corpus:
    try:
        doc_pos.remove('')
    except:
        doc_pos = doc_pos
    try:    
        temp_list = nltk.pos_tag(doc_pos)
    except:
        continue
    pos_tagged.append(temp_list)

# # Converting to wordnet compatible stuff
from nltk.corpus import wordnet

def get_wordnet_pos(tup):
    treebank_tag = str(tup[1])
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return 'n'


check_list = [] 
new_tagged_corpus = []
for i in pos_tagged:
    single_doc = []
    for j in i:
        new_tag = (get_wordnet_pos(j))
        single_doc.append((j[0],new_tag))
    new_tagged_corpus.append(single_doc)

len(new_tagged_corpus)
new_tagged_corpus[0][0][0]
lemmatizer.lemmatize(new_tagged_corpus[0][0][0],new_tagged_corpus[0][0][1])


# # POS Tagged lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
test_list = []
lemmatized_corp = []
for document in new_tagged_corpus:
    lemmatized_doc_temp = []
    for word in document:
        lemmatized_doc_temp.append(lemmatizer.lemmatize(word[0],word[1]))
    lemmatized_corp.append((lemmatized_doc_temp))
print (len(lemmatized_corp))
lemmatized_corp[0]

## Vectorization
from gensim import corpora
dictionary = corpora.Dictionary(lemmatized_corp)
#Here we assigned a unique integer id to all words appearing in the corpus with the gensim.corpora.dictionary.Dictionary class. This sweeps across the texts, collecting word counts and relevant statistics. In the end, we see there are twelve distinct words in the processed corpus, which means each document will be represented by twelve numbers (ie., by a 12-D vector). To see the mapping between words and their ids:
print(dictionary.token2id)
from six import iteritems
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq < 5]
# ignore words that appear in less than 20 documents or more than 10% documents
dictionary.filter_extremes(no_below = 2)
print(dictionary)
print(dictionary.token2id)
dictionary.compactify()  # remove gaps in id sequence after words that were removed
print(dictionary.dfs)
print(dictionary.token2id)


#manual vectorization
accepted_list = [k for k, v in dictionary.token2id.items()]
accepted_list


## Cleaning the corpus 
filtered_lemmatized_corpus = []
for song in lemmatized_corp:
    new_song = []
    for word in song:
        if word in accepted_list:
            new_song.append(word)
    filtered_lemmatized_corpus.append(new_song)


#filtered_lemmatized_corpus

#creating a new dictionary using this filtered lemmatized corpus :)
dictionary = corpora.Dictionary(filtered_lemmatized_corpus)

#The function doc2bow() simply counts the number of occurrences of each distinct word, converts the word to its integer word id and returns the result as a sparse vector. The sparse vector [(0, 1), (1, 1)] therefore reads: in the document “Human computer interaction”, the words computer (id 0) and human (id 1) appear once; the other ten dictionary words appear (implicitly) zero times.
corpus_new = [dictionary.doc2bow(text) for text in filtered_lemmatized_corpus]
(corpus_new[0])

# # Trying TF-IDF

# In[53]:

from gensim import corpora, models, similarities
tfidf = models.TfidfModel(corpus_new) # step 1 -- initialize a model

corpus_tfidf =  tfidf[corpus_new]
for doc in corpus_tfidf:
    try:
        print(doc)
    except:
        print("error")

print(len(corpus_tfidf))
time_pass = corpus_tfidf[1]
time_pass
test_dict = dictionary.token2id
inv_map = {v: k for k, v in test_dict.items()}
test_dict
#https://radimrehurek.com/topic_modeling_tutorial/2%20-%20Topic%20Modeling.html

# ## LDA
mport gensim
lda_model = gensim.models.LdaModel(corpus_new, num_topics=10, id2word=dictionary, passes=4)
lda_model.print_topic(1, topn=10)
lda_model.print_topic(7, topn=10)
lda_model[corpus_new[6]]
doc_lda


topic_dist_list = []
for song_bow in corpus_new:
     topic_dist_list.append(lda_model[song_bow])
len(topic_dist_list)

all_topics = [0,1,2,3,4,5,6,7,8,9]
updated_topic_dist_list = []
for element in topic_dist_list:
    print(element)
    present_list =[]
    for tup in element:
        print(tup[0])
        present_list.append(tup[0])
        print(present_list)
        not_present = set(all_topics) - set(present_list)
        not_present_tuple = [(topic,0.0) for topic in not_present]
        print(not_present_tuple)
    element = list(set(element).union(set(not_present_tuple)))
    print(element)
    break

all_topics = [0,1,2,3,4,5,6,7,8,9]
updated_topic_dist_list = []
for element in topic_dist_list:
    present_list =[]
    for tup in element:
        present_list.append(tup[0])
        not_present = set(all_topics) - set(present_list)
        not_present_tuple = [(topic,0.0) for topic in not_present]
    element = list(set(element).union(set(not_present_tuple)))
    updated_topic_dist_list.append(element)
len(updated_topic_dist_list)
dict(updated_topic_dist_list[99])

# For each song, I have the topic breakdown. Each song has a total of 10 features. I can cluster them based on these features
print(updated_topic_dist_list[1]) #Topic dist
print(corpus_new[1]) #word id
print(filtered_lemmatized_corpus[1]) #Lyrics
counter = 0
updated_topic_dist_list_counter = []
for tup in updated_topic_dist_list:
    tup.append(("id",counter))
    updated_topic_dist_list_counter.append(tup)
    counter = counter+1
updated_topic_dist_list[1]
updated_topic_dist_list_counter[1]


# # Creating a dictionary from the tuple

import pandas as pd
updated_topic_dist_dict_list = []


for tup in updated_topic_dist_list_counter:
    updated_topic_dist_dict_list.append(dict(tup))
    

# # Panda Panda Panda

import pandas as pd
pre_clustering_data = pd.DataFrame(updated_topic_dist_dict_list)
pre_clustering_data
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
import numpy as np
pre_clustering_data_noid = pre_clustering_data.iloc[:,0:10]
pre_clustering_data_noid
plt.figure()
pre_clustering_data.iloc[:,0:2].plot()
plt.show()

#clustering
km = KMeans(n_clusters = 5)
myFit = km.fit(pre_clustering_data_noid)
myFit
labels = myFit.labels_
print(labels)
pre_clustering_data['clusters'] = labels
cluster_count = pre_clustering_data.groupby(['clusters'])['id'].count().tolist()
cluster_count
objects = ('Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5')
y_pos = np.arange(len(objects))
performance = cluster_count
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('count')
plt.title('Song Clusters')
plt.show()
plt.figure()
print(filtered_lemmatized_corpus[4])
print(filtered_lemmatized_corpus[6])
plt.figure()
pre_clustering_data

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.fit(pre_clustering_data_noid)
pca.components_
pca.explained_variance_ratio_ 
