#!/usr/bin/env python
# coding: utf-8

# In[293]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.util import ngrams
from wordcloud import WordCloud
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import twint
import nest_asyncio
import datetime as dt
import seaborn as sns
from textblob import TextBlob
from nltk import pos_tag
from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


# In[51]:


nltk.download('punkt')
nltk.download('stopwords')


# In[19]:


tweets = pd.read_csv("tweets.csv", dtype = "string")


# In[205]:


tweets_data = tweets.head(500)


# In[206]:


tweets_data


# In[207]:


tweets_data = tweets_data.drop_duplicates(subset = ["text"])
tweets_data = tweets_data.dropna(subset = ["text"])


# In[33]:


def convert_to_text(dataframe):
    text = ""
    for i in dataframe['text']:
        text = text+i
    return text


# In[34]:


text = convert_to_text(tweets_data)


# In[66]:


def remove_links(text):
    result = re.sub(r'http\S+', '', text, flags=re.MULTILINE)
    return text


# In[67]:


text = remove_links(text)


# In[68]:


def remove_specialwords(text):
    filteredText=re.sub('[^A-Za-z0-9.]+', ' ',text)
    return filteredText


# In[69]:


text = remove_specialwords(text)


# In[43]:


def word_tokenizer(text):
    tokens = word_tokenize(text)
    return tokens


# In[44]:


tokens = word_tokenize(text)


# In[52]:


def remove_punct(token):
     return [word for word in token if word.isalpha()]


# In[53]:


tokens = remove_punct(tokens)


# In[56]:


def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w.lower() in stop_words]
    return filtered_tokens


# In[57]:


tokens = remove_stopwords(tokens)


# In[60]:


def stemming(tokens):
    ps = PorterStemmer()
    stemmed_tokens = [ps.stem(words_sent) for words_sent in tokens]
    return stemmed_tokens


# In[61]:


tokens = stemming(tokens)


# In[64]:


def lemmetization(tokens):
    lemmatizer = WordNetLemmatizer()
    lem_sent = [lemmatizer.lemmatize(words_sent) for words_sent in tokens]
    return lem_sent


# In[65]:


tokens = lemmetization(tokens)


# In[79]:


def bigrams(tokens):
    bi_grams = ngrams(tokens , 2)
    return list(bi_grams)


# In[81]:


def trigrams(tokens):
    tri_grams = ngrams(tokens , 3)
    return list(tri_grams)


# In[82]:


bi_grams = bigrams(tokens)


# In[83]:


tri_grams = trigrams(tokens)


# In[107]:


def bigrams_analysis(bi_grams):
    count_bigrams = FreqDist(bi_grams)
    dic_bigrams = dict(Counter(count_bigrams))
    return dic_bigrams  


# In[108]:


def trigrams_analysis(tri_grams):
    count_trigrams = FreqDist(tri_grams)
    return dic_trigrams


# In[111]:


bigrams_count = bigrams_analysis(bi_grams)
trigrams_count = trigrams_analysis(tri_grams)


# In[161]:


def bigrams_analysis(bigrams):
    n = len(bigrams)
    bigrams_list = list(bigrams.items())
    swapped = False
    for i in range(n-1):
        for j in range(0, n-i-1):
            if bigrams_list[j][1] < bigrams_list[j + 1][1]:
                swapped = True
                bigrams_list[j], bigrams_list[j+1] = bigrams_list[j+1] , bigrams_list[j]         
        if not swapped:
            return
    common_bigrams = []
    common_bigrams_count = []
    for i in range(10):
        common_bigrams.append(str(bigrams_list[i][0]))
        common_bigrams_count.append(bigrams_list[i][1])
    
    plt.bar(common_bigrams, common_bigrams_count, color = "blue" , width = 0.5)


# In[163]:


def trigrams_analysis(trigrams):
    n = len(trigrams)
    trigrams_list = list(trigrams.items())
    swapped = False
    for i in range(n-1):
        for j in range(0, n-i-1):
            if trigrams_list[j][1] < trigrams_list[j + 1][1]:
                swapped = True
                trigrams_list[j], trigrams_list[j+1] = trigrams_list[j+1] , trigrams_list[j]         
        if not swapped:
            return
    common_trigrams = []
    common_trigrams_count = []
    for i in range(10):
        common_trigrams.append(str(trigrams_list[i][0]))
        common_trigrams_count.append(trigrams_list[i][1])
    
    plt.bar(common_trigrams, common_trigrams_count, color = "blue" , width = 0.5)


# In[165]:


bigrams_analysis(bigrams_count)


# In[166]:


trigrams_analysis(trigrams_count)


# In[167]:


def word_cloud(final_text):
    final_words=" ".join(str(x) for x in final_text)
    wc = WordCloud(background_color="white",max_words=200,width=800, height=400).generate(final_words)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# In[168]:


word_cloud(text)


# In[171]:


get_ipython().system('pip3 install scipy')


# In[173]:


roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']


# In[195]:


sentiments_roberta = []
for i in tweets_data["text"]:
    tweet = i
    tweet = remove_links(tweet)
    tweet = remove_specialwords(tweet)
    encoded_tweet = tokenizer(tweet, return_tensors='pt')
    output = model(**encoded_tweet)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    negative = scores[0]
    positive = scores[1]
    neutral = scores[2]
    label = ""
    if negative > positive and negative > neutral:
        label = "negative"
    elif positive > negative and positive > neutral:
        label = "positive"
    elif neutral > positive and neutral > negative:
        label = "neutral"
    sentiments_roberta.append(label)
    


# In[196]:


sentiments_textblob = []
for i in tweets_data["text"]:
    tweet = i
    tweet = remove_links(tweet)
    tweet = remove_specialwords(tweet)
    analysis = TextBlob(tweet)
    
    if analysis.sentiment[0]>0:
        label = "positive"
    elif analysis.sentiment[0]<0:
        label = "negative"
    else:
        label = "neutral"
    sentiments_textblob.append(label)


# In[202]:


final_sentiments = []
for i in range(len(sentiments_roberta)):
    if sentiments_roberta[i] == sentiments_textblob[i]:
        final_sentiments.append(sentiments_textblob[i])
    else:
        final_sentiments.append(sentiments_roberta[i])
    


# In[208]:


tweets_data.insert(12, "Sentiment" , final_sentiments ,True)


# In[209]:


tweets_data


# In[211]:


count_words = len(tokens)
print("The total number of words are : " , count_words)


# In[212]:


def check_space(string):
    count = 0
    for i in range(0, len(string)):
        if string[i] == " ":
            count += 1
    return count


# In[213]:


count_spaces = check_space(text)


# In[214]:


print("The total number of spaces are : " , count_spaces)


# In[215]:


count_characters = len(text)
print("The total number of characters are : " , count_characters)


# In[219]:


def verb_finder(tokens):
    tokens_tagges = pos_tag(tokens)
    verb_count = 0
    for tagged_token in tokens_tagged:
        if tagged_token[1] == 'VB' or tagged_token[1] == 'VBD' or tagged_token[1] == 'VBG' or tagged_token[1] == 'VBN' or tagged_token[1] == 'VBP' or tagged_token[1] == 'VBZ':
            verb_count +=1
    return verb_count


# In[220]:


count_verbs = verb_finder(tokens)
print("The total number of verbs are : " , count_verbs)


# In[230]:


def starts_with_nameletter(tokens):
    count = 0
    for item in tokens:
        if item.startswith("A") or item.startswith("a"):
            count += 1
    return count


# In[232]:


count_A = starts_with_nameletter(tokens)
print("The total number of words starting with A/a are : ", count_A)


# In[234]:


sent_tokens = sent_tokenize(text)


# In[235]:


count_sentence_tokens = len(sent_tokens)
print("The total number of sentence tokens are : " , count_sentence_tokens)


# In[236]:


specialCharCount = 0
for char in text:
    if not (char.isalnum() or char.isspace()):
        specialCharCount += 1
print("The total number of special characters are : " , specialCharCount)


# In[237]:


data = [count_sentence_tokens, count_words, count_spaces, count_characters, count_verbs, count_A, specialCharCount]


# In[238]:


features_list = ["sentence length" , "word count" , "space count" , "char count" , "verb count" , "name count" , "special char count"]


# In[240]:


data_ = {
    "Features" : features_list,
    "Count" : data
}


# In[243]:


features = pd.DataFrame(data_)


# In[244]:


features


# In[246]:


tfIdf_vectorizer = TfidfVectorizer()


# In[248]:


tfIdf_vectorizer.fit(tweets_data["text"])


# In[249]:


tfIdf_vector = tfIdf_vectorizer.transform(tweets_data['text'])


# In[252]:


count_vectorizer = CountVectorizer()


# In[253]:


count_vectorizer.fit(tweets_data["text"])


# In[254]:


count_vector = count_vectorizer.transform(tweets_data["text"])


# In[255]:


print(count_vector.toarray())


# In[258]:


w2v_model = Word2Vec(tweets_data["text"])


# In[ ]:





# In[268]:


feature_extraction = tweets_data.copy()


# In[274]:


feature_extraction = feature_extraction.dropna()


# In[275]:


feature_extraction


# In[276]:


lbe = LabelEncoder()


# In[277]:


for i in feature_extraction:
    feature_extraction[i] = lbe.fit_transform(feature_extraction[i])


# In[279]:


feature_extraction = feature_extraction.drop("text" , axis = 1)


# In[281]:


corr_tweets = feature_extraction.corr()
dataplot = sns.heatmap(corr_tweets, cmap="YlGnBu", annot=True)


# In[290]:


model_data = tweets_data.copy()
model_data = model_data.dropna()
for i in model_data:
    model_data[i] = lbe.fit_transform(model_data[i])


# In[ ]:


model_1 = RandomForestClassifier(n_estimators=200)
model_1.fit(tfIdf_vector, model_data["Sentiment"])
predictions = model_1.predict(model_data["text"])


# In[ ]:


model_2 = RandomForestClassifier(n_estimators=200)
model_2.fit(count_vector, model_data["Sentiment"])
predictions = model_2.predict(model_data["text"])


# In[ ]:


model_3 = RandomForestClassifier(n_estimators=200)
model_2.fit(w2v_vector, model_data["Sentiment"])
predictions = model_2.predict(model_data["text"])


# In[ ]:


classifier = GaussianNB();
classifier.fit(tfIdf_vecor, model_data["Sentiment"])
y_pred = classifier.predict(model_data["text"])
cm = confusion_matrix(model_data["Sentiment"], y_pred)


# In[ ]:


classifier.fit(count_vecor, model_data["Sentiment"])
y_pred = classifier.predict(model_data["text"])
cm = confusion_matrix(model_data["Sentiment"], y_pred)


# In[ ]:


classifier.fit(w2v_vecor, model_data["Sentiment"])
y_pred = classifier.predict(model_data["text"])
cm = confusion_matrix(model_data["Sentiment"], y_pred)


# In[ ]:


# Import various layers needed for the architecture from keras
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Input, Dropout
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.callbacks import ModelCheckpoint
# The Input layer 
sequence_input = Input(shape=(30,), dtype='int32')
# Inputs passed to the embedding layer
embedding_sequences = embedding_layer(sequence_input)
# dropout and conv layer 
x = SpatialDropout1D(0.2)(embedding_sequences)
x = Conv1D(64, 5, activation='relu')(x)
# Passed on to the LSTM layer
x = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
# Passed on to activation layer to get final output
outputs = Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(sequence_input, outputs)


# In[ ]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
model.compile(optimizer=Adam(learning_rate=LR), loss='binary_crossentropy',metrics=['accuracy'])
ReduceLROnPlateau = ReduceLROnPlateau(factor=0.1,min_lr = 0.01, monitor = 'val_loss',verbose = 1)


# In[ ]:


training_tfidf = model.fit(tfIdf_vector, model_data["Sentiment"], batch_size=1024, epochs=10,
                    validation_data=(x_test, y_test), callbacks=[ReduceLROnPlateau])


# In[ ]:


training = model.fit(count_vector, model_data["Sentiment"], batch_size=1024, epochs=10,
                    validation_data=(x_test, y_test), callbacks=[ReduceLROnPlateau])


# In[ ]:


training = model.fit(w2v_vector, model_data["Sentiment"], batch_size=1024, epochs=10,
                    validation_data=(x_test, y_test), callbacks=[ReduceLROnPlateau])

