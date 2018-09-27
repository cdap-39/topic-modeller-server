import re
import pandas as pd
import json

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings, sys
warnings.filterwarnings("ignore",category=DeprecationWarning)

import requests

# NLTK Stop words
from nltk.corpus import stopwords
if __name__ == '__main__':
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    # Import Dataset
    # df = pd.read_json('https://raw.githubusercontent.com/cdap-39/data/master/news.json')
    df = pd.read_json('input.json')

    # Convert to list
    data = df.content.values.tolist()

    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]

    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    data_words = list(sent_to_words(data))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    print(data_lemmatized[:1])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # Human readable format of corpus (term-frequency)
    [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

    import os
    os.environ.update({'MALLET_HOME': r'C:\\mallet-2.0.8\\mallet-2.0.8\\'})

    mallet_path = 'C:\\mallet-2.0.8\\mallet-2.0.8\\bin\\mallet'
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=70, id2word=id2word)


    # Compute Coherence Score
    coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    print('\nCoherence Score: ', coherence_ldamallet)

    optimal_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=70, id2word=id2word)

    from gensim import similarities
    index = similarities.MatrixSimilarity(optimal_model[corpus])

    query = "COLOMBO (News 1st) – Secretary of the Ministry of Disaster Management, Engineer Sisira Kumara states that owing to the prevailing weather over 350 houses have been damaged.He added that a sum of Rs.10,000 is to be paid as an advance for the affected families from today (September 25).He further added that measures have been taken to estimate the cost of the damage caused and compensation to be paid."

    vec_bow = id2word.doc2bow(query.lower().split())

    vec_lda = optimal_model[vec_bow]
    sims = index[vec_lda]

    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    top_ten = (sims[:5])

    def getSimilarArticles(query):
        vec_bow = id2word.doc2bow(query.lower().split())
        vec_lda = optimal_model[vec_bow]
        index = similarities.MatrixSimilarity(optimal_model[corpus])
        sims = index[vec_lda]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        top_ten = (sims[:5])
        return top_ten

    df['ref']=None
    for index, row in df.iterrows():
        df.at[index, 'ref']=[]

    for index, row in df.iterrows():
        print(index)
        similarArticles = getSimilarArticles(row['content'])
        i = 0
        matches = []
        while (i < len(similarArticles)):
            record_index = similarArticles[i][0]
            if record_index == index:
                i += 1
                continue
            if (similarArticles[i][1] > 0.90):
                source = ""
                link = df.iloc[record_index]['link']
                if "newsfirst" in link:
                    source = "newsfirst"
                elif "hirunews" in link:
                    source = "hirunews"
                match = {
                    "source": source,
                    "link": df.iloc[record_index]['link']
                }
                matches.append(match)
            i += 1

        ref = df.at[index, 'ref']
        ref.append(matches)
        df.at[index, 'ref'] = matches
    df.head()

    enriched=[]
    for index, row in df.iterrows():
        imageSrc = ''
        if 'image' in row:
            imageSrc = row['image']
        struct={
            "category": {
                "category": "",
                "pob": ""
            },
            "image": imageSrc,
            "media-link": "",
            "video-link": "",
            "heading": row['heading'],
            "link": row['link'],
            "content": row['content'],
            "ref": row['ref'],
            "media_ethics": {
                "violations": False,
                "reason": ""
            }
        }
        enriched.append(struct)
    # print(json.dumps(enriched))
    headers = {"Content-Type": "application/json"}
    r = requests.put('http://35.237.151.220:8081/api/processed_news', data=json.dumps(enriched), headers=headers)
    print(r)
