"""
This module implements the main functionality  for data modelling using techniques of Topic Modelling, Mood Prediction and Song Similarity
Author: Cristóbal Veas
Linkedln: https://www.linkedin.com/in/cristobal-veas/
"""

__author__ = "Cristóbal Veas"
__email__ = "cristobal.veas.ch@gmail.com"
__status__ = "planning"


from collections import defaultdict
import numpy as np 
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as SWE
from spacy.lang.es.stop_words import STOP_WORDS as SWS
from spacy.lang.fr.stop_words import STOP_WORDS as SWF
from spacy.lang.de.stop_words import STOP_WORDS as SWD
import string
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import joblib
import pkg_resources
from langdetect import detect


class SpotiSciencePredicter():

    """
    Class for data modelling.
    Attributes
    ----------
    STOPWORDS: list - list of english stop words (words which are filtered out before or after processing of natural language text)
    NLP: object - natural language processing object used in spacy library
    PUNTUACTION: list - list of  text symbols 
    PATHMODEL: str  path for multi class classification model weights  using a random forest classifier
    """

    def __init__(self):
        self.languages = {
            'en' : 'english',
            'fr' : 'french',
            'es': 'spanish',
            'de': 'german'
        }
        self.STOPWORDSENGLISH = list(SWE)
        self.STOPWORDSSPANISH = list(SWS)
        self.STOPWORDSFRENCH = list(SWF)
        self.STOPWORDSGERMAN = list(SWD)

        self.NLPENGLISH = spacy.load('en_core_web_lg')
        self.NLPSPANISH = spacy.load('es_core_news_lg')
        self.NLPFRENCH = spacy.load('fr_core_news_lg')
        self.NLPGERMAN = spacy.load('de_core_news_lg')
        self.PUNTUACTION = string.punctuation
        self.PATHMODEL = pkg_resources.resource_filename(__name__,"weights/mood.joblib")

    #----------------
    # TOPIC MODELLING
    #----------------
    def __inner__lyric_to_list(self,lyric):
        """
        Return a list Transform, clean and split sentences returning a list of  lyric's sentences
        Attributes
        ----------
        lyric: str - the lyric of the song
        """
        lyric = lyric.replace("EmbedShare Url:CopyEmbed:Copy","")
        lyric_list = lyric.split("\n")
        lyric_list = [sentence for sentence in lyric_list if len(sentence)>1]
        return lyric_list

    def __inner__spacy_tokenizer(self,lyric,lang):
        """
        Return a list with tokenize sentence of lyric list
        Attributes
        ----------
        lyric: str - the lyric of the song
        lang: the lenguage model used for tokenize (available are spanish, english, french, german)
        """
        if "english" in lang:
            mytokens = self.NLPENGLISH(lyric)
            mytokens = [ word.lemma_.lower() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
            # suppression pronoms
            mytokens = [ word for word in mytokens if word not in self.STOPWORDSENGLISH and word not in self.PUNTUACTION] 
        if "spanish" in lang:
            mytokens = self.NLPSPANISH(lyric)
            mytokens = [ word.lemma_.lower() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
            mytokens = [ word for word in mytokens if word not in self.STOPWORDSSPANISH and word not in self.PUNTUACTION] 
        if "french" in  lang:
            mytokens = self.NLPFRENCH(lyric)
            mytokens = [ word.lemma_.lower() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
            mytokens = [ word for word in mytokens if word not in self.STOPWORDSFRENCH and word not in self.PUNTUACTION] 
        if "german" in  lang:
            mytokens = self.NLPGERMAN(lyric)
            mytokens = [ word.lemma_.lower() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
            mytokens = [ word for word in mytokens if word not in self.STOPWORDSGERMAN and word not in self.PUNTUACTION] 
        mytokens = " ".join([i for i in mytokens])
        return mytokens

    def __inner__lyric_vectorizer(self,lyrics,stop_words,n_grams):
        """
        vectorize the lyrics returning the vectorizer object and a list with the vectorized lyrics
        Attributes
        ----------
        lyrics: list - a list with sentences of lyric.
        stop_words: str - the language used for vectorizer stop words 
        n_grams: tuple - the number of n-grams used for vectorizer  model  (1gram: (1,1) ,  1 or 2 gram (1,2))

        see more info about stop_words and n_grams parameters in sklearn documentation
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
        """
        #vectorizer = CountVectorizer(input = 'content',
        #    min_df= 1, # issue : int
        #    max_df=1.0, 
        #    stop_words=stop_words, 
        #    token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}',
        #    ngram_range=n_grams
        #    )
        vectorizer = CountVectorizer(token_pattern="[a-zA-Z\-][a-zA-Z\-]{2,}")
        lyrics_vectorized = vectorizer.fit_transform(lyrics)
        return vectorizer,lyrics_vectorized

    def __train__model(self,lyrics_vectorized,modelname,num_topics):
        """
        train the topic modelling model
        Attributes
        ----------
        lyrics_vectorized: list -  the song lyrics vectorized 
        modelname: str - the name of the model used for Topic MOdelling (available options are "lda","nmf" and "lsi")
        num_topics: int - the number of topics for the model 
        """
    
        if "lda" in modelname:
            # Latent Dirichlet Allocation Model
            model = LatentDirichletAllocation(n_components=num_topics, max_iter=10, learning_method='online',verbose=False)
        if "nmf" in modelname:
        # Non-Negative Matrix Factorization Model
            model = NMF(n_components=num_topics)
        if "lsi" in modelname:
            # Latent Semantic Indexing Model using Truncated SVD
            model = TruncatedSVD(n_components=num_topics)
        model.fit(lyrics_vectorized)
        return model 

    def __inner__selected_topics(self,model, vectorizer, top_n):
        """
        Predict the topics using vectorizer model and topic modelling model 
        Attributes
        ----------
        model: object - the trained topic modelling model
        vectorizer: object -  the trained vectorizer model 
        top_n: int - the max number of words per topic
        """
        RESULTS = defaultdict(list)
        for idx, topic in enumerate(model.components_):
            n_topic = "Topic %d:" % (idx)
            topics = [(vectorizer.get_feature_names()[i], topic[i])
                            for i in topic.argsort()[:-top_n - 1:-1]]
            
            RESULTS[n_topic].append(topics)
        return RESULTS

    def predict_topic_lyric(self,lyric,model='lda',lang='english',stopwords=None,n_grams=(1,1),n_topics=10,top_n=10):
        """
        main method used to predict the topics of lyrics 
        Attributes
        ----------
        lyric: str - the lyrics of the song
        model: str - the name of the model used for Topic MOdelling (available options are "lda","nmf" and "lsi")
        lang: the language of the lyrics (available options are "spanish" and "english")
        stop_words: str - the language used for vectorizer stop words (default is None) 
        n_grams: tuple - the number of n-grams used for vectorizer  model  (1gram: (1,1) ,  1 or 2 gram (1,2)) (default is (1,1))
        n_topics: int - the number of topics for the model  (default is 10)
        top_n: int - the max number of words per topic

        to see more info about models and vectorizer attributes
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
        https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
        """
        
        lyric_list = self.__inner__lyric_to_list(lyric)
        processed_lyric = [self.__inner__spacy_tokenizer(lyric=sentence,lang=lang) for sentence in lyric_list]
        vectorizer, data_vectorized = self.__inner__lyric_vectorizer(processed_lyric,stop_words=stopwords,n_grams=n_grams)
        trained_model = self.__train__model(data_vectorized,modelname=model,num_topics=n_topics)
        topics = self.__inner__selected_topics(trained_model,vectorizer,top_n=top_n)
        return topics

    #---------------
    #Mood Prediction
    #---------------

    def predict_song_mood(self,song):
        """
        main method used to predict the mood of a song 
        Attributes
        ----------
        song: dict - the data of the song 
        """
        spotify_mood_indicators = ['length',
            'acousticness',
            'danceability',
            'energy',
            'instrumentalness',
            'liveness',
            'valence',
            'loudness',
            'speechiness',
            'tempo',
            'key',
            'time_signature']
        
        predictmodel = joblib.load(self.PATHMODEL)
        moods = {0:'calm',1:'energy',2:'happy',3:'sad'}

        indicators = np.array([song[i] for i in spotify_mood_indicators])
        song_features = indicators.reshape(-1,1).T
        
        preds = predictmodel.predict(song_features)
        return moods[preds[0]]
        
    #---------------
    #Song Similarity
    #---------------
    def predict_similar_songs(self,object,target,distance='l2',n_features=6,top_n=10):
        """
        main method used to calculate most similar songs
        Attributes
        ----------
        object: str - the id or copy link of the song to analyze
        target: str - the id or copy of the object to search for similar songs (availabel objects are: song or album or playlist)
        distance: str - the type of distance used to predict similar words (available type are l1 or l2, default is l2)
        n_features: int - number of features of song used to predict similarity (default is 6)
                         - model is training with the following features [acousticness, length,danceability, energy, instrumentalness,liveness,valence,loudness,speechiness,tempo,key,time_signature]
        top_n: int - number of most similar songs returns as result (default is 10)

        for more information about features go to spotify song features
        https://developer.spotify.com/documentation/web-api/reference/#category-tracks
        """
        similarity = []
        similar_songs = defaultdict(list)

        object_feature = list(object.values())[n_features:]
        object_name = list(object.values())[1]

        for key in target.keys():
            for song in target[key]:
                target_feature = list(song.values())[n_features:]
                target_name = list(song.values())[1]
                target_artist = list(song.values())[2]

                if object_name not in target_name:
                    object_feature = np.array([float(i)/sum(object_feature) for i in object_feature])
                    target_feature = np.array([float(i)/sum(target_feature) for i in target_feature])
                    if "l2" in distance:
                        n_similarity = np.linalg.norm(object_feature-target_feature)
                    if "l1" in distance: 
                        n_similarity = np.linalg.norm(object_feature-target_feature,ord=1)
                    results = (target_name,target_artist,n_similarity)
                    similarity.append(results)
                    similarity = sorted(similarity ,key =lambda tup:tup[2])
                    similarity = [tup for tup in similarity if tup[2]>0]

            similar_songs[object_name] = similarity[0:top_n]
        return similar_songs
    def __get_token(self,lang,words):
        if "english" in lang:
            mytokens = self.NLPENGLISH(words)
        if "spanish" in lang:
            mytokens = self.NLPSPANISH(words)
        if "french" in  lang:
            mytokens = self.NLPFRENCH(words)
        if "german" in  lang:
            mytokens = self.NLPGERMAN(words)
        return mytokens
    def calculate_topics_similarities(self,song1,song2):
        topics1 = " ".join(song1['topics'])
        coef1 = song1["topics_coeff"]
        coef2 = song2["topics_coeff"]
        topics2 = " ".join(song2['topics'])
        tokens1 = self.__get_token(song1['lang'],topics1)
        tokens2 = self.__get_token(song2['lang'],topics2)
        print(tokens1,"/",tokens2)
        similarity = 0
        # we calculate the weighted sum of the similarity of each topics
        for i in range(len(tokens1)):
            print(tokens1[i])
            similarity_i = 0
            for j in range(len(tokens2)):
                similarity_i += coef2[j]*tokens1[i].similarity(tokens2[j])
            similarity = similarity + coef1[i] * similarity_i/sum(coef2)
        return similarity/sum(coef1)
    def calculate_genres_links(self, genres, song):
        scores = {}
        for genre in genres:
            lang_genre = "english" #self.languages[detect(genre)]
            token = self.__get_token(lang_genre,genre)
            similarity = 0
            if len(song['genre']) == 0:
                scores[genre] = None
            else:
                try:
                    for g in song['genre']:
                        lang_g = "english"#self.languages[detect(g)]
                        tk = self.__get_token(lang_g,g)
                        s = token.similarity(tk)
                        if s > similarity: # on prend le score qui match le mieux
                            # the mean would dilute too much the similarity or boost it too much
                            similarity = s
                except TypeError as e:
                    print(song)
                scores[genre]=similarity
        return scores
    
    def calculate_to_main_topics_similarities(self, main, song):
        try:
            if song['has_lyrics']:
                topics = song['topics']
                coef = song['topics_coeff']
                tokens = self.__get_token(song['lang']," ".join(topics))
                main_token = self.__get_token('english', main)
                similarity = 0
                #print(len(tokens),len(main_token),len(coef))
                for i  in range(len(tokens)):
                    t = tokens[i]
                    c = coef[i]
                    similarity += main_token.similarity(t)*c
                return similarity/sum(coef)
            else:
                return 0
        except TypeError as e:
            print(e)
            print(song['has_lyrics'],print(song['lyrics']))
            print(song['lang'])

    
