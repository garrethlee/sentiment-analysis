import re       
import pickle
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class SentimentDetectorPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        """Initializes a sklearn transformer that'll preprocess textual data"""
        self.lemmatizer = WordNetLemmatizer()
         

    def clean(self, t):
        
        """Replaces non-speech features in tweets with regex"""

        URL_PATTERN = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
        HASHTAG_PATTERN = r"#\S+[a-zA-Z]"
        USERNAME_PATTERN = r"@[a-zA-Z]\S*"
        NUMBER_PATTERN = r"\d+"
        APOSTROPHE_PATTERN = r"\w+'\w+"
        NONWORD_PATTERN = r"[^a-zA-Z]+"

        t = re.sub(HASHTAG_PATTERN, "HASHTAG", t)
        t = re.sub(URL_PATTERN, "URL", t)
        t = re.sub(USERNAME_PATTERN, "USER", t)
        t = re.sub(APOSTROPHE_PATTERN, "", t)
        t = re.sub(NUMBER_PATTERN, "NUMBER", t)
        t = re.sub(NONWORD_PATTERN, " ", t)

        return t.lower()
        
    def remove_stopwords(self, tokens):
        """Removes stopwords from a given sentence"""        

        stopwords = ['i',
            'me',
            'my',
            'myself',
            'we',
            'our',
            'ours',
            'ourselves',
            'you',
            "you're",
            "you've",
            "you'll",
            "you'd",
            'your',
            'yours',
            'yourself',
            'yourselves',
            'he',
            'him',
            'his',
            'himself',
            'she',
            "she's",
            'her',
            'hers',
            'herself',
            'it',
            "it's",
            'its',
            'itself',
            'they',
            'them',
            'their',
            'theirs',
            'themselves',
            'what',
            'which',
            'who',
            'whom',
            'this',
            'that',
            "that'll",
            'these',
            'those',
            'am',
            'is',
            'are',
            'was',
            'were',
            'be',
            'been',
            'being',
            'have',
            'has',
            'had',
            'having',
            'do',
            'does',
            'did',
            'doing',
            'a',
            'an',
            'the',
            'and',
            'but',
            'if',
            'or',
            'because',
            'as',
            'until',
            'while',
            'of',
            'at',
            'by',
            'for',
            'with',
            'about',
            'against',
            'between',
            'into',
            'through',
            'during',
            'before',
            'after',
            'above',
            'below',
            'to',
            'from',
            'up',
            'down',
            'in',
            'out',
            'on',
            'off',
            'over',
            'under',
            'again',
            'further',
            'then',
            'once',
            'here',
            'there',
            'when',
            'where',
            'why',
            'how',
            'all',
            'any',
            'both',
            'each',
            'few',
            'more',
            'most',
            'other',
            'some',
            'such',
            'no',
            'nor',
            'only',
            'own',
            'same',
            'so',
            'than',
            'too',
            'very',
            's',
            't',
            'can',
            'will',
            'just',
            'don',
            "don't",
            'should',
            "should've",
            'now',
            'd',
            'll',
            'm',
            'o',
            're',
            've',
            'y',
            'ain',
            'aren',
            "aren't",
            'couldn',
            "couldn't",
            'didn',
            "didn't",
            'doesn',
            "doesn't",
            'hadn',
            "hadn't",
            'hasn',
            "hasn't",
            'haven',
            "haven't",
            'isn',
            "isn't",
            'ma',
            'mightn',
            "mightn't",
            'mustn',
            "mustn't",
            'needn',
            "needn't",
            'shan',
            "shan't",
            'shouldn',
            "shouldn't",
            'wasn',
            "wasn't",
            'weren',
            "weren't",
            'won',
            "won't",
            'wouldn',
            "wouldn't"]
        
        # remove apostrophes from stopwords
        final_stop_words = list(map(lambda x: re.sub(r'\W+', '', x), stopwords))

        return (" ".join([word for word in tokens if word not in final_stop_words])).lower()


    def lemmatize_data(self, text):
        return [self.lemmatizer.lemmatize(tok) for tok in text.split()]

    
    def fit(self, X, y = None):
        return self
        
    def transform(self, X, y = None):
        if type(X) == pd.DataFrame:
            X = X['tweet'] 
        if type(X) != pd.Series:
            X = pd.Series(X)
        
        data = X.apply(self.clean)
        data = data.apply(self.lemmatize_data)
        data = data.apply(self.remove_stopwords)
        return data
    
    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.transform(X)
        
        