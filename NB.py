#----------------------------------------------#
#-------| Written By: Muhammad Shaffay |-------#
#----------------------------------------------#

# A Naive Bayes algorithm ...

import re
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

def parse_string(string): 

    # Lowercase
    string = string.lower()

    # Split
    string = string.split()

    # Removing Any Tags & Punctuation
    for i, value in enumerate(string):

          value = re.sub(re.compile('<.*?>'), '', value) # removing tags

          value = re.sub('[^a-zA-Z]', '', value) # removing puntuations
          string[i] = re.sub('\s\s+', '', value) # removing spaces

    while ('' in string):
       string.remove('') # removing '' indexes

    return string

def defaulter(): return 1

class NaiveBayes:

    def __init__(self, classes):
        ''' Implements the Naive Bayes For Classification... '''
        self.classes = classes
        
        self.count_classes = 0
        self.class_words_count = 0
        
        self.vocab_size = 0
        self.vocab = 0
        
        self.DefaultDictionaries = 0
        self.classprobabilities = 0
       
    # Probability Counter
    def probability(self, count_word, count_class, vocab_size): 
        return (count_word + 1) / (count_class + vocab_size)
        
    def train(self, X, Y):

    # Unique CLasses
        self.count_classes = len(np.unique(Y))
        
    # Tokenizing    
        Xstrings = []
        for i, value in enumerate(X):
            Xstrings.append(parse_string(str(value[0])))
        
    # Grouping & Concating Strings
        Xgrouped_data = []
        for i , value in enumerate(self.classes):
            SingleClass = []
            for j , string in enumerate(Xstrings):
                if Y[j] == value:
                    for k in string:
                        SingleClass.append(k)

            Xgrouped_data.append(SingleClass)
   
   # Counting Each Class's Probability
        classprobabilities = []
        for i , value in enumerate(self.classes):    
            counter = 0
            for j , string in enumerate(Xstrings):
                if Y[j] == value:
                    counter += 1
            classprobabilities.append(counter/len(X))
        self.classprobabilities = classprobabilities

    
   # Counting Frequencies & |C|
        dictionaries , Cs = [] , []
        
        for i in range(self.count_classes):
            dictionaries.append(Counter(Xgrouped_data[i]))
            Cs.append(len(Xgrouped_data[i]))
            
        self.class_words_count = Cs   
        
   # Calculating |V|
        unique_words_corpus = []
    
        for i in range(self.count_classes):
            for j in list(dictionaries[i].keys()):
                if j not in unique_words_corpus:
                    unique_words_corpus.append(j)

        self.vocab = unique_words_corpus
        self.vocab_size = len(unique_words_corpus)
        
    # Default Dictionary
        DefaultDictionaries = []
        for i in range(self.count_classes):
            default = defaultdict(defaulter)
            for key , value in dictionaries[i].items():
                default[key] = self.probability(value, self.class_words_count[i], self.vocab_size)
            DefaultDictionaries.append(default)   
            
        self.DefaultDictionaries = DefaultDictionaries  
    
    def test(self, X):
        
        # Pre Processing
        processed = []
        for i , value in enumerate(X):
            processed.append(parse_string(value[0]))
        
        # Evaluating
        pclasses , probabilities = [] , []
        for i , text in enumerate(processed):
            class_probabilities = []
            
            for j , value in enumerate(self.classes):
                p = self.classprobabilities[j] # returning class probability
                d = self.DefaultDictionaries[j] # return that class' default dictionary

                for k , word in enumerate(text):
                    if word not in list(d.keys()):
                        p = p + np.log(self.probability(0, self.class_words_count[j], self.vocab_size))
                        self.DefaultDictionaries[j][word] = self.probability(0, self.class_words_count[j], self.vocab_size)
                    else:
                        p = p + np.log(d[word])
                    
                class_probabilities.append(p)

            indexes = np.argsort(class_probabilities)
            index = indexes[-1]
            
            pclasses.append(self.classes[index])
            probabilities.append(class_probabilities[index])            
            
        return np.array(pclasses) 
        
    def predict(self, x):
        
    # Evaluating
    
        class_probabilities = []
        for j , value in enumerate(self.classes):        

            p = self.classprobabilities[j] # returning class probability
            d = self.DefaultDictionaries[j] # return that class' default dictionary

            for k , word in enumerate(x):
                if word not in list(d.keys()):
                    p = p + np.log(self.probability(0, self.class_words_count[j], self.vocab_size))
                    self.DefaultDictionaries[j][word] = self.probability(0, self.class_words_count[j], self.vocab_size)
                else:
                    p = p + np.log(d[word])
                    
            class_probabilities.append(p)

        indexes = np.argsort(class_probabilities)
        index = indexes[-1]
            
        return str(self.classes[index])