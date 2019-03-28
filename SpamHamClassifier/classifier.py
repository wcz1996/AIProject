import sys
import string
import math
import numpy as np

class NbClassifier(object):

    """
    A Naive Bayes classifier object has three parameters, all of which are populated during initialization:
    - a set of all possible attribute types
    - a dictionary of the probabilities P(Y), labels as keys and probabilities as values
    - a dictionary of the probabilities P(F|Y), with (feature, label) pairs as keys and probabilities as values
    """
    def __init__(self, training_filename, stopword_file):
        self.attribute_types = set()
        self.label_prior = {}    
        self.word_given_label = {}   

        self.collect_attribute_types(training_filename)
        if stopword_file is not None:
            self.remove_stopwords(stopword_file)
        self.train(training_filename)


    """
    A helper function to transform a string into a list of word strings.
    You should not need to modify this unless you want to improve your classifier in the extra credit portion.
    """
    def extract_words(self, text):
        no_punct_text = "".join([x for x in text.lower() if not x in string.punctuation])
        return [word for word in no_punct_text.split()]


    """
    Given a stopword_file, read in all stop words and remove them from self.attribute_types
    Implement this for extra credit.
    """
    def remove_stopwords(self, stopword_file):
        
        stopword = set()
        input = open(stopword_file, 'r', encoding = 'utf-8')
        for i in input.readlines():
            x = i.split()
            if x[0] not in stopword:
                stopword.add(x[0])
        self.attribute_types.difference(stopword)


    """
    Given a training datafile, add all features that appear at least m times to self.attribute_types
    """
    def collect_attribute_types(self, training_filename, m=1):
        self.attribute_types = set()
        vocabulary = []
        show_time = {}
        input = open(training_filename, 'r', encoding = 'utf-8')
        for i in input.readlines():
            x = i.split("\t")
            msg = self.extract_words(x[1])
            for j in msg:
                if j not in vocabulary:
                    vocabulary.append(j)
                    show_time[j] = 1
                else:
                    show_time[j] += 1
        for i in vocabulary:
            if show_time[i] >= m:
                self.attribute_types.add(i)



    """
    Given a training datafile, estimate the model probability parameters P(Y) and P(F|Y).
    Estimates should be smoothed using the smoothing parameter k.
    """
    def train(self, training_filename, k=0.1):
        self.label_prior = {}
        self.word_given_label = {}
        all_msg = []
        label_msg = []
        ham_num = 0
        spam_num = 0
        self.label_prior['ham'] = 0
        self.label_prior['spam'] = 0
        input = open(training_filename, 'r', encoding = 'utf-8')
        t = 0
        n = 0
        for i in self.attribute_types:
            self.word_given_label[('ham',i)] = 0;
            self.word_given_label[('spam',i)] = 0;

        for i in input.readlines():
            x = i.split("\t")
            #print(x)
            self.label_prior[x[0]] += 1
            
            msg = self.extract_words(x[1])
            for j in msg:
                if x[0] == 'ham':
                    ham_num += 1
                if x[0] == 'spam':
                    spam_num += 1
                self.word_given_label[(x[0],j)] += 1 
            t += 1

        self.label_prior['ham'] /= t
        self.label_prior['spam'] /= t 

        sum_ham = 0
        sum_spam = 0
        for i in self.attribute_types:
            self.word_given_label[('ham',i)] = (self.word_given_label[('ham',i)] + k) / (ham_num + k * len(self.attribute_types))
            self.word_given_label[('spam',i)] = (self.word_given_label[('spam',i)] + k) / (spam_num + k * len(self.attribute_types))
            sum_ham += self.word_given_label[('ham',i)] 
            sum_spam += self.word_given_label[('spam',i)] 

    """
    Given a piece of text, return a relative belief distribution over all possible labels.
    The return value should be a dictionary with labels as keys and relative beliefs as values.
    The probabilities need not be normalized and may be expressed as log probabilities. 
    """
    def predict(self, text):
        belief = {}
        belief['ham'] = math.log(self.label_prior['ham'])
        belief['spam'] = math.log(self.label_prior['spam'])
        for i in text:
            if ('ham',i) in self.word_given_label.keys() and ('spam',i) in self.word_given_label.keys(): 
                belief['ham'] += math.log(self.word_given_label[('ham',i)])
                belief['spam'] += math.log(self.word_given_label[('spam',i)])
            #if ('ham',i) not in self.word_given_label.keys() and ('spam',i) not in self.word_given_label.keys():
        return belief


    """
    Given a datafile, classify all lines using predict() and return the accuracy as the fraction classified correctly.
    """
    def evaluate(self, test_filename):
        input = open(test_filename, 'r')
        t = 0
        correct = 0
        for i in input.readlines():
            x = i.split("\t")
            text = self.extract_words(x[1])
            belief = self.predict(text)
            if belief['ham'] > belief['spam']:
                result = 'ham'
            else:
                result = 'spam'
            if result == x[0]:
                correct += 1
            t += 1
        correct /= t
        return correct


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("\nusage: ./hmm.py [training data file] [test or dev data file] [(optional) stopword file]")
        exit(0)
    elif len(sys.argv) == 3:
        classifier = NbClassifier(sys.argv[1], None)
    else:
        classifier = NbClassifier(sys.argv[1], sys.argv[3])
    print(classifier.evaluate(sys.argv[2]))