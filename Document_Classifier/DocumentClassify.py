#importing required packages
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus.reader import CategorizedPlaintextCorpusReader
import nltk
import random
import os
#scikit-learn algorithms with NLTK
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
import pickle

class DocumentClassify:
    documents = []
    data = []

    def __init__(self):
        self.accuracy_ = 0
        self.classifier_ = None

    def load_data(self, path):
        # using NLTK's CategorizedPlaintextCorpusReader to load corpus for category classification
        reader = CategorizedPlaintextCorpusReader(path + "/Corpus", r'.*\.txt', cat_pattern=r'(\w+)/*')
        print("Categories: " + str(reader.categories()))

        docs = [(list(reader.words(fileid)), category) for category in reader.categories() for fileid in
                reader.fileids(category)]
        # shuffling the documents for better training data
        random.shuffle(docs)
        documents = docs

    def cleanse_data(self):
        all_words = []

        for w in reader.words():
            all_words.append(w.lower())

        # frequency distribution of the words
        all_words = nltk.FreqDist(all_words)
        # load stop words for english language
        stop_words = set(stopwords.words('english'))

        # remove all stop words
        clean_data = [row for row in all_words.keys() if row not in stop_words]
        # remove all special characters and numeric data
        clean_data = [row for row in clean_data if row.isalpha()]
        # remove single letters as thet doesn't add much value
        clean_data = [row for row in clean_data if len(row) > 1]
        data = clean_data

    def find_features(doc):
        words = set(doc)
        features = {}
        for w in data:
            features[w] = (w in words)
        return features

    def train_test_split(self):
        train_test_data = [(find_features(doc), category) for (doc, category) in documents]
        # lets divide this data into train and test data
        train_data = train_test_data[:int(len(train_test_data) * 0.7)]
        test_data = train_test_data[int(len(train_test_data) * 0.7):]
        return train_data, test_data

        # will start with Naive Bayes Classifier of NLTK

    def nltk_naivebayes(self):
        train_data, test_data = self.train_test_split()
        NBClf = nltk.NaiveBayesClassifier.train(train_data)
        self.classifier_ = NBClf
        self.accuracy_ = nltk.classify.accuracy(NBClf, test_data) * 100

    def skl_multinomialnb(self):
        train_data, test_data = self.train_test_split()
        MNBClf = SklearnClassifier(MultinomialNB())
        self.classifier_ = MNBClf
        MNBClf.train(train_data)
        self.accuracy_ = nltk.classify.accuracy(MNBClf, test_data) * 100

    def skl_bernoullinb(self):
        train_data, test_data = self.train_test_split()
        BerClf = SklearnClassifier(BernoulliNB())
        self.classifier_ = BerClf
        BerClf.train(train_data)
        self.accuracy_ = nltk.classify.accuracy(BerClf, test_data) * 100

    def skl_logitreg(self):
        train_data, test_data = self.train_test_split()
        logClf = SklearnClassifier(LogisticRegression())
        self.classifier_ = logClf
        logClf.train(train_data)
        self.accuracy_ = nltk.classify.accuracy(logClf, test_data) * 100

    def skl_sgdclf(self):
        train_data, test_data = self.train_test_split()
        sgdClf = SklearnClassifier(SGDClassifier())
        self.classifier_ = sgdClf
        sgdClf.train(train_data)
        self.accuracy_ = nltk.classify.accuracy(sgdClf, test_data) * 100

    def skl_svc(self):
        train_data, test_data = self.train_test_split()
        svcClf = SklearnClassifier(SVC())
        self.classifier_ = svcClf
        svcClf.train(train_data)
        self.accuracy_ = nltk.classify.accuracy(svcClf, test_data) * 100

    def skl_linearsvc(self):
        train_data, test_data = self.train_test_split()
        linSVC = SklearnClassifier(LinearSVC())
        self.classifier_ = linSVC
        linSVC.train(train_data)
        self.accuracy_ = nltk.classify.accuracy(linSVC, test_data) * 100

    def skl_nusvc(self, *args, **kwargs):
        nu = kwargs.get('nu', None)
        train_data, test_data = self.train_test_split()
        nusvc = SklearnClassifier(NuSVC(nu))
        self.classifier_ = nusvc
        nusvc.train(train_data)
        self.accuracy_ = nltk.classify.accuracy(nusvc, test_data) * 100

''' saving clssifier for future use

    def save_classifier(self, classifier, file_name):
        pickle_file = open(file_name, "wb")
        pickle.dump(classifier, pickle_file)
        print("Clssifier saved successfully")

    def load_saved_classifier(self, file_name):
        pickle_file = open(file_name, "rb")
        classifier = pickle.load(file_name)
        return classifier
'''