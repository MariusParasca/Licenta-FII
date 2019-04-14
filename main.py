from nltk import word_tokenize, PorterStemmer, ngrams, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet as swn
from nltk.corpus.reader.wordnet import WordNetError
# from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from pymetamap import MetaMap
import numpy as np
import re
import string

# transformare semtypes si c
# ~
# count vectorizer


class Preprocesor:
    ADJ = 'JJ'
    NOUN = 'NN'
    VERB = 'VB'
    ADVERB = 'RB'

    def __init__(self, metamap_path='/home/noway/Facultate/Licenta/public_mm/bin/metamap18'):
        self.mm = MetaMap.get_instance(metamap_path)
        self.x = []
        self.y = []
        self.corpus = []
        self.umls_semtypes_cuis = []
        self.sem_abbreviation_translations = []

    @staticmethod
    def metamap_pos_to_sentiwordnet_pos(pos):
        if Preprocesor.NOUN in pos:
            return "n"
        elif Preprocesor.VERB in pos:
            return "v"
        elif Preprocesor.ADJ in pos:
            return "a"
        elif Preprocesor.ADVERB in pos:
            return "r"
        else:
            return ""

    def ntlk_pos(self, s, string_tokenezed=True, to_stem=True, to_string=False):
        if to_stem:
            s = self.get_nltk_porter_stemming(word_tokenize(s), to_string=to_string)
        elif string_tokenezed:
            pass
        else:
            s = word_tokenize(s.lower())
        return pos_tag(s)

    @staticmethod
    def get_nltk_porter_stemming(tokens, to_string=True):
        """Porter stemming using ntlk word tokenizer"""
        porter_stemmer = PorterStemmer()
        stems = []
        for item in tokens:
            stems.append(porter_stemmer.stem(item))
        if to_string:
            return " ".join(stems)
        else:
            return stems

    @staticmethod
    def get_porter_stemming(text):
        """Porter stemming using my string tokenizing method"""
        tokens = Preprocesor.get_string_tokenizing(text)
        porter = PorterStemmer()
        porter_stemming = [porter.stem(t) for t in tokens]
        return porter_stemming

    @staticmethod
    def get_string_tokenizing(text):
        """ My string tokenizing method"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return [token for token in text.split(" ") if token != ""]

    @staticmethod
    def get_n_grams_words(tokenized_string, n):
        """"""
        return list(ngrams(tokenized_string, n))
    
    @staticmethod
    def tokenize_for_tfidf(text):
        tokens = word_tokenize(text)
        stems = []
        for item in tokens:
            stems.append(PorterStemmer().stem(item))
        return stems

    def transform_data_to_numpy_array(self):
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def concat_x_with(self, data):
        np.concatenate((self.x, data), axis=1)

    def n_grams_fit_transform(self, ngram_range=(1, 3)):
        vectorizer = CountVectorizer(ngram_range=ngram_range)
        self.x = vectorizer.fit_transform(self.corpus)

    def tfidf_transformer_fit_traform(self):
        tfidf_transformer = TfidfTransformer()
        x = tfidf_transformer.fit_transform(self.x)
        return x

    def tfidf_fit_transform(self):
        tfidf = TfidfVectorizer(tokenizer=Preprocesor.tokenize_for_tfidf, stop_words='english')
        tfidf.fit_transform(self.corpus)
        print(tfidf)
        return tfidf

    @staticmethod
    def delete_punctuation(text):
        return text.lower().translate(str.maketrans('', '', string.punctuation))

    def read_rel_extension_file(self, filepath):
        with open(filepath, 'r') as fd:
            for line in fd.readlines():
                data = line.rstrip().split("|")[:2]
                self.y.append(0)
                self.corpus.append(Preprocesor.delete_punctuation(data[1]))

    def read_txt_extension_file(self, filepath):
        with open(filepath, 'r') as fd:
            for line in fd.readlines():
                aux = re.findall(r"(\d+)\s+(NEG)\s+(.+)", line)
                self.y.append(1)
                self.corpus.append(Preprocesor.delete_punctuation(aux[0][2]))

    def shuffle_data(self):
        self.x, self.y = shuffle(self.x, self.y)

    def test_for_good_shuffle(self):
        count = 0
        for i in range(0, len(self.x)):
            if self.x[i][len(self.x) - 1] == self.y[0]:
                count += 1
        print(count, len(self.x))

    def get_concept(self, text_array):  # tfidf mai trebuie facut =-----------------------------------------
        concepts, error = self.mm.extract_concepts(text_array)
        semantic_types = set()
        cuis = set()

        if error:
            print(error)

        for concept in concepts:
            for semantic_type in re.findall(r'\w+', concept.semtypes):
                semantic_types.add(semantic_type)
            cuis.add(concept.cui)
        return list(semantic_types), list(cuis)

    @staticmethod
    def get_synonyms(word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonyms.add(l.name())
        return list(synonyms)

    @staticmethod
    def get_syn_set(pos_text):  # tfidf mai trebuie facut =-----------------------------------------
        result = []
        for word, pos_word in pos_text:
            if Preprocesor.NOUN in pos_word:
                result.append(list(set([word] + Preprocesor.get_synonyms(word))) + [pos_word])
            elif Preprocesor.ADJ in pos_word:
                result.append(list(set([word] + Preprocesor.get_synonyms(word))) + [pos_word])
            elif Preprocesor.VERB in pos_word:
                result.append(list(set([word] + Preprocesor.get_synonyms(word))) + [pos_word])
            else:
                result.append([word, pos_word])
        return result

    def get_sentiment_score(self, text):  # nu stiu daca e corect si bun cum am facut
        sentence_pos = self.ntlk_pos(text, to_stem=False)
        word_net_lemmatizer = WordNetLemmatizer()

        sum_score = 0
        sum_pos_socre = 0
        sum_neg_score = 0
        for word, word_pos in sentence_pos:
            swn_pos_tag = Preprocesor.metamap_pos_to_sentiwordnet_pos(word_pos)
            if swn_pos_tag != "":
                aux = word_net_lemmatizer.lemmatize(word) + '.' + swn_pos_tag + '.01'
                try:
                    sum_score += swn.senti_synset(aux).obj_score()
                    sum_pos_socre += swn.senti_synset(aux).pos_score()
                    sum_neg_score += swn.senti_synset(aux).neg_score()
                except WordNetError:
                    pass
        return sum_pos_socre / len(sentence_pos), sum_neg_score / len(sentence_pos), sum_score / len(sentence_pos)

    # def get_features(self):
    #     features = defaultdict(list)
    #     self.transform_data_to_numpy_array()
    #
    #     # for data in self.x:
    #     #     break
    #     # tokenized_string = Preprocesor.get_string_tokenizing(data[1])  # word_tokenize(data[1])
    #     # print(tokenized_string)
    #     # features["1-grams"].append(Preprocesor.get_n_grams_words(tokenized_string, 1))
    #     # features["2-grams"].append(Preprocesor.get_n_grams_words(tokenized_string, 2))
    #     # features["3-grams"].append(Preprocesor.get_n_grams_words(tokenized_string, 3))
    #     # features["semantic"].append(self.get_concept([data[1]]))
    #     # sentences_pos = self.ntlk_pos(tokenized_string, string_tokenezed=True, to_stem=False)
    #     # features["sentence_pos"].append(sentences_pos)
    #     # features["synset"].append(Preprocesor.get_syn_set(sentences_pos))
    #     # features["sentiment-score"].append(self.get_sentiment_score(data[1]))
    #     # features["id"].append(data[0])
    #     # features["stemmed-text"].append(Preprocessor.get_nltk_porter_stemming(tokenized_string))
    #
    #     return features

    def write_n_grams_to_file(self, filename, data):
        with open(filename, 'w') as fd:
            for i, x in enumerate(self.x):
                fd.write(" ".join(np.concatenate((x, data[i]))) + '\n')

    def split_train_test(self, test_size=0.25, random_state=0):
        return train_test_split(self.x, self.y, test_size=test_size, random_state=random_state)

    @staticmethod
    def train_fit_with_naive_bayes(x_train, y_train):
        model = MultinomialNB()
        model.fit(x_train, y_train)
        return model

    @staticmethod
    def train_fit_with_svc(x_train, y_train, kernel='linear', random_state=0):
        model = SVC(kernel=kernel, random_state=random_state)
        model.fit(x_train, y_train)
        return model

    @staticmethod
    def test_model(model, model_name, x_test, y_test):
        y_predicted = model.predict(x_test)
        print("Accuracy " + model_name + ": ", accuracy_score(y_test, y_predicted))

    def create_sem_abbreviation_translations(self, file_path=r'./abreviations_files/SemanticTypes_2018AB.txt'):
        with open(file_path, 'r') as fd:
            for line in fd.readlines():
                aux = line.split("|")
                self.sem_abbreviation_translations.append((aux[0], aux[len(aux) - 1].rstrip()))

    def get_sem_abbreviation_translation(self, sem_type):
        for data in self.sem_abbreviation_translations:
            if data[0] == sem_type:
                return data[1]
        return ""

    def translate_semantic_abbreviation(self, sem_types_abber):
        if not self.sem_abbreviation_translations:
            raise Exception("Please call 'create_sem_abbreviation_translations' for creating the " +
                            "abbreviation translations")
        result = []
        for sem_type in sem_types_abber:
            result.append(self.get_sem_abbreviation_translation(sem_type))
        return result

    def save_umls_features(self, filepath=r'./raw_features/semantic_types_ADE.txt'):
        with open(filepath, 'w') as fd:
            for text in self.corpus:
                sem_types, cuis = self.get_concept([text])
                aux = self.translate_semantic_abbreviation(sem_types)
                fd.write(" ".join(aux + cuis) + '\n')

    def load_umls_features(self, filepath=r'./raw_features/semantic_types_ADE.txt'):
        with open(filepath, 'r') as fd:
            for line in fd.readlines():
                self.umls_semtypes_cuis.append(line.rstrip())

    def create_concepts_file(self, filepath_to_save=r'./raw_features/semantic_types_ADE.txt',
                             filepath_to_abbr_file=r'./abreviations_files/SemanticTypes_2018AB.txt'):
        self.create_sem_abbreviation_translations(filepath_to_abbr_file)
        self.save_umls_features(filepath_to_save)

    def create_features(self):
        self.n_grams_fit_transform()

    def train_model(self, model_name='naive_bayes'):
        model = None
        self.create_features()
        self.shuffle_data()
        x_train, x_test, y_train, y_test = self.split_train_test()
        if model_name == 'naive_bayes':
            model = Preprocesor.train_fit_with_naive_bayes(x_train, y_train)  # NB accuracy score: 0.8462323524408913
        elif model_name == 'svc':
            model = Preprocesor.train_fit_with_svc(x_train, y_train)  # SVM accuracy score: 0.9027045415887056

        return model, x_test, y_test

# 4.1.2. N-grams - done
# 4.1.3. UMLS semantic types and concept IDs - half -> no tfidf -> Pe ce trebuie sa fac tdidf?
# 4.1.4. Syn-set expansion - half -> no tfidf -> Pe ce trebuie sa fac tdidf?
# 4.1.5. Change phrases - neimplementat
# 4.1.6. ADR lexicon matches - neimplementat
# 4.1.7. Sentiword scores - done -> nu stiu daca e corect
# 4.1.8. Topic-based feature - neimplementat -> nu inteleg cum trebuie sa iau topic-ul
#   see: https://rare-technologies.com/tutorial-on-mallet-in-python/


def main():
    p = Preprocesor()

    p.read_rel_extension_file(r"./corpus/ADE-Corpus-V2/DRUG-AE.rel")
    p.read_txt_extension_file(r"./corpus/ADE-Corpus-V2/ADE-NEG.txt")
    print(len(p.corpus))
    p.load_umls_features()

    return
    model_name = "naive_bayes"
    model, x_test, y_test = p.train_model("naive_bayes")

    Preprocesor.test_model(model, model_name, x_test, y_test)

    # transform_data_to_numpy_array()
    # p.n_grams_fit_transform()

    # p.get_features()
    # p.get_features(dataset)

    # TFDIF TEST:
    # sem, cuis = p.get_concept([non_adr[3][1]])
    # tfidf = p.create_fit_transform_tfidf(non_adr[1][1])
    # print(p.get_tfidf(tfidf, sem))

    # POS TEST:
    # sentences_pos = p.ntlk_pos(non_adr[1][1], to_stem=False)
    # print(sentences_pos)

    # Lenght in words
    # print(len(sentences_pos))

    # SYNs TEST:
    # print(Preprocesor.get_syn_set(sentences_pos))

    # Sentiment score TEST:
    # print(p.get_sentiment_score(non_adr[1][1]))

    # print(WordNetLemmatizer().lemmatize("was"))
    # print(p.get_features(non_adr)["1-grams"][1])
   

if __name__ == '__main__':
    main()
