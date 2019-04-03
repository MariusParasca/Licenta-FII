from nltk import word_tokenize, PorterStemmer, ngrams
from nltk.tag import pos_tag
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from pymetamap import MetaMap
import numpy as np
import re
import string
# import jpype


class Preprocesor:
    def __init__(self, metamap_path='/home/noway/Facultate/Licenta/public_mm/bin/metamap18'):
        self.mm = MetaMap.get_instance(metamap_path)
        self.tfidf = TfidfVectorizer(tokenizer=self.tokenize_for_tfidf, stop_words='english')
        self.dataset = []
        self.dataset_without_punctuation = []

    # def stanford_parse(self, s, jvm_path=r"C:\Program Files\Java\jdk1.8.0_161\jre\bin\server\jvm.dll",
    #                    model=".\stanford-postagger-2018-10-16\models\english-left3words-distsim.tagger",
    #                    classpath=r".\3rd party java\stanford-parser.jar"):
    #     """Standford Part-Of-Speech using jpype as an interface to Java"""
    #     jpype.startJVM(jvm_path, "-Djava.class.path=%s" % classpath)
    #     nlp = jpype.JPackage("edu").stanford.nlp
    #     tagger = nlp.tagger.maxent.MaxentTagger(model)
    #     tokenize_text = nlp.tagger.maxent.MaxentTagger.tokenizeText
    #     text = jpype.java.io.StringReader(s)
    #     sentences = tokenize_text(text)
    #     result = []
    #     for sentence in sentences:
    #         tsentence = tagger.tagSentence(sentence)
    #         string = str(tsentence.toString())
    #         result.append(string.replace("[", "").replace("]", "").replace(", ", " "))
    #
    #     jpype.shutdownJVM()
    #     return result

    def ntlk_pos(self, s):
        return pos_tag(word_tokenize(s))

    def get_nltk_porter_stemming(self, tokens):
        """Porter stemming using ntlk word tokenizer"""
        stems = []
        for item in tokens:
            stems.append(PorterStemmer().stem(item))
        return " ".join(stems)

    def get_porter_stemming(self, string):
        """Porter stemming using my string tokenizing method"""
        tokens = self.get_string_tokenizing(string)
        porter = PorterStemmer()
        porter_stemming = [porter.stem(t) for t in tokens]
        return porter_stemming

    def get_string_tokenizing(self, string):
        """ My string tokenizing method"""
        string = string.lower()
        string = re.sub(r'[^a-zA-Z0-9\s]', ' ', string)
        return [token for token in string.split(" ") if token != ""]

    def get_n_grams(self, tokenized_string, n):
        """"""
        return list(ngrams(tokenized_string, n))

    def tokenize_for_tfidf(self, text):
        tokens = word_tokenize(text)
        stems = []
        for item in tokens:
            stems.append(PorterStemmer().stem(item))
        return stems

    # Cum fac tfidf-ul ? Doar pe textul pentru un medicament?

    def delete_punctuation(self):
        for data in self.dataset:
            text = data[1].lower().translate(str.maketrans('', '', string.punctuation))
            self.dataset_without_punctuation.append(text)

    def fit_transform_tfidf(self):
        self.delete_punctuation()
        self.tfidf.fit_transform(self.dataset_without_punctuation)

    def get_tfidf(self, text):
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        response = self.tfidf.transform([text])
        # print(response)
        feature_names = self.tfidf.get_feature_names()
        # for col in response.nonzero()[1]:
        #     print(feature_names[col], ' - ', response[0, col])

    def read_rel_extension_file(self, filepath):
        result = []
        with open(filepath, 'r') as fd:
            for line in fd.readlines():
                data = line.rstrip().split("|")[:2]
                result.append([data[0], data[1], False])
        self.dataset += result
        return result

    def read_txt_extension_file(self, filepath):
        result = []
        with open(filepath, 'r') as fd:
            for line in fd.readlines():
                aux = re.findall(r"(\d+)\s+(NEG)\s+(.+)", line)
                result.append([aux[0][0], aux[0][2], True])
        self.dataset += result
        return result

    def get_concept(self, text_array):
        concepts, error = self.mm.extract_concepts(text_array)
        if error:
            print(error)
        return concepts

    def get_features(self, dataset):
        features = defaultdict(list)
        # self.fit_transform_tfidf()
        for data in dataset:
            # self.get_tfidf(data[1])
            tokenized_string = word_tokenize(data[1])
            features["id"].append(data[0])
            features["stemmed-text"].append(self.get_nltk_porter_stemming(tokenized_string))
            features["2-grams"].append(self.get_n_grams(tokenized_string, 2))
        return features


def main():
    p = Preprocesor()
    non_adr = p.read_rel_extension_file(r"./corpus/ADE-Corpus-V2/DRUG-AE.rel")
    # adr = p.read_txt_extension_file(r"./corpus/ADE-Corpus-V2/ADE-NEG.txt")
    print(p.get_concept([non_adr[2][1]]))
    # dataset = np.array(non_adr + adr)
    # print(p.get_features(dataset))
    # string = input("Enter an string")
    # porter_stemming = get_porter_stemming(s)
    # two_grams = get_n_grams(s, n=2)
    # three_grams = get_n_grams(s, n=3)
    # print("Poter Stemming:", porter_stemming)
    # print(two_grams)
    # print(three_grams)
    # print(stanford_parse(s))
    # print(ntlk_pos(s))


if __name__ == '__main__':
    main()
