from nltk import word_tokenize, PorterStemmer, ngrams, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet as swn
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from pymetamap import MetaMap
import numpy as np
import re
import string


class Preprocesor:
    ADJ = 'JJ'
    NOUN = 'NN'
    VERB = 'VB'
    ADVERB = 'RB'

    def __init__(self, metamap_path='/home/noway/Facultate/Licenta/public_mm/bin/metamap18'):
        self.mm = MetaMap.get_instance(metamap_path)
        # self.tfidf = TfidfVectorizer(tokenizer=self.tokenize_for_tfidf, stop_words='english')
        self.dataset = []
        self.dataset_without_punctuation = []

    def metamap_POS_to_sentiwordnet_POS(self, pos):
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

    def get_nltk_porter_stemming(self, tokens, to_string=True):
        """Porter stemming using ntlk word tokenizer"""
        porter_stemmer = PorterStemmer()
        stems = []
        for item in tokens:
            stems.append(porter_stemmer.stem(item))
        if to_string:
            return " ".join(stems)
        else:
            return  stems

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

    def delete_punctuation(self):
        for data in self.dataset:
            text = data[1].lower().translate(str.maketrans('', '', string.punctuation))
            self.dataset_without_punctuation.append(text)

    def fit_transform_tfidf(self):
        self.delete_punctuation()
        self.tfidf.fit_transform(self.dataset_without_punctuation)

    def create_fit_transform_tfidf(self, text):
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        tfidf = TfidfVectorizer(tokenizer=self.tokenize_for_tfidf, stop_words='english')
        tfidf.fit_transform([text])
        return tfidf

    def get_tfidf(self, tfidf, textArray):
        response = tfidf.transform(textArray)
        feature_names = tfidf.get_feature_names()
        for col in response.nonzero()[1]:
            print(feature_names[col], ' - ', response[0, col])

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

    def get_concept(self, text_array):  # tfidf mai trebuie facut =-----------------------------------------
        concepts, error = self.mm.extract_concepts(text_array)
        semantic_types = set()
        cuis = set()
        if error:
            print(error)
        print(concepts)
        for concept in concepts:
            for semantic_type in re.findall(r'\w+', concept.semtypes):
                semantic_types.add(semantic_type)
            cuis.add(concept.cui)
        return list(semantic_types), list(cuis)

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonyms.add(l.name())
        return list(synonyms)

    def get_syn_set(self, pos_text): # tfidf mai trebuie facut =-----------------------------------------
        result = []
        for word, pos_word in pos_text:
            if Preprocesor.NOUN in pos_word:
                result.append(list(set([word] + self.get_synonyms(word))) + [pos_word])
            elif Preprocesor.ADJ in pos_word:
                result.append(list(set([word] + self.get_synonyms(word))) + [pos_word])
            elif Preprocesor.VERB in pos_word:
                result.append(list(set([word] + self.get_synonyms(word))) + [pos_word])
            else:
                result.append([word, pos_word])
        return result

    def get_sentiment_score(self, text): # nu stiu daca e corect si bun cum am facut
        sentence_pos = self.ntlk_pos(text, to_stem=False)
        word_net_lemmatizer = WordNetLemmatizer()

        sum_score = 0
        sum_pos_socre = 0
        sum_neg_score = 0
        for word, word_pos in sentence_pos:
            swn_pos_tag = self.metamap_POS_to_sentiwordnet_POS(word_pos)
            if swn_pos_tag != "":
                aux = word_net_lemmatizer.lemmatize(word) + '.' + swn_pos_tag + '.01'
                try:
                    sum_score += swn.senti_synset(aux).obj_score()
                    sum_pos_socre += swn.senti_synset(aux).pos_score()
                    sum_neg_score += swn.senti_synset(aux).neg_score()
                except:
                    pass
        return sum_pos_socre / len(sentence_pos), sum_neg_score / len(sentence_pos), sum_score / len(sentence_pos)

    def get_features(self, dataset):
        features = defaultdict(list)
        # self.fit_transform_tfidf()
        for data in dataset:
            tokenized_string = self.get_string_tokenizing(data[1])  # word_tokenize(data[1])
            print(tokenized_string)
            features["1-grams"].append(self.get_n_grams(tokenized_string, 1))
            features["2-grams"].append(self.get_n_grams(tokenized_string, 2))
            features["3-grams"].append(self.get_n_grams(tokenized_string, 3))
            features["semantic"].append(self.get_concept([data[1]]))
            sentences_pos = self.ntlk_pos(tokenized_string, string_tokenezed=True, to_stem=False)
            features["sentence_pos"].append(sentences_pos)
            features["synset"].append(self.get_syn_set(sentences_pos))
            features["sentiment-score"].append(self.get_sentiment_score(data[1]))
            print(features)
            break
            # self.get_tfidf(data[1])
            # features["id"].append(data[0])
            # features["stemmed-text"].append(self.get_nltk_porter_stemming(tokenized_string))

        return features

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
    non_adr = p.read_rel_extension_file(r"./corpus/ADE-Corpus-V2/DRUG-AE.rel")
    adr = p.read_txt_extension_file(r"./corpus/ADE-Corpus-V2/ADE-NEG.txt")
    dataset = np.array(non_adr + adr)


    p.get_features(dataset)


    # TFDIF TEST:
    # sem, cuis = p.get_concept([non_adr[3][1]])
    # tfidf = p.create_fit_transform_tfidf(non_adr[1][1])
    # print(p.get_tfidf(tfidf, sem))

    # POS TEST:
    # sentences_pos = p.ntlk_pos(non_adr[1][1], to_stem=False)
    # print(sentences_pos)

    #Lenght in words
    # print(len(sentences_pos))

    # SYNs TEST:
    # print(p.get_syn_set(sentences_pos))

    # Sentiment score TEST:
    # print(p.get_sentiment_score(non_adr[1][1]))

    # print(WordNetLemmatizer().lemmatize("was"))
    # print(p.get_features(non_adr)["1-grams"][1])


if __name__ == '__main__':
    main()


