from nltk import word_tokenize, PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet as swn
from nltk.corpus.reader.wordnet import WordNetError
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from scipy.sparse import csr_matrix, hstack
from pymetamap import MetaMap
import re
import string


class Preprocesor:
    ADJ = 'JJ'
    NOUN = 'NN'
    VERB = 'VB'
    ADVERB = 'RB'
    SYN_TAG = "SYN"

    def __init__(self, metamap_path='/home/noway/Facultate/Licenta/public_mm/bin/metamap18'):
        self.mm = MetaMap.get_instance(metamap_path)
        self.sem_abbreviation_translations = []
        self.x = csr_matrix([])
        self.y = []
        self.corpus = []
        self.umls_semtypes_cuis = []
        self.syns_features = []
        self.sentiment_scores = []

    @staticmethod
    def metamap_pos_to_sentiwordnet_pos(pos):
        """Transform metamap part-of-speech to sentiwordnet part-of-speech.
           If part-of-speech is different than noun, verb, adjective, or adverb we return the empty string: ''."""
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

    @staticmethod
    def get_nltk_porter_stemming(tokens, to_string=True):
        """Porter stemming using ntlk word tokenizer.
           If to_string=True we join the list by a ' '. Example: " ".join(stems)."""
        porter_stemmer = PorterStemmer()
        stems = []
        for item in tokens:
            stems.append(porter_stemmer.stem(item))
        if to_string:
            return " ".join(stems)
        else:
            return stems

    @staticmethod
    def delete_punctuation(text):
        """Delete the punctuation from text."""
        return text.lower().translate(str.maketrans('', '', string.punctuation))

    @staticmethod
    def get_synonyms(word):
        """Get all the synonyms that a word have."""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonyms.add(l.name() + Preprocesor.SYN_TAG)
        return list(synonyms)

    @staticmethod
    def get_syn_set(pos_text):
        result = []
        for word, pos_word in pos_text:
            if Preprocesor.NOUN in pos_word or Preprocesor.ADJ in pos_word or Preprocesor.VERB in pos_word:
                result += Preprocesor.get_string_from_syn_set(word)
        return result

    @staticmethod
    def get_string_from_syn_set(word):
        aux = Preprocesor.get_synonyms(word)
        if not aux:
            return ""
        else:
            return aux

    @staticmethod
    def train_fit_with_naive_bayes(x_train, y_train):
        model = MultinomialNB()
        model.fit(x_train, y_train)
        return model

    @staticmethod
    def train_fit_with_svc(x_train, y_train, kernel='rbf', random_state=0):
        model = SVC(kernel=kernel, random_state=random_state)
        model.fit(x_train, y_train)
        return model

    @staticmethod
    def test_model(model, model_name, x_test, y_test):
        y_predicted = model.predict(x_test)
        print("Accuracy " + model_name + ": ", accuracy_score(y_test, y_predicted))

    def ntlk_pos(self, s, string_tokenezed=True, to_stem=True, to_string=False):
        if to_stem:
            s = self.get_nltk_porter_stemming(word_tokenize(s), to_string=to_string)
        elif string_tokenezed:
            pass
        else:
            s = word_tokenize(s.lower())
        return pos_tag(s)

    def concat_x_with(self, data, frmt='csr'):
        self.x = hstack([self.x, data], format=frmt)

    def create_n_grams(self, ngram_range=(1, 3)):
        vectorizer = CountVectorizer(ngram_range=ngram_range)
        self.x = vectorizer.fit_transform(self.corpus)

    def read_rel_extension_file(self, filepath):
        with open(filepath, 'r') as fd:
            for line in fd.readlines():
                data = line.rstrip().split("|")[:2]
                self.y.append(0)
                self.corpus.append(Preprocesor.delete_punctuation(data[1]))
        print("Data loaded from", filepath)

    def read_txt_extension_file(self, filepath):
        with open(filepath, 'r') as fd:
            for line in fd.readlines():
                aux = re.findall(r"(\d+)\s+(NEG)\s+(.+)", line)
                self.y.append(1)
                self.corpus.append(Preprocesor.delete_punctuation(aux[0][2]))
        print("Data loaded from:", filepath)

    def shuffle_data(self):
        self.x, self.y = shuffle(self.x, self.y)

    def get_concept(self, text_array):
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

    def split_train_test(self, test_size=0.25, random_state=0):
        return train_test_split(self.x, self.y, test_size=test_size, random_state=random_state)

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
        print("UMLS features saved in:", filepath)

    def load_umls_features(self, filepath=r'./raw_features/semantic_types_ADE.txt'):
        with open(filepath, 'r') as fd:
            for line in fd.readlines():
                self.umls_semtypes_cuis.append(line.rstrip())
        print("UMLS features loaded from:", filepath)

    def load_syns_features(self, filepath=r'./raw_features/syns_ADE.txt'):
        with open(filepath, 'r') as fd:
            for line in fd.readlines():
                self.syns_features.append(line.rstrip())
        print("Synonym features loaded from:", filepath)

    def save_syns_features(self, filepath=r'./raw_features/syns_ADE.txt'):
        with open(filepath, 'w') as fd:
            for line in self.corpus:
                sentences_pos = self.ntlk_pos(line, to_stem=True)
                synonyms = Preprocesor.get_syn_set(sentences_pos)
                if not synonyms:
                    fd.write("-\n")
                else:
                    fd.write(" ".join(synonyms) + "\n")
        print("Synonym features saved in:", filepath)

    def save_sentiment_scores(self, filepath=r'./raw_features/sentiment_scores.txt'):
        with open(filepath, 'w') as fd:
            for line in self.corpus:
                scores = list(self.get_sentiment_score(line))
                fd.write(" ".join(map(str, scores)) + '\n')
        print("Sentiment scores saved in:", filepath)

    @staticmethod
    def load_sentiment_scores(filepath=r'./raw_features/sentiment_scores.txt'):
        result = []
        with open(filepath, 'r') as fd:
            for line in fd.readlines():
                scores = [float(i) for i in line.rstrip().split(" ")]
                result.append(scores)
        print("Sentiment scores loaded from:", filepath)
        return result

    def create_sentiment_scores(self):
        self.sentiment_scores = csr_matrix(Preprocesor.load_sentiment_scores())
        self.concat_x_with(self.sentiment_scores)

    def create_concepts_file(self, filepath_to_save=r'./raw_features/semantic_types_ADE.txt',
                             filepath_to_abbr_file=r'./abreviations_files/SemanticTypes_2018AB.txt'):
        self.create_sem_abbreviation_translations(filepath_to_abbr_file)
        self.save_umls_features(filepath_to_save)

    def create_tfidf_syns(self):
        self.load_syns_features()
        vectorizer = CountVectorizer()
        x_aux = vectorizer.fit_transform(self.syns_features)
        tfidf_transformer = TfidfTransformer()
        x_to_stack = tfidf_transformer.fit_transform(x_aux)
        self.concat_x_with(x_to_stack)

    def create_tfidf_umls(self):
        self.load_umls_features()
        vectorizer = CountVectorizer()
        x_aux = vectorizer.fit_transform(self.umls_semtypes_cuis)
        tfidf_transformer = TfidfTransformer()
        x_to_stack = tfidf_transformer.fit_transform(x_aux)
        self.concat_x_with(x_to_stack)

    def create_features(self):
        self.create_n_grams()
        self.create_tfidf_umls()
        self.create_tfidf_syns()
        self.create_sentiment_scores()

    def train_model(self, model_name='naive_bayes'):
        self.create_features()
        self.shuffle_data()
        x_train, x_test, y_train, y_test = self.split_train_test()
        print("Start training using", model_name)
        if model_name == 'naive_bayes':
            model = Preprocesor.train_fit_with_naive_bayes(x_train, y_train)  # NB accuracy score: 0.8462323524408913 ->
            # 0.8617111753699609
        elif model_name == 'svm':
            model = Preprocesor.train_fit_with_svc(x_train, y_train)  # SVM accuracy score:
            # Liniar kernel: 0.9027045415887056, 0.9055961898282021
            # RBF kernel: 0.70
        else:
            raise Exception("Unknown model. Available models: naive_bayes, svc")
        print("Finishing training")
        return model, x_test, y_test

# 4.1.2. N-grams - done
# 4.1.3. UMLS semantic types and concept IDs - done (nu stiu daca CUI-urile trebuie transformate cumva,
# asa ca le-am lasat asa cum sunt ele) -> aprox 0.5-1% imbunatatire cu NB
# 4.1.4. Syn-set expansion - done -> aprox 0.5-1% imbunatatire cu NB
# 4.1.5. Change phrases - neimplementat
# 4.1.6. ADR lexicon matches - neimplementat
# 4.1.7. Sentiword scores - done -> nu stiu daca e corect -> nu pare a avea vreo imbunatatirele a acuratetei
# 4.1.8. Topic-based feature - neimplementat -> nu inteleg cum trebuie sa iau topic-ul
#   see: https://rare-technologies.com/tutorial-on-mallet-in-python/


def main():
    p = Preprocesor()

    p.read_rel_extension_file(r"./corpus/ADE-Corpus-V2/DRUG-AE.rel")
    p.read_txt_extension_file(r"./corpus/ADE-Corpus-V2/ADE-NEG.txt")

    model_name = "svm"
    model, x_test, y_test = p.train_model(model_name)

    Preprocesor.test_model(model, model_name, x_test, y_test)

    # transform_data_to_numpy_array()
    # p.create_n_grams()

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
