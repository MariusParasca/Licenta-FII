from nltk import word_tokenize, PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet as swn
from nltk.corpus.reader.wordnet import WordNetError
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from scipy.sparse import csr_matrix, hstack
from pymetamap import MetaMap
from gensim.utils import simple_preprocess
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import f1_score
import pandas as pd
import gensim.corpora as corpora
import string
import re
import gensim
import spacy


class Preprocesor:
    ADJ = 'JJ'
    ADJ_COMPARATIVE = 'JJR'
    ADJ_SUPERLATIVE = 'JJS'
    MODAL = 'MD'
    NOUN = 'NN'
    VERB = 'VB'
    ADVERB = 'RB'
    SYN_TAG = "SYN"
    NAIVE_BAYES = 'naive_bayes'
    SVM = 'svm'

    def __init__(self, metamap_path='/home/noway/Facultate/Licenta/public_mm/bin/metamap18',
                 mallet_path=r'../mallet-2.0.8/bin/mallet'):
        self.mm = MetaMap.get_instance(metamap_path)
        self.mallet_path = mallet_path
        self.stop_words = stopwords.words('english')
        self.sem_abbreviation_translations = []
        self.x = []
        self.y = []
        self.corpus = []
        self.umls_semtypes_cuis = []
        self.syns_features = []
        self.sentiment_scores = []
        self.topics_features = []
        self.other_features = []

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
    def ntlk_pos(s, string_tokenezed=True, to_stem=True, to_string=False):
        if to_stem:
            s = Preprocesor.get_nltk_porter_stemming(word_tokenize(s), to_string=to_string)
        elif string_tokenezed:
            pass
        else:
            s = word_tokenize(s.lower())
        return pos_tag(s)

    @staticmethod
    def get_basic_preprocessig(text):
        text = Preprocesor.delete_punctuation(text)
        tokens = word_tokenize(text)
        return Preprocesor.get_nltk_porter_stemming(tokens, to_string=True)

    @staticmethod
    def get_sentiment_score(text):  # nu stiu daca e corect si bun cum am facut
        sentence_pos = Preprocesor.ntlk_pos(text, to_stem=False)
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

    @staticmethod
    def create_basic_processing_rel_file(corpus_filepah=r"./corpus/ADE-Corpus-V2/DRUG-AE.rel",
                                         processed_filepath=r'./raw_features/DRUG-AE_processed.rel'):
        with open(corpus_filepah, 'r') as fdr:
            with open(processed_filepath, 'w') as fdw:
                for line in fdr.readlines():
                    data = line.rstrip().split("|")[:2]
                    fdw.write(Preprocesor.get_basic_preprocessig(data[1]) + '\n')
        print("Basic processing rel extension file saved in", processed_filepath)

    @staticmethod
    def create_basic_processing_txt_file(corpus_filepah=r"./corpus/ADE-Corpus-V2/ADE-NEG.txt",
                                         processed_filepath=r'./raw_features/ADE-NEG_processed.txt'):
        with open(corpus_filepah, 'r') as fdr:
            with open(processed_filepath, 'w') as fdw:
                for line in fdr.readlines():
                    aux = re.findall(r"(\d+)\s+(NEG)\s+(.+)", line)
                    fdw.write(Preprocesor.get_basic_preprocessig(aux[0][2]) + '\n')
        print("Basic processing rel extension file saved in", processed_filepath)

    @staticmethod
    def load_sentiment_scores(filepath=r'./raw_features/sentiment_scores.txt'):
        result = []
        with open(filepath, 'r') as fd:
            for line in fd.readlines():
                scores = [float(i) for i in line.rstrip().split(" ")]
                result.append(scores)
        print("Sentiment scores loaded from:", filepath)
        return result

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
        return y_predicted

    def read_rel_extension_file(self, filepath=r'./raw_features/DRUG-AE_processed.rel'):
        with open(filepath, 'r') as fd:
            for line in fd.readlines():
                self.y.append(1)
                self.corpus.append(line.rstrip())
        print("Data loaded from", filepath)

    def read_txt_extension_file(self, filepath=r'./raw_features/ADE-NEG_processed.txt'):
        with open(filepath, 'r') as fd:
            for line in fd.readlines():
                self.y.append(0)
                self.corpus.append(line.rstrip())
        print("Data loaded from:", filepath)

    def concat_x_with(self, data, frmt='csr'):
        self.x = hstack([self.x, data], format=frmt)

    def create_n_grams(self, ngram_range=(1, 3)):
        vectorizer = CountVectorizer(ngram_range=ngram_range)
        self.x = vectorizer.fit_transform(self.corpus)
        print("1,2,3-grams created")

    def shuffle_data(self):
        self.x, self.y = shuffle(self.x, self.y)

    def get_concept(self, text_array):
        """Extract the concept using metamap. To make this work you need to open an terminal and change directory to
           public_mm folder of metamap and the run the commands:
           ./bin/skrmedpostctl start
           ./bin/wsdserverctl start"""
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

    def save_umls_features(self, filepath=r'./raw_features/old/semantic_types_ADE.txt'):
        """corpus need to be without any preprocessing"""
        self.create_sem_abbreviation_translations()
        with open(filepath, 'w') as fd:
            for text in self.corpus:
                sem_types, cuis = self.get_concept([text])
                aux = self.translate_semantic_abbreviation(sem_types)
                fd.write(" ".join(aux + cuis) + '\n')
        print("Not final UMLS features saved in:", filepath, "Please use translate_cuis.py to create the final file.")

    def load_umls_features(self, filepath=r'./raw_features/semantic_types_cuis_name_ADE.txt'):
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
                sentences_pos = Preprocesor.ntlk_pos(line, to_stem=True)
                synonyms = Preprocesor.get_syn_set(sentences_pos)
                if not synonyms:
                    fd.write("-\n")
                else:
                    fd.write(" ".join(synonyms) + "\n")
        print("Synonym features saved in:", filepath)

    def save_sentiment_scores(self, filepath=r'./raw_features/sentiment_scores.txt'):
        with open(filepath, 'w') as fd:
            for line in self.corpus:
                scores = list(Preprocesor.get_sentiment_score(line))
                fd.write(" ".join(map(str, scores)) + '\n')
        print("Sentiment scores saved in:", filepath)

    def save_other_features(self, filepath=r'./raw_features/other_features.txt'):
        with open(filepath, 'w') as fd:
            for sentence in self.corpus:
                aux = [str(len(sentence.split(" ")))]
                word_poses = Preprocesor.ntlk_pos(sentence, to_stem=True)
                adj_c = False
                adj_s = False
                modal = False
                for _, pos in word_poses:
                    if not adj_c and pos == Preprocesor.ADJ_COMPARATIVE:
                        aux.append('1')
                        adj_c = True
                if not adj_c:
                    aux.append('0')
                for _, pos in word_poses:
                    if not adj_s and pos == Preprocesor.ADJ_SUPERLATIVE:
                        aux.append('1')
                        adj_s = True
                if not adj_s:
                    aux.append('0')
                for _, pos in word_poses:
                    if not modal and pos == Preprocesor.MODAL:
                        aux.append('1')
                        modal = True
                if not modal:
                    aux.append('0')
                fd.write(" ".join(aux) + '\n')
        print("Other features saved in:", filepath)

    def load_other_features(self, filepath=r'./raw_features/other_features.txt'):
        with open(filepath, 'r') as fd:
            for line in fd.readlines():
                self.other_features.append([int(i) for i in line.rstrip().split(" ")])
        print("Other features laoded from:", filepath)

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

    def create_topics_features(self):
        self.load_topics_features()
        self.topics_features = csr_matrix(self.topics_features)
        self.concat_x_with(self.topics_features)

    def create_other_features(self):
        self.load_other_features()
        self.concat_x_with(csr_matrix(self.other_features))

    def create_features(self):
        self.create_n_grams()

        self.create_tfidf_umls()

        self.create_tfidf_syns()

        # self.create_sentiment_scores()
        # self.create_topics_features()
        self.create_other_features()
        return


    def oversample_dataset(self):
        print('Original dataset shape %s' % Counter(self.y))
        sm = SMOTE(random_state=42)
        self.x, self.y = sm.fit_resample(self.x, self.y)
        print('Original dataset shape %s' % Counter(self.y))

    def prepare_data_fro_training(self, data_oversample=True):
        self.create_features()
        self.shuffle_data()
        if data_oversample:
            self.oversample_dataset()

    def cross_validate_model(self, model_name, cv=3, svm_kervel='linear'):
        self.prepare_data_fro_training(data_oversample=True)
        if model_name == Preprocesor.NAIVE_BAYES:
            model = MultinomialNB()
        elif model_name == Preprocesor.SVM:
            model = SVC(kernel=svm_kervel)
        else:
            raise Exception("Unknown model. Available models: naive_bayes, svm")
        cv_results = cross_validate(model, self.x, self.y, cv=cv, return_train_score=True)
        # print("Fit time", cv_results['fit_time'])
        print("Test score", cv_results['test_score'])
        print("Train score", cv_results['train_score'])

    def train_model(self, model_name='naive_bayes', svm_kernel='rbf'):
        self.prepare_data_fro_training()
        x_train, x_test, y_train, y_test = self.split_train_test()
        print("Start training using", model_name)
        if model_name == Preprocesor.NAIVE_BAYES:
            model = Preprocesor.train_fit_with_naive_bayes(x_train, y_train)  # NB accuracy score: 0.8462323524408913 ->
            # 0.8617111753699609 -> 0.8710665079095085 -> 0.873958156149005 -> 0.8788909678516754 -> 0.8845041673754039
        elif model_name == Preprocesor.SVM:
            model = Preprocesor.train_fit_with_svc(x_train, y_train, kernel=svm_kernel)  # SVM accuracy score:
            # Linear kernel: 0.9027045415887056, 0.9055961898282021, 0.9069569654703181
            # RBF kernel: 0.70
        else:
            raise Exception("Unknown model. Available models: naive_bayes, svm")
        print("Finishing training")
        return model, x_test, y_test

    def create_tokenized_corpus(self):
        aux = []
        for sentence in self.corpus:
            aux.append(word_tokenize(sentence))
        return aux

    def create_bigrams_models(self):
        bigram = gensim.models.Phrases(self.create_tokenized_corpus(), min_count=5, threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)

        return bigram_mod

    @staticmethod
    def remove_stopwords(texts):
        stop_words = stopwords.words('english')
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    @staticmethod
    def make_bigrams(bigram_mod, texts):
        return [bigram_mod[doc] for doc in texts]

    @staticmethod
    def lemmatization(texts, allowed_postags=None):
        """https://spacy.io/api/annotation"""
        if not allowed_postags:
            allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
        nlp = spacy.load('en', disable=['parser', 'ner'])
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    @staticmethod
    def preprocess_for_topics(data_words, bigram_mod):
        data_words_nostops = Preprocesor.remove_stopwords(data_words)

        data_words_bigrams = Preprocesor.make_bigrams(bigram_mod, data_words_nostops)

        data_lemmatized = Preprocesor.lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        print(data_lemmatized[:3])
        id2word = corpora.Dictionary(data_lemmatized)
        print(id2word)

        texts = data_lemmatized

        corpus = [id2word.doc2bow(text) for text in texts]

        return corpus, id2word

    @staticmethod
    def create_lda_mallet(corpus, id2word, mallet_path=r'../mallet-2.0.8/bin/mallet'):
        ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)
        return ldamallet

    @staticmethod
    def format_topics_sentences(ldamodel, corpus, texts):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                        pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return sent_topics_df

    def save_topics_features(self, filepath='./raw_features/topics.txt'):
        data_words = self.create_tokenized_corpus()
        bigram_mod = self.create_bigrams_models()
        corpus, id2word = Preprocesor.preprocess_for_topics(data_words, bigram_mod)
        optimal_model = Preprocesor.create_lda_mallet(corpus, id2word)

        df_topic_sents_keywords = Preprocesor.format_topics_sentences(ldamodel=optimal_model, corpus=corpus,
                                                                      texts=self.corpus)

        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

        with open(filepath, 'w') as fd:
            fd.write(df_dominant_topic.to_string())

    def load_topics_features(self, filepath='./raw_features/topics.txt'):
        with open(filepath, 'r') as fd:
            lines = fd.readlines()
            for line in lines[1:]:
                self.topics_features.append([float(i) for i in re.findall(r'\d+\.\d+', line)])
# use max entropy
# K fold cross validations
# 4.1.2. N-grams - done
# 4.1.3. UMLS semantic types and concept IDs - done
# 4.1.4. Syn-set expansion - done
# 4.1.5. Change phrases - neimplementat
# 4.1.6. ADR lexicon matches - neimplementat
# 4.1.7. Sentiword scores - done
# 4.1.8. Topic-based feature - done
#   see: https://rare-technologies.com/tutorial-on-mallet-in-python/
# 4.1.9. Other features - done

# With SMOTE:
# Accuracy svm Linear:  0.956756109247724
# Accuracy naive_bayes:  0.8959032103497844
# Accuracy svm RBF:  0.567680881648299


def main():
    p = Preprocesor()

    p.read_rel_extension_file()
    p.read_txt_extension_file()

    model_name = "svm"

    # p.cross_validate_model(model_name, cv=5)

    model, x_test, y_test = p.train_model(model_name, svm_kernel='rbf')

    y_predicted = Preprocesor.test_model(model, model_name, x_test, y_test)
    f_score = f1_score(y_test, y_predicted, average='weighted')
    print("F score is: ", f_score)

    # plt.spy(p.x, markersize=10.0)
    # plt.show()
   

if __name__ == '__main__':
    main()
