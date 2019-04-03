import nltk

print(nltk.corpus.wordnet.synsets('good')[0])

my_word = nltk.corpus.wordnet.synset('good.n.02')
print(my_word.lemmas())  # Output: [Lemma('good.n.01.good')]
print(my_word.lemmas()[1].name())  # Output: good
print(my_word.lemmas()[0].antonyms())  # Output: []
