import nltk, re, pprint
from nltk import word_tokenize
from urllib import request
from bs4 import BeautifulSoup

import nltk
nltk.download('averaged_perceptron_tagger')

# tokenizetion from guternberg book

url = "http://www.gutenberg.org/files/2554/2554-0.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')

tokens = word_tokenize(raw)
text = nltk.Text(tokens)
text.collocations()

beg = raw.find("PART I")
end = raw.rfind("End of Project Gutenberg's Crime")
raw = raw[beg:end]

# tokenizetion from HTML pages

url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
html = request.urlopen(url).read().decode('utf8')
raw = BeautifulSoup(html).get_text()
tokens = word_tokenize(raw)
tokens = tokens[110:390]
text = nltk.Text(tokens)
text.concordance('gene')


# Normalization text | Stemming
raw = """DENNIS: Listen, strange women lying in ponds distributing swords
is no basis for a system of government.  Supreme executive power derives from
a mandate from the masses, not from some farcical aquatic ceremony."""
tokens = word_tokenize(raw)

porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
wnl = nltk.WordNetLemmatizer()
# removes affixes from words
[wnl.lemmatize(t) for t in tokens]
[porter.stem(t) for t in tokens] 
[lancaster.stem(t) for t in tokens] 

# N-grams
a = zip(*[[1,2,3], [1,2,3], [1,2,3]])
for i in a:
    print(i)



