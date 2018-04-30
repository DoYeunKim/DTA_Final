from nltk.corpus import PlaintextCorpusReader, stopwords
from nltk.tokenize import RegexpTokenizer
import sys
import codecs

regxT = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('spanish'))

corpus_root = "./DTA_FinalCorpus/"
my_corpus = PlaintextCorpusReader(corpus_root, '.*txt')
documents = []
titles = []
shortest = sys.maxsize

# Read in the files.
for text in my_corpus.fileids():
	afile = codecs.open(corpus_root + text, 'r', 'utf-8-sig')
	txt = afile.read()

	# regex tokenize the texts
	regxd = regxT.tokenize(str(txt).lower())

	# create list of non-stop words for each text
	prechunk = []	
	for tokens in regxd:
		if (tokens not in stop_words):
			prechunk.append(tokens)
	print(len(prechunk))

	# add preprocessed texts to the documents list (as string)
	documents.append(' '.join(prechunk))
	titles.append(text[:-4])
	

# Get the length of each file


# chunk them and save as individual files
