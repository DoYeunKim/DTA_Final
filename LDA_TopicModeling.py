from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import glob, os

# Initialize regular expression tokenizer
regxT = RegexpTokenizer(r'\w+')

# We are using NLTK's spanish stop words and adding a few words that we think are missing from the provided set
sSW = set(stopwords.words('spanish'))
#print(type(stop_words))
additionalSW = set(['me', 'se', 'las', 'hacia', 'ser', 'los', 'hacer', 'en', 'don', 'así', 'podía'])
#print(type(additionalSW))
# Concat the two sets
sSW = sSW|additionalSW
#print(type(stop_words))


## Helper function to print the words of each topic
def display_topics(model, feature_names, no_top_words, numT):
	for topic_idx, topic in enumerate(model.components_):
		print ("Topic %d:" % (topic_idx))
		print (" ".join([feature_names[i]+ ' ' + str(round(topic[i], 2))+' \n ' for i in topic.argsort()[:-no_top_words - 1:-1]]))

		# Output the topics as files
		textName = '../LDA_' + str(numT) + '_' + str(max_iter) + '_' + str(chunk_size) + '.txt'
		with open(textName, 'a') as f:
			print ("Topic %d:" % (topic_idx), file=f)
			print (" ".join([feature_names[i]+ ' ' + str(round(topic[i], 2))+' \n ' for i in topic.argsort()[:-no_top_words - 1:-1]]), file=f)


## Helper Function to break a string into chunks        
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

# Change current folder to the location where your files are stored\n",
os.chdir('./texts')

documents = []
    
# Lets create documents with 1000 words
chunk_size = 2000
max_iter = 400

# Read in the text files    
for filename in glob.glob("*.txt"):
	## Open and read the file
	file = open(filename, "r")
	text = file.read()

	#Use tokenizer to split the file text into words
	file_words = regxT.tokenize(text)
	
	# Remove stop words 
	prechunk = []
	for tokens in file_words:
		if (tokens not in sSW):
			prechunk.append(tokens)

	# Now we will partion the file into documents of the size size (chunk_size)
	words_chunks = list(chunks(prechunk,chunk_size))  
	for i in range(len(words_chunks)):
		documents.append(' '.join(words_chunks[i]))

    
#Define the maximum number of features to be considered
no_features = 1000
   
# Create the Vector Space with CountVectorizers
tf_vectorizer= CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words=sSW)
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

# We want to see 5, 20, and 40 topics
no_topics = [40, 60, 80]
# Run LDA
for num in no_topics:
	lda = LatentDirichletAllocation(n_topics=num, max_iter=400).fit(tf)

	no_top_words = 10
	display_topics(lda, tf_feature_names, no_top_words, num)

	lda.perplexity(tf)
	print("Done with " + str(num) + " topics")

