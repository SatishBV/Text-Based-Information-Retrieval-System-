import sys										#importing libraries
import nltk
import os, time
import math
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


x = str(sys.argv[1]) #take input from command line
#print x
class token_list:

	def __init__(self):
		self.document_count=0
		self.term_frequency=0
		self.id_list=[]
		self.doc_term_count={}
		self.tfid={}

	def add_document_id(self,document_id,term_count):
		self.id_list.append(document_id)
		self.document_count=self.document_count+1;
		self.term_frequency+=term_count
		self.doc_term_count[document_id]=term_count

	def calculate_tfid(self,N):
		for d_id in self.id_list:
			self.tfid[d_id]=(1+math.log(self.doc_term_count[d_id]))*(math.log(N/self.document_count))


#phase1
st = time.time()
documents=[]												#empty document list
stop_words=stopwords.words("english")						#stop_words contain all the english stopwords
for file in [doc for doc in os.listdir(dir_path+"\\corpus")	#path for corpus
if doc.endswith(".txt")]:
    documents.append(file)
#documents =["python_wiki","java_wiki"]
no_of_documents=len(documents)
#print (no_of_documents)
#print (documents)
terms={}												#creating terms dictionary to hold (term,term_count)
tokenizer = RegexpTokenizer(r'\w+')						#tokenizing the terms using regex
stemmer = SnowballStemmer("english")					#stemming
for doc_name in documents:
	file_t=open(dir_path+"\\corpus"+"\\"+doc_name,'r')	#opening each document in read mode
	text=file_t.read()
	tokenized_list=tokenizer.tokenize(text)				#this list contains all tokenized terms
	stemmed_list=[stemmer.stem(words) for words in tokenized_list]	#stemmed list contains terms resulted after stemming
	copy_tokenized_list=list(stemmed_list)
	for j in copy_tokenized_list:
		if (j in terms):
			if (doc_name in terms[j].id_list):
				continue;
			else:										#adding terms to list corresponding to that term
				terms[j].add_document_id(doc_name,copy_tokenized_list.count(j))
		else:											#creating lists for new terms
			terms[j]=token_list()
			terms[j].add_document_id(doc_name,copy_tokenized_list.count(j))
														#ex. term[hello] list contains all docs having hello
key= terms.keys()										#taking all terms to list key
#for p in key:
#	print p, terms[p].document_count, terms[p].term_frequency

for p in key:
	terms[p].calculate_tfid(no_of_documents)			#calculating tf-idf score
#print(time.time()-st)
#print(len (terms))
#PHASE 2

doc_normalization={}									#creating doc_normalization dictionary
for doc_name in documents:
	value=0
	value2=0 
	for j in terms:
		if (doc_name in terms[j].id_list):
			value=value+(terms[j].tfid[doc_name] * terms[j].tfid[doc_name])	#calculating sum of squares of tf-idf values within a document
			value2=value2+value
	doc_normalization[doc_name]=math.sqrt(value2)
#print(time.time()-st)

def search():
	"""Retuns the list documents sorted in the order of relevance to the given query

	Given query is tokenized using RegexpTokenizer, stemmed using snowballStemmer
	This then gives the required list of documents containing query based upon
	their respective cosine scores
	"""
	st = time.time()
	#print st
	query = x
	#print query
	query = query.lower()
	tokenized_query = tokenizer.tokenize(query)
	query_store = tokenized_query[:]
	tokenized_query = [word for word in tokenized_query if word not in stop_words]
	if (len(tokenized_query) == 0):
		tokenized_query = query_store[:]
	query = [stemmer.stem(words) for words in tokenized_query]


	# find out the cosine angle between the two vectors
	doc_score = {}
	for q in query:
		if (q in terms):
			for j in terms[q].id_list:
				if (j in doc_score):
					doc_score[j] += (terms[q].tfid[j] / doc_normalization[j])
				else:
					doc_score[j] = (terms[q].tfid[j] / doc_normalization[j])
	result = doc_score.items()
	#print result

	result = sorted(result, key=lambda x: x[1], reverse=True)
	
	print("The total no of results are ")
	print(len(result))
	for r in result:
		print(r[0])

	print ("total Running Time is" + str(time.time() - st))		
search()
