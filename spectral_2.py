#this is for generating summary of 200 words, threshold for inclusion of sentence in problem is 0.2, for inclusion of sentence in summary is 0.8 and target is to minimise summation of inverse_sentence_score*value(weighted)*leng where sentence score is calculated by its words' importance, and threshold for putting a fragment in a cluster is 0.07,extra stop words included.

from __future__ import division, unicode_literals
import nltk
import itertools
import string
from operator import itemgetter
#import networkx as nx
import os
import sys
import numpy as np
#from networkx.utils import *
from nltk.corpus import wordnet as wn
from itertools import product
import math
import time
import copy
import numpy as np
#import matlab
#import matlab.engine
import scipy.cluster.vq as CL
from scipy import linalg as LA
from sklearn.cluster import spectral_clustering
import nltk
import itertools
import re, math
import copy
from collections import Counter
import string
from operator import itemgetter
import networkx as nx
import os
import sys
import numpy as np
from networkx.utils import *
from nltk.corpus import wordnet as wn
from itertools import product
import math
import time
from pulp import *

WORD = re.compile('\w+')
total_words=0
def ltt(matrix,n,sentences,frags,folder,dict1):
	st2=time.time()
	s1=matrix
	lim=len(s1[0])
	for i in range(lim):
                s1[i][i]=0
	data=copy.deepcopy(s1)
	'''
	for i in range(lim):
		for j in range(lim):
			data[i][j]=s1[i][j]*(-1)
	for i in range(lim):
		for j in  range(lim):
			data[i][j]=math.exp(data[i][j])
	'''
	m=20
	s2=np.array(data)
	IDX = spectral_clustering(s2, n_clusters=m, n_components=None, eigen_solver=None, random_state=None, n_init=10, eigen_tol=0.0, assign_labels='kmeans')
	#for i in range(0,len(IDX)):
		#print(IDX[i])
	cluster=[None for _ in range(m)]
	for i in range(0,len(IDX)):
		if(cluster[IDX[i]]==None):
			l=[]
			l.append(i)
			cluster[IDX[i]]=l
		else:
			cluster[IDX[i]].append(i)
	n = len(frags)
	m = len(sentences) 
	sent_score=[None for _ in range(m)]
	b = [[None for _ in range(m)] for _ in range(n)]
	for i in range(n):
		for j in range(m):
			b[i][j]=0
	x=0
	y=0
	#f=0
	for first in frags :
		for second in sentences :
			wordTokens1=nltk.word_tokenize(first)
			wordTokens2=nltk.word_tokenize(second)
					
			#log returns natural log
			idf_Ti=0.000000000000001
			idf_Tj=0.000000000000001
			idf_comm=0.000000000000001
			for i in wordTokens1:
				idf1= math.log(n/float(1+n_containing2(i,dict1)))
				idf_Ti+=idf1
				if i in wordTokens2:
					idf_comm+=idf1
			
			'''if f == 0:
				sent_score[y]=0.000000000000001
				for i in wordTokens2:
					sent_score[y]=sent_score[y]+n_containing(i,dict1)
				sent_score[y]=total_words/float(sent_score[y])'''
			
		#Assigning values and calculte max,min
			b[x][y]=float(idf_comm)/idf_Ti
			y=y+1
		y=0
		#f=1
		x=x+1	

	#print "Generating TE score matrix to " + 'TEscoresMonz2/' +folder+'/'
	'''with open('C:/nlp project/simvipfragmenting/TEscoresMonz2/'+folder+'score.txt','w') as LM:
		print>>LM, 's1[\n'
		for i in range(n):
			for j in range(m):
				print>>LM,'{:.4f}'.format(float(d[i][j]))+' \n'
		print>>LM,'\n'
	'''
	for i in range(0,len(b[0])):
		sent_score[i]=0
		for j in range(0,len(b)):
			sent_score[i]+=b[j][i]
	print("...%s seconds..." %(time.time()-st2))
	set_cover(b,n,m,cluster,sentences,folder, sent_score)
	
	
def extractSentences(sentences,frags,folder,dict1,dict2):
	st1=time.time()
	nodePairs = list(itertools.combinations(frags, 2))
	n=len(frags)
	d = [[None for _ in range(n)] for _ in range(n)]
	for i in range(n):
		for j in range(n):
			d[i][j]=0
	#maximum=0
	#minimum=sys.maxint+1
	for pair in nodePairs:
		firstString = pair[0]
		secondString = pair[1]
		wordTokens1=nltk.word_tokenize(firstString)
		wordTokens2=nltk.word_tokenize(secondString)
		idf_Ti=0
		idf_Tj=0
		idf_comm=0
	   # print "\nWUP Scores- NOUNS\n"
		for i in wordTokens1:
			var1=float(1+n_containing(i,dict2))
			if var1<=n :
				idf1= math.log(n/var1) #if den>num causes negative values
			else:
				idf1= math.log(n/float(n_containing(i,dict2))) # or not to consider in idf1 since num will be =  to den 
			idf_Ti+=idf1
			if i in wordTokens2:
				idf_comm+=idf1
		for j in wordTokens2:
			var2=float(1+n_containing(j,dict2))
			if var2<=n:
				idf2= math.log(n/var2)
			else:
				idf2= math.log(n/float(n_containing(j,dict2)))
			idf_Tj+=idf2       
		for i in range(n):
			if frags[i]==firstString:
				x=i 
			if frags[i]==secondString:
				y=i
		var1=float(idf_Ti)
		var2=float(idf_Tj)
		if var1==0 and var2==0:
			d[x][y]=1
			d[y][x]=1
		if var1==0 and var2!=0:
			d[x][y]=1
			d[y][x]=idf_comm/var2
		if var2==0 and var1!=0:
			d[x][y]=idf_comm/var1 
			d[y][x]=1
		if var1!=0 and var2!=0:
			d[x][y]=idf_comm/var1    
			d[y][x]=idf_comm/var2
  
	#print "generated matrix"
	for i in range(n):
		for j in range(n):
			d[i][j]=float("{:.6f}".format(d[i][j]))
	print("...%s seconds..." %(time.time()-st1))
	ltt(d,n,sentences,frags,folder,dict1)	


def n_containing2(word,dict1):
	if(dict1.get(word)==None):
		return 0
	return dict1[word]


def set_cover(b,q,l,cluster,sentences,folder, sent_score):
	st3=time.time()
	values = [None for _ in range(l)]
	summary=[]
	x=0
	y=0
	#cno=1
	#slen=0
	for c in cluster :
		#r=slen
		minisummary=[]
		n=len(c)
		m=len(sentences)
		#print('len(c) : ',n)
		for i in range(m):
			values[i] = LpVariable("x"+str(i),0,1)
		prob = LpProblem("problem",LpMinimize)
	
		#creating the problem statement for linear programming.
		z= None
		first = 1;	
		for i in range(n):
			for j in range(m):
				if b[i][j]> 0.1 :    
					if first == 1 :
						z = values[j]
						first=0
					else :
						z = z + values[j]
			if first==0:
				#print(z, type(z), type(values[i]+values[j]))
				prob+= z>=1
			first =1
			z=None

		#leng = [None for _ in range(m)]
		#for i in range(m):
		#	leng[i]= len( sentences[i].split(' '))
		first = sent_score[0]*values[0]
		for i in range(1,m):
			first = first + sent_score[i]*values[i]
			prob +=first

		status = prob.solve(GLPK(msg=0))
		for i in range(m):
			#if value(values[i])>=0.95:
			minisummary.append((value(values[i]),sentences[i]))
		minisummary.sort(reverse=True)                
		no_of_words=(n/float(q))*(200)
		a=0
		flag=0
		for i in minisummary:
			if y>=200:
				break
			if flag==1 and a+len(i[1].split(" "))>abs(no_of_words+x):
				break
			if(not i[1] in summary):
				summary.append(i[1])
				flag=1
				a=a+len(i[1].split(" "))	
				y+=len(i[1].split(" "))
		x=no_of_words+x-a
		print('no_of_words and a and x and y :',no_of_words,a,x,y)
	LD = open('simvipfragmenting/Summary9sn_2/'+folder +'/summary.txt', 'w')
	for i in summary:
		LD.write(i+'\n')                
	LD.close();
	print("...%s seconds..." %(time.time()-st3))
	

def n_containing(word,dict1):
	if(dict1.get(word)==None):
		return 0
	return dict1[word]
	
	
def get_cosine(vec1, vec2):
	 #print(vec1.keys(), vec2.keys())
	 intersection = set(vec1.keys()) & set(vec2.keys())
	 #print(intersection)
	 numerator = sum([vec1[x] * vec2[x] for x in intersection])
	 sum1 = sum([vec1[x]**2 for x in vec1.keys()])
	 sum2 = sum([vec2[x]**2 for x in vec2.keys()])
	 #print("sum1 sum2 ",sum1,sum2)
	 denominator = math.sqrt(sum1) * math.sqrt(sum2)
	 #print("denominator ", denominator)
	 if not denominator:
		 return 0.0
	 else:
		 return float(numerator) / denominator

def text_to_vector(text):
	 #print(text)
	 words = WORD.findall(text)
	 #print(words)
	 return Counter(words)
#Remove the words before quotes(") , and creates a fragment of words in between the quotes.     
def quotes(s):
	if '"' not in s or '``' not in s:
		return s
	n = ""
	save = 0
	i = 0
	while(i < len(s)):
		if s[i] == '"' or s[i] == '``'  :
			save = i
			i += 1
			if(i > len(s)-1):
				break
			while s[i] != '"' and s[i] != '``' and i < len(s)-1 :
				if(i > len(s)-1):
					break
				i += 1
			i += 1
			n += s[save:i] + " *** "
			save = i
		else: 
			i += 1
	n += s[save:]
	return n
start_time=time.time()
folders=os.listdir("OriginalDocs1")
#if not os.path.exists("simvipfragmenting/TEscoresMonz2"):
#	os.mkdir("simvipfragmenting/TEscoresMonz2")
if not os.path.exists("simvipfragmenting/Summary9sn_2"):
	os.mkdir("simvipfragmenting/Summary9sn_2")
for folder in folders:
	if not os.path.exists("simvipfragmenting/Summary9sn_2/"+folder):
		os.mkdir("simvipfragmenting/Summary9sn_2/"+folder)
	print(folder)
	st=time.time()
	#if not os.path.exists("C:/nlp project/simvipfragmenting/TEscoresMonz2/"+folder):    
	#	os.mkdir("C:/nlp project/simvipfragmenting/TEscoresMonz2/"+folder)
	#if os.path.exists('OriginalDocs1/'+folder+'/'+'fragments3.txt'):
		#os.remove('OriginalDocs1/'+folder+'/'+'fragments3.txt')
	#if os.path.exists('OriginalDocs1/'+folder+'/'+'cluster.txt'):
		#os.remove('OriginalDocs1/'+folder+'/'+'cluster.txt')
	articles = os.listdir("OriginalDocs1/"+folder)
	NFrags = []
	sentences=[]
	dict1={}
	dict2={}
	for article in articles:
		#print ('Reading articles/' + article)
		articleFile = open('OriginalDocs1/'+folder+'/'+article, 'r')
		text = articleFile.read()
		sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
		frags = sent_detector.tokenize(text)
		sentences += sent_detector.tokenize(text)
		for i in frags:
			new = i
	
		#fragment them, remove these words.
			new = new.replace(" and","***")
			new = new.replace(" but","***")
			new = new.replace(", ","***")
			new = new.replace(" either ","***")
			new = new.replace(" or ","***")
			new = new.replace(" neither ","***")
			new = new.replace(" nor ","***")
			new = new.replace(" yet ","***")
			new = new.replace(";","***")
			new = new.replace(":","***")
			new = new.replace(" as far as i know ","***")
			new = new.replace("frankly speaking","***")
			new = new.replace(" so ","***")
			new = new.replace(" for example","***")
			new = new.replace(" for instance","***")
			new = new.replace(" as an example","***")
			new = new.replace(" to illustrate","***")
			new = new.replace(" as an illustration","***")
			new = new.replace(" not only","***")
			new = new.replace(" moreover","***")
			new = new.replace(" furthermore","***")
			new = new.replace(" in addition to","***")
			new = new.replace(" in addition","***")
			new = new.replace(" likewise","***")
			new = new.replace(" similarly","***")
			new = new.replace(" as well as","***")
			new = new.replace(" most probably","***")
			new = new.replace(" just in case","***")
			new = new.replace(" as soon as possible","***")
			new = new.replace(" on the other hand","***")
			new = new.replace(" in contrast to","***")
			new = new.replace(" as much as ","***")
			new = new.replace(" nevertheless","***")
			new = new.replace(" even so","***")
			new = new.replace(" even though","***")
			new = new.replace(" although","***")
			new = new.replace(" despite","***")
			new = new.replace(" so ","***")
			new = new.replace(" as a result","***")
			new = new.replace(" therefore","***")
			new = new.replace(" thus","***")
			new = new.replace(" as a consequence","***")
			new = new.replace(" consequently","***")
			new = new.replace(" in conclusion","***")
			new = new.replace(" in summary","***")
			new = new.replace(" finally","***")
			new = new.replace(" meanwhile","***")
			new = new.replace(" whereas","***") 
			 
			##breaking before
			new = new.replace(" such as","*** such as") 
			new = new.replace(" namely","*** namely") 
			new = new.replace(" specifically","*** specifically") 
			new = new.replace(" when","*** when") 
			new = new.replace(" while","*** while") 
			new = new.replace(" which","*** which") 
			new = new.replace(" who","*** who")
			new = new.replace(" whose","*** whose")        
			new = new.replace(" where","*** where")
			new = new.replace(" whose","*** whose")
			new = new.replace(" because","*** because")
			new = new.replace(" even if","*** even if")
			new = new.replace(" as if","*** as if")
			new = new.replace(" as long as","*** as long as")
			new = new.replace(" now that","*** now that")
			new = new.replace(" rather than","*** rather than")
			new = new.replace(" whenever","*** whenever")
			new = new.replace(" while","*** while")
			 
			 
			#brackets
			new = new.replace("(","***(")
			new = new.replace(")",")***")
			
			#double quotes
			new = quotes(new)
			NFrags += new.split("***")

	#i = 0
	#frags = NFrags[:]
	
	'''LD = open('/home/simran/Desktop/project//nlp project//OriginalDocs1//'+folder+'//'+'sentences.txt', 'w')
	LD.write('sentences=[/n')
	for i in sentences:
		LD.write(i)
		LD.write('/n')
	LD.write(']')
	LD.close()'''
	i=0
	while(i < len(NFrags)):
	#remove empty words to count the number of words
		NFrags[i]=NFrags[i].strip(' \t\n\r')
		NFrags[i]=NFrags[i].replace('\t', "")
		NFrags[i]=NFrags[i].replace('\n',"")
		NFrags[i]=NFrags[i].replace('\r',"")
		'''if(len(NFrags[i])<2):
			print('1',NFrags[i],len(NFrags[i]))
			NFrags.remove(NFrags[i])
			print('2',NFrags[i], len(NFrags[i]))
			continue'''
		words = NFrags[i].split(' ')
		no = words.count("")
		for j in range(0,no):
			words.remove("")        
		no = words.count(" ")
		for j in range(0,no):
			words.remove(" ")
		' '.join(NFrags[i])
	#remove small fragments
		if(len(words) <= 2 ):
			NFrags.remove(NFrags[i])
			continue
	#remove fragments having ALL words starting with a capital letter
		else:
			flag = 0
			for word in words:
				if not word[0] < 'Z' or word[0] > 'A':
					flag += 1
			if flag <= 1:
				#if(NFrags[i] in frags):
				NFrags.remove(NFrags[i])
		i += 1	
	#LD = open('C:/nlp project/OriginalDocs1/'+folder+'/'+'fragments3.txt', 'w')
	#LD.write('s1=[/n')
	frags=NFrags[:]
	for i in frags:
		words=nltk.word_tokenize(i)
		for j in words:
			total_words+=1
			if(dict2.get(j)==None):
				dict2[j]=1
			else:
				dict2[j]=dict2[j]+1
	for i in sentences:
		words=nltk.word_tokenize(i)
		visited={}
		for j in words:
			if (visited.get(j)==None):
				if(dict1.get(j)==None):
					dict1[j]=1
				else:	
					dict1[j]+=1
				visited[j]=1	
	
	extractSentences(sentences,frags,folder,dict1,dict2)
	print("...%s seconds..." %(time.time()-st))
print("...%s seconds..." %(time.time()-start_time))
	#print d
	#e=[]
'''for i in range(0 to len(frags))
		   e[i]=0
		   for j in range(0 to len(frags))
			  e[i]+=d[i][j]
	e.sort(reverse=True)
	cluster=[]
	j=0
	threshold=0.05
	for i in range(0 to len(frags)) and j <=50
		 '''           
		
			
				
