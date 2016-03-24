#this is for generating summary of 200 words, threshold for inclusion of sentence in problem is 0.2, for inclusion of sentence in summary is 0.8 and target is to minimise summation of inverse_sentence_score*value(weighted)*leng where sentence score is calculated by its words' importance, and threshold for putting a fragment in a cluster is 0.07,extra stop words included.

from __future__ import division, unicode_literals
import nltk
import itertools
import re, math
from collections import Counter
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
from pulp import *

WORD = re.compile('\w+')
total_words=0
def extractSentences(sentences,frags,folder,cluster,dict1):
	n = len(frags)
	m = len(sentences) 
	sent_score=[None for _ in range(m)]
	d = [[None for _ in range(m)] for _ in range(n)]
	for i in range(n):
		for j in range(m):
			d[i][j]=0
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
				idf1= math.log(n/float(1+n_containing(i,dict1)))
				idf_Ti+=idf1
				if i in wordTokens2:
					idf_comm+=idf1
			
			'''if f == 0:
				sent_score[y]=0.000000000000001
				for i in wordTokens2:
					sent_score[y]=sent_score[y]+n_containing(i,dict1)
				sent_score[y]=total_words/float(sent_score[y])'''

			
		#Assigning values and calculte max,min
			d[x][y]=float(idf_comm)/idf_Ti
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
	for i in range(0,len(d[0])):
		sent_score[i]=0
		for j in range(0,len(d)):
			sent_score[i]+=d[j][i]
	set_cover(d,n,m,cluster,sentences,folder,sent_score)   
	#print("...%s seconds..." %((time.time()-st)/60))
	

def set_cover(b,q,l,cluster,sentences,folder, sent_score):
	
	values = [None for _ in range(l)]
	summary=[]
	x=0
	y=0
	for c in cluster :
		minisummary=[]
		n=len(c)
		m=len(sentences)
		for i in range(m):
			values[i] = LpVariable("x"+str(i),0,1)
		prob = LpProblem("problem",LpMinimize)
    
		#creating the problem statement for linear programming.
		z= None
		first = 1;	
		for i in range(n):
			for j in range(m):
				if b[i][j]> 0.05 :    
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

		leng = [None for _ in range(m)]
		for i in range(m):
			leng[i]= len( sentences[i].split(' '))
		first = sent_score[0]*values[0]*leng[0]
		for i in range(1,m):
			first = first + sent_score[i]*values[i]*leng[i]
			prob +=first

		status = prob.solve(GLPK(msg=0))
		#print(status)
		k=0;
		for i in range(m):
			if value(values[i])>=0.4:
				k=k+1
				minisummary.append((value(values[i]),sentences[i]))
		minisummary.sort(reverse=True)                
		no_of_words=(n/float(q))*(200)
		#print(no_of_sentences,n,q,l)
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
		#LD.write('\n\n\n\n\n')
	LD = open('simvipfragmenting/Summary10f_2/'+folder +'_reference.txt', 'w')
	for i in summary:
		LD.write(i+'\n')                
	LD.close();
	

def n_containing(word,dict1):
	if(dict1.get(word)==None):
		return 0
	return dict1[word]
	#return sum(1 for sentence in sentenceTokens  if word in sentence)
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
folders=os.listdir("OriginalDocs")
#if not os.path.exists("simvipfragmenting/TEscoresMonz2"):
#	os.mkdir("simvipfragmenting/TEscoresMonz2")
if not os.path.exists("simvipfragmenting/Summary10f_2"):
	os.mkdir("simvipfragmenting/Summary10f_2")
for folder in folders:
	print(folder)
	st=time.time()
	#if not os.path.exists("C:/nlp project/simvipfragmenting/TEscoresMonz2/"+folder):    
	#	os.mkdir("C:/nlp project/simvipfragmenting/TEscoresMonz2/"+folder)
	#if os.path.exists('OriginalDocs/'+folder+'/'+'fragments3.txt'):
		#os.remove('OriginalDocs/'+folder+'/'+'fragments3.txt')
	#if os.path.exists('OriginalDocs/'+folder+'/'+'cluster.txt'):
		#os.remove('OriginalDocs/'+folder+'/'+'cluster.txt')
	articles = os.listdir("OriginalDocs/"+folder)
	NFrags = []
	sentences=[]
	dict1={}
	for article in articles:
		#print ('Reading articles/' + article)
		articleFile = open('OriginalDocs/'+folder+'/'+article, 'r')
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
	
	'''LD = open('/home/simran/Desktop/project//nlp project//OriginalDocs//'+folder+'//'+'sentences.txt', 'w')
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
	#LD = open('C:/nlp project/OriginalDocs/'+folder+'/'+'fragments3.txt', 'w')
	#LD.write('s1=[/n')
	frags=NFrags[:]
	for i in frags:
		words=nltk.word_tokenize(i)
		for j in words:
			total_words+=1
			if(dict1.get(j)==None):
				dict1[j]=1
			else:
				dict1[j]=dict1[j]+1
	k=0
	'''for k in frags:
		LD.write(k)
		LD.write('/n')
	LD.write(']')
	LD.close()'''
	#print "dsadqdqwdqwd"
	d=[[None for x in range(len(frags))]for x in range(len(frags))]
	#print(len(frags))
	for i in range(0,len(frags)):
		for j in range(0,len(frags)):
			#print(i, j)
			vector1 = text_to_vector(frags[i])
			vector2 = text_to_vector(frags[j])
			#print(vector1, vector2)
			d[i][j] = get_cosine(vector1, vector2)
			#print (d[i][j])			
	#print('d made')
	e=[]
	print(len(frags))
	for i in range(0,len(frags)):
		sumi=0
		for j in range(0,len(frags)):
			sumi+=d[i][j]
		#if(i==1):
			#print(sumi)
		#print((i,sumi))		
		e.append((sumi,i))
	#f=e[:]
	e.sort(reverse=True )
	
	#for i in range(0,len(e)):
		#if(i[1]==0.0):
			#print(frags[i[0]], len(frags[i[0]]))
		#print(f[i],e[i])
	cluster=[]
	index=[]
	j=0
	threshold=0.11
	for i in e:
		#print(i)
		#if j>=37:
		#	break
		flag=0
		for k in cluster:
			for m in k:
				sim=d[m][i[1]]
				if sim>=threshold :
					flag=1
					break
		if flag==0 and i[0]>1:
			l=[]
			l.append(i[1])
			e.remove(i)
			cluster.append(l)
			j=j+1
	for i in e:
		maxi=-1
		maxsim=0
		for j in cluster:
			if maxsim<d[j[0]][i[1]]:
				maxi=j
				maxsim=d[j[0]][i[1]]
		if not maxi==-1:
			maxi.append(i[1])
	print('cluster length: ',len(cluster))    
	#with open('OriginalDocs/'+folder+'/'+'cluster.txt','w') as LM:
	#	for j in range(0,len(cluster)):
	#		    print>>LM,'cluster ' + str(j)+'\n'
	#		    for k in cluster[j]:
	#			    print>>LM,str(k)+'\n'
	#		    print>>LM,'\n'      
	extractSentences(sentences,frags,folder,cluster,dict1)
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
		
		    
			    
