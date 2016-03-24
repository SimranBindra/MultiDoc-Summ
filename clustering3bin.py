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
from pulp import *


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
def n_containing(word, sentenceTokens):
	return sum(1 for sentence in sentenceTokens  if word in sentence)
start_time=time.time()
#retrieve each of the articles
folders=os.listdir("OriginalDocs")
if not os.path.exists("simvipfragmenting/TEscoresMonz2"):
	os.mkdir("simvipfragmenting/TEscoresMonz2")
if not os.path.exists("simvipfragmenting/Summary2"):
	os.mkdir("simvipfragmenting/Summary2")
for folder in folders:
	print(folder)
	if not os.path.exists("simvipfragmenting/"+folder):
		os.mkdir("simvipfragmenting/"+folder)
	if not os.path.exists("simvipfragmenting/TEscoresMonz2/"+folder):    
		os.mkdir("simvipfragmenting/TEscoresMonz2/"+folder)
	if os.path.exists('OriginalDocs/'+folder+'/'+'fragments3.txt'):
		os.remove('OriginalDocs/'+folder+'/'+'fragments3.txt')
	if os.path.exists('OriginalDocs/'+folder+'/'+'cluster.txt'):
		os.remove('OriginalDocs/'+folder+'/'+'cluster.txt')
	
	articles = os.listdir("OriginalDocs/"+folder)
	NFrags = []
	sentences=[]
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
	
	#LD = open('\home\simran\Desktop\project\nlp project\\OriginalDocs\\'+folder+'\\'+'fragments3.txt', 'w')
	#LD.write('s1=[\n')
	#for i in frags:
	#	LD.write(i)
	#	LD.write('\n')
	#LD.write(']')
	#LD.close()
	i=0
	while(i < len(NFrags)):
	#remove empty words to count the number of words
		words = NFrags[i].split(' ')
		no = words.count("")
		for j in range(0,no):
			words.remove("")        
		no = words.count(" ")
		for j in range(0,no):
			words.remove(" ")
		NFrags[i]=(' ').join(words)
	#remove small fragments
		if(len(words) <= 2):
			#if(NFrags[i] in frags):
			NFrags.remove(NFrags[i])
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
		
	'''LD = open('OriginalDocs\\'+folder+'\\'+'fragments3.txt', 'w')
	LD.write('s1=[\n')
	frags=NFrags[:]
	k=0
	for k in frags:
		LD.write(k)
		LD.write('\n')
	LD.write(']')
	LD.close()'''
	cluster=[]# cluster is list of clusters- which in turn is list of fragments
	histogram=[]#every element of histogram is list of hr_sim_total, hr_sim_above threshold(including threshold), hr_old(current histogram ratio) of corresponding cluster with 
	n= len(frags)#total number of fragments
	hr_limit=0.5
	print(len(frags))
	for i in frags:
		words=nltk.word_tokenize(i)
		idf_Ti=0.000000000000001
		if(len(cluster)==0):
			l=[]
			l.append(i)
			cluster.append(l)
			hist_el= [0,0,0]
		    #hist_el[0]=0
		    #hist_el[1]=0
		    #hist_el[2]=0
			histogram.append(hist_el)
			#print(histogram[len(histogram)-1])
		else:
			flag=0
			for j in range(0,len(cluster)):
				#print(j,histogram[j])
				hr_new=histogram[j][2]
				hr_new_aboveT = histogram[j][1]
				hr_new_total=histogram[j][0]
				for k in cluster[j]:
					words2 = nltk.word_tokenize(k)
					idf_comm=0.0
					idf_Tj=0.000000000000001
					for m in words2:
						idf2=math.log(float(1+n_containing(m,frags))/n)
						idf_Tj+=idf2
					for m in words : 
						idf1= math.log(float(1+n_containing(m,frags))/n)#number of fragments containing word i
						idf_Ti+=idf1
						if m in words2:
							idf_comm+=idf1	
					sim=2*idf_comm/float(idf_Ti+idf_Tj)
					if(sim>=0.5):
						hr_new_aboveT+=1
						hr_new_total+=1
					else:
						hr_new_total+=1
				hr_new=hr_new_aboveT/float(hr_new_total)
				if(hr_new>=histogram[j][2] and hr_new>=hr_limit):
					histogram[j][0]=hr_new_total
					histogram[j][1]=hr_new_aboveT
					histogram[j][2]=hr_new
					cluster[j].append(i)
					#print(i,'append in old cluster')
					flag=1
					break
			if(flag==0):
				#print(i,'append in new cluster')
				l=[]
				l.append(i)
				cluster.append(l)
				hist_el= [0,0,0]
				histogram.append(hist_el)
	print(len(cluster))			    
	with open('simvipfragmenting/'+folder+'/'+'cluster.txt','w') as LM:

		for j in range(0,len(cluster)):
			print('cluster ' + str(j), end="\n", file=LM)
			for k in cluster[j]:
				print(str(k) , end="\n", file=LM)
			print('', end="\n", file=LM)
			   # if(j==1):
				    #LM.write('sdadwedewewde')
	    
print("...%s seconds..." %(time.time()-start_time))
