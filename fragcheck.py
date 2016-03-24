

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

start_time=time.time()
#retrieve each of the articles
folders=os.listdir("/home/simran/Desktop/project/nlp project/OriginalDocs")
if not os.path.exists("/home/simran/Desktop/project/nlp project/simvipfragmenting/TEscoresMonz2"):
    os.mkdir("/home/simran/Desktop/project/nlp project/simvipfragmenting/TEscoresMonz2")
if not os.path.exists("/home/simran/Desktop/project/nlp project/simvipfragmenting/Summary2"):
    os.mkdir("/home/simran/Desktop/project/nlp project/simvipfragmenting/Summary2")
for folder in folders:
    if not os.path.exists("/home/simran/Desktop/project/nlp project/simvipfragmenting/TEscoresMonz2/"+folder):    
         os.mkdir("/home/simran/Desktop/project/nlp project/simvipfragmenting/TEscoresMonz2/"+folder)
    os.remove('/home/simran/Desktop/project/nlp project/OriginalDocs/'+folder+'/'+'fragments.txt')
    os.remove('/home/simran/Desktop/project/nlp project/OriginalDocs/'+folder+'/'+'fragments3.txt')
    articles = os.listdir("/home/simran/Desktop/project/nlp project/OriginalDocs/"+folder)
    NFrags = []
    for article in articles:
        print ('Reading articles/' + article)
        articleFile = open('/home/simran/Desktop/project/nlp project/OriginalDocs/'+folder+'/'+article, 'r')
        text = articleFile.read()
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        frags = sent_detector.tokenize(text)
        sentences = sent_detector.tokenize(text)
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
            NFrags.extend(new.split("***"))

    i = 0
    frags = NFrags[:]
    
    LD = open('/home/simran/Desktop/project/nlp project/OriginalDocs/'+folder+'/'+'fragments.txt', 'w')
    LD.write('s1=[\n')
    for i in frags:
        LD.write(i)
        LD.write('\n')
    LD.write(']')
    LD.close()
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
        '''no = words.count("\n")
        for j in range(0,no):
            words.remove("\n")'''
    #remove small fragments
        if(len(words) <= 2):
            #if(NFrags[i] in frags):
            frags.remove(NFrags[i])
    #remove fragments having ALL words starting with a capital letter
        else:
            flag = 0
            for word in words:
                if not word[0] < 'Z' or word[0] > 'A':
                    flag += 1
            if flag <= 1:
                #if(NFrags[i] in frags):
                frags.remove(NFrags[i])
        i += 1
    LD = open('/home/simran/Desktop/project/nlp project/OriginalDocs/'+folder+'/'+'fragments3.txt', 'w')
    LD.write('s1=[\n')
    k=0
    for k in frags:
        LD.write(k)
        LD.write('\n')
    LD.write(']')
    LD.close()
    
print("...%s seconds..." %(time.time()-start_time))
