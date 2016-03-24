import os
import string
import sys
import time
import nltk

start_time=time.time()
articles=os.listdir("simvipfragmenting/Summary9s_2")
if not os.path.exists("sysSumHTM9s_2"):
	os.mkdir("sysSumHTM9s_2")
for article in articles:
	f=article[:4]
	print(article,f)
	if not os.path.exists("sysSumHTM9s_2/"+f):
		os.mkdir("sysSumHTM9s_2/"+f)
	'''if not os.path.exists("summaries/"+folder+"/"+"200"):
		f2=open("summaries/"+folder+"/"+"200e",'r')
		f1=open("summaries/" + folder + "/" + "200", 'w')
		c=f2.read(1)
		while(1):
			if not c:
				break
			#c=f2.read(1)
			#print(c)
			if(c=='<'):
				a=f2.read(1)
				while(a!='>'):
					#print(a)
					a=f2.read(1)
			else :
				f1.write(c)
			c=f2.read(1)	
		f1.close()
		f2.close()'''
	#f1=open("simvipfragmenting/Summary2/" + article , 'r')
	'''while(1):
		c=f1.read(1)
		if not c:
			break
		if(c=='<'):
			while(f1.read(1)!='>'):
				continue
			
			break
	#article = article.split(".")'''
	f2=open("sysSumHTM9s_2/"+f +"/"+article[:len(article)-4]+".htm", 'w')
	f2.write("<html>\n")
	f2.write("<head>\n")
	f2.write("<title>out.html</title>\n")
	f2.write("</head>\n")
	f2.write("<body bgcolor=\"white\">\n")
	lim=1
	articleFile = open("simvipfragmenting/Summary9s_2/" + article , 'r')
	text = articleFile.read()
	sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
	sentences = sent_detector.tokenize(text)
	for line in sentences:
		#print lim,line
		if(line is None):
			break
		if line[0]!='\n':
			s=line
			temp=str(lim)
			f2.write("<a name=\"")
			f2.write(temp)
			f2.write("\"")
			f2.write(">")
			f2.write("[")
			f2.write(temp)
			f2.write("]")
			f2.write("</a> <a href=\"#")
			f2.write(temp)
			f2.write("\"")
			f2.write(" id=")
			f2.write(temp)
			f2.write(">")
			f2.write(s)
			pos=f2.tell()-1
			f2.seek(pos,0)
			f2.write("</a><br>\n")
			lim=lim+1
			f2.write("</a>\n")
		f2.write("</a>\n")
		f2.write("</body>\n")
		f2.write("</html>\n")
print("...%s seconds",time.time()-start_time)
