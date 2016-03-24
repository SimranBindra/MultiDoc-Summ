import os
import string
import sys
import time
import nltk

start_time=time.time()
folders=os.listdir("summaries")
if not os.path.exists("refSumHTM"):
	os.mkdir("refSumHTM")
for folder in folders:
	f=folder[:4]
	print(folder,f)
	#flag=0
	if not os.path.exists("refSumHTM/"+f):
		os.mkdir("refSumHTM/"+f)
	if not os.path.exists("summaries/"+folder+"/"+"200"):
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
		f2.close()
	f1=open("summaries/" + folder + "/" + "200", 'r')
	'''while(1):
		c=f1.read(1)
		if not c:
			break
		if(c=='<'):
			while(c!='>'):
				c=f1.read(1)
			pos=f1.tell()-1
			f1.seek(pos,0)
		break'''	
	#article = article.split(".")
	
	text = f1.read()
	sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
	sentences = sent_detector.tokenize(text)
	f2=open("refSumHTM/"+f +"/"+folder+"_200.htm", 'w')
	f2.write("<html>\n")
	f2.write("<head>\n")
	f2.write("<title>out.html</title>\n")
	f2.write("</head>\n")
	f2.write("<body bgcolor=\"white\">\n")
	lim=1
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
