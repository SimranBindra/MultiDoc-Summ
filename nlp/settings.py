import os
import string
import sys
f1=open('rougeScript9s_2.xml','w')
f1.write("<ROUGE_EVAL version=\"1.5.5\">\n")
#count=61
folders=os.listdir("systems9s_2")
val1=1
for folder in folders:
	temp=str(folder[1:4])
	f1.write("<EVAL ID=\"")
	f1.write(temp)
	f1.write("\">\n")
	f1.write("<PEER-ROOT>\n")
	f1.write("/home/simran/Desktop/project/nlp project/nlp/systems9s_2/"+folder)
	f1.write("</PEER-ROOT>\n")
	f1.write("<MODEL-ROOT>\n")
	f1.write("/home/simran/Desktop/project/nlp project/nlp/models/"+folder+"\n")
	f1.write("</MODEL-ROOT>\n")
	f1.write("<INPUT-FORMAT TYPE=\"SEE\">\n")
	f1.write("</INPUT-FORMAT>\n")
	f1.write("<PEERS>\n")
	article= os.listdir("systems9s_2/"+folder)
	#for article in articles:
	print ('Reading articles/'+article[0])
	f1.write("<P ID=\"")
	temp1=str(val1)
	val1=val1+1
	f1.write(temp1)
	f1.write("\">"+article[0]+"</P>\n")
	#val=val+1;
	f1.write("</PEERS>\n")
	f1.write("<MODELS>\n")
	val=1
	articles=os.listdir("models/"+folder)
	for article in articles:
		print ('Reading articles/'+article)
		f1.write("<M ID=\"")
		temp1=str(val)
		f1.write(temp1)
		f1.write("\">"+article+"</M>\n")
		val=val+1;
	f1.write("</MODELS>\n")
	f1.write("</EVAL>\n\n")
	'''if temp==87:
		count+=2
	else:
		count=count+1'''
f1.write("</ROUGE_EVAL>")
f1.close()
