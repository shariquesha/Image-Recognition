import Image
import os

path = "./dataset/"

count = 0
try:
	for f in os.listdir(path):
		#print f
		img = Image.open('./dataset/' + f)
		img.resize((150,150)).save('./dataset/' + f)
		count=count+1
		print count
except IOError:
		print "exception"