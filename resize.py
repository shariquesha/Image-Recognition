import Image
import os

path = "./dataset/"

count = 0

for f in os.listdir(path):
	try:
		img = Image.open('./dataset/' + f)
		img.resize((150,150)).save('./dataset/' + f)
		count=count+1
		print count
	except IOError:
		print "exception"