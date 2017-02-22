import os

path = "./dataset/"

count = 0

for f in os.listdir(path):
	count+=1
	print count