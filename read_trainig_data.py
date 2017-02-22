import cPickle as pickle

f=open("imageclassification.pkl","rb")
data=pickle.load(f);

X_train=[] #training image names  in X_train
Y_train=[] #trainig image labels in Y_train

X_val=[] #validation image names  in X_val
Y_val=[] #valdation image labels in Y_val

X_test=[] #test image names  in X_test
Y_test=[] #test image labels in Y_test

index =0
str1 = "No Indexing"

for value in data['img_detail']:
	if value :
		if str1.find(value['major'][0]) :
			X_train.append(data['img_name'][index])
			Y_train.append(value['major'][0])
	index+=1

count =1 
for i in X_train:
	print count,i + "train"
	count+=1
'''for i in X_val:
	print count,i + "val"
	count+=1
for i in X_test:
	print count,i + "test"
	count+=1
'''

f.close()