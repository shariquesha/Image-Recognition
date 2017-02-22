import pickle
import json
f=open("imageclassification.pkl","rb")
data=pickle.load(f);
index =0
for value in data['img_detail']:
	print data['img_name'][index] + " ", 
	print  value
	print "==> " 
	print index
	index+=1
# output=open("write.txt","w");
#output.write(str(data))
#output.flush()
#output.close()
