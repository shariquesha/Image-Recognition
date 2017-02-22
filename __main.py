import os
import requests
import urllib
import json
import pickle
import time

path='/home/sharq/NLMCXR_png_piiRemoved/'
if os.path.exists(path):
    print "true"
else:
    print "false"
img={}
img['img_detail']=[]
count=0
for dirpath,dirname,filenames in os.walk(path):
    img['img_name']=filenames
    #print "dirname list",dirname
for i,j in enumerate(img['img_name']):
    #print j[:j.rfind('.')]
    # if i==20:
        # break
    try:
        #print "holding for 10 sec"
        #time.sleep(20)
        link='https://openi.nlm.nih.gov/retrieve.php?img='+j[:j.rfind('.')]+'&query=&it=xg&coll=cxr&req=4'

        count=count+1
        #print count
        #print "Hitting..."
        info=requests.get(link)
        #print "Request get"
        img_json=json.loads(info.text)
        #img['img_detail']=[]
        #print img_json['list'][0]["MeSH"] 
        
        img['img_detail'].insert(i,img_json['list'][0]["MeSH"])
        print count,str(img['img_name'][i])+"==>"+str(img['img_detail'][i])
    except Exception, e:
        print e
        img['img_detail'].insert(i,None)
        print str(img['img_name'][i])+"==>"+str(img['img_detail'][i])
        #print "exception "
        #pass
# img["img_detail"]=img_detail"""

pkl=open('imageclassification.pkl','wb')
pickle.dump(img,pkl,pickle.HIGHEST_PROTOCOL)
# //pickle.dump(img)



#print img['img_detail']
