X_train=[]
y_train=[]
pkl=open('/home/sharique/minor/dataset/training.pkl','rb')
total=pickle.load(pkl)
for item in total:
    X_train.append(item[0])
    y_train.append(item[1])

pkl.close()
