from scipy.misc import imread, imsave, imresize
path = "./dataset/CXR1_1_IM-0001-3001.png"
image = imread(path)

if(len(image.shape)<3):
      print 'gray'
elif len(image.shape)==3:
      print 'Color(RGB)'
else:
      print 'others'