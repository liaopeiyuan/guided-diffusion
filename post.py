import cv2
import numpy as np 

d = np.load('/study-temp/alpaca/train_2/samples_100x32x32x3.npz')['arr_0']

for i in range(d.shape[0]):
    img = d[i,:,:,:]
    print(img)
    assert(cv2.imwrite(f'/study-temp/alpaca/train_2/results/{i}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))
