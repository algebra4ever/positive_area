from color_deconv import color_deconv
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('201901848P16_s.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred_img = cv2.GaussianBlur(gray_img, (5, 5), sigmaX=0)

DAB,HE,hed = color_deconv(img)
DAB = (DAB*255).astype(np.uint8)
HE = (HE*255).astype(np.uint8)
gray_DAB = cv2.cvtColor(DAB, cv2.COLOR_BGR2GRAY)
blurred_po = cv2.GaussianBlur(gray_DAB, (3, 3), sigmaX=1)


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl2 = clahe.apply(cv2.cvtColor(HE, cv2.COLOR_BGR2GRAY))
cl2_po = clahe.apply(blurred_po)

cl2_po = np.where(cl2_po > 200, 0, cl2_po)
cv2.imwrite('positive area.jpg',cl2_po)
s1 = np.count_nonzero(cl2_po)

cl2 = np.where(cl2 > 200, 0, cl2)
cv2.imwrite('HE area.jpg',cl2)
s2 = np.count_nonzero(cl2)
percentage = s1/s2 #proportion of positive area


X,Y,_ = np.shape(img)
m = np.arange(X)
n = np.arange(Y)
x,y = np.meshgrid(m,n)
z = hed[:, :, 2]
z = clahe.apply((z*255).astype(np.uint8))

# plot positive area 
z = np.where((z>200)|(z<10),np.NaN,z)
fig = plt.figure(figsize=(8,8))
ax1 = plt.axes(projection='3d')
ax1.plot_wireframe(x,y,z.T,rstride = 1, cstride = 1,cmap='rainbow')
ax1.contour(x,y,z.T,offset=-2, cmap = 'rainbow')
ax1.set_title('3D-wireframe')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

plt.show()

