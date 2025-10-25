import cv2 # opencv 默认读取的格式是BGR， 其他读的都是RGB
import numpy as np
import matplotlib.pyplot as plt



img=cv2.imread("images/Study.jpg")
cv2.imshow("Study",img)
cv2.waitKey(0)#等待按个键，去消失窗口
cv2.destroyAllWindows()
# plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# plt.axis('off')#坐标轴关了
# plt.show()
