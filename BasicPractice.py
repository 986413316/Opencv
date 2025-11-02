import cv2 # opencv 默认读取的格式是BGR， 其他读的都是RGB
import numpy as np
import matplotlib.pyplot as plt
from babel.dates import format_interval

from CommonFunc import CommonFunc

Common=CommonFunc()

# region 读写图片部分
# img=cv2.imread("images/Study.jpg",cv2.IMREAD_GRAYSCALE)#cv2.IMREAD_GRAYSCALE 灰色的     cv2.COLOR_BGR2RGB 彩色的
#
# # plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# # plt.axis('off')#坐标轴关了
# # plt.show()
# Common.showPic(img)
# cv2.imwrite("images/StudyChage.jpg",img)
# endregion

#region 读取视频流
# vd=cv2.VideoCapture("Videos/bi.mp4")
# if vd.isOpened():
#     open,frame=vd.read()#read返回的是turble 元组信息，(bool，图片)
# else:
#     open=False
#
# while open:
#     ret, frame=vd.read()
#     if frame is None:
#         break
#     if ret==True:
#         gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         cv2.imshow("Result",gray)
#         if cv2.waitKey(1) & 0xFF == 27:  #每10s退出，或者摁esc退出
#             break
# vd.release()
# cv2.destroyAllWindows()
#endregion

#region 截取和组合图片
# img=cv2.imread("images/Study.jpg")
# cat=img[0:50,0:200]#图片是高度，宽度
# cv2.imshow("Study", cat)
# print(cat.shape)#高度，宽度，通道
# cv2.waitKey(0)
# b,g,r=cv2.split(cat)#三个通道拆分出来
#
# #以灰度图显示出来：亮度表示红色强度，但不是红色图像。
# # 亮的地方红色值高，暗的地方红色值低
# cv2.imshow("r", r)#单独显示r通道的
# cv2.waitKey(0)
# img2=cv2.merge([b,g,r])
# cv2.imshow("Study", img2)
# cv2.waitKey(0)
#
# cur_img=cat.copy()
# cur_img[:,:,0]=0#：冒号表示取所有， 总体是高度宽度不变，b通道所有数变成0
# cur_img[:,:,1]=0#g通道所有数变成0
#
# #以彩色方式显示：真正的红色图像，红色区域会呈现红色，其他区域为黑色。
# cv2.imshow("Study", cur_img)
# cv2.waitKey(0)
#endregion

#region 外边缘填充
# img=cv2.imread("images/Study.jpg")
# print(img.shape)
# cat=img[552:792,542:692]#截取552-691高度的和 542-591宽的图片
# # cv2.imshow("cat", cat)
# # cv2.waitKey(0)  #设置要看的图片
#
# top_size,botton_size,left_size,right_size=(20,20,20,20)
#
#
# replicate=cv2.copyMakeBorder(cat,top_size,botton_size,left_size,right_size,borderType=cv2.BORDER_REPLICATE)#复制法，复制最边缘像素
# reflect=cv2.copyMakeBorder(cat,top_size,botton_size,left_size,right_size,borderType=cv2.BORDER_REFLECT)#反射法，
# reflect101=cv2.copyMakeBorder(cat,top_size,botton_size,left_size,right_size,borderType=cv2.BORDER_REFLECT_101)#反射法，
# wrap=cv2.copyMakeBorder(cat,top_size,botton_size,left_size,right_size,borderType=cv2.BORDER_WRAP)#外包装法
# constant=cv2.copyMakeBorder(cat,top_size,botton_size,left_size,right_size,borderType=cv2.BORDER_CONSTANT,value=1)#添加常数，
#
# target_size = (cat.shape[1], cat.shape[0])  # 宽，高
# Resize=cv2.resize(img,target_size)
# replicate = cv2.resize(replicate, target_size) #调整图像宽高
#
# plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.4, wspace=0.4)  #设plt的间距 hspace是上下，wspace是左右两个的
# plt.subplot(241),plt.imshow(cv2.cvtColor(Resize, cv2.COLOR_BGR2RGB)),plt.title("Resize") #subplot(231) 表示总共两行，三列，当前是第一个图
# plt.subplot(242),plt.imshow(cv2.cvtColor(cat, cv2.COLOR_BGR2RGB)),plt.title("Original") #subplot(231) 表示总共两行，三列，当前是第一个图
# plt.subplot(243),plt.imshow(cat,"gray"),plt.title("Gray")  # 三个逗号表示 把它们当作一个元组表达式，每个函数都会执行。 清晰一点。最好是不要逗号，一行一个
# plt.subplot(244),plt.imshow(replicate,"gray"),plt.title("Replicate")
# plt.subplot(245),plt.imshow(reflect,"gray"),plt.title("reflect")
# plt.subplot(246),plt.imshow(reflect101,"gray"),plt.title("reflect101")
# plt.subplot(247),plt.imshow(wrap,"gray"),plt.title("wrap")
# plt.subplot(248),plt.imshow(constant,"gray"),plt.title("constant")
# plt.show()
#endregion

#region 数值计算
# img=cv2.resize(cv2.imread("images/Study.jpg"),(150,220))
# print(img[0:2,:])
# #全部都加
# addimg=img+20
# print("add after")
# print(addimg[0:2,:])
#
# #两个图像相加。超过255， 256就变成1
# img+addimg
# #add 超过255，算255 256就变成255
# cv2.add(img,addimg)
# #按比例融合 y=ax1+bx2+c
# res=cv2.addWeighted(img,0.9,addimg,0.1,2)
#
#
# plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.4, wspace=0.4)  #设plt的间距 hspace是上下，wspace是左右两个的
# plt.subplot(2,2,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),plt.title('img')
# plt.subplot(2,2,2),plt.imshow(cv2.cvtColor(addimg, cv2.COLOR_BGR2RGB)),plt.title('addimg')
# plt.subplot(2,2,3),plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB)),plt.title('res')
# plt.show()
#endregion

#region 阈值处理
# img=cv2.resize(cv2.imread("images/Study.jpg",cv2.IMREAD_GRAYSCALE),(150,220))
#
# # ret表示转化与否，thresh1表示输出图 .127表示阈值是127，255表示设定的值
# ret,thresh1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)#. 大于阈值设为最大值，否则设为 0（常见的黑白二值化）
# ret,thresh2=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)#大于阈值设为 0，否则设为最大值
# ret,thresh3=cv2.threshold(img,127,255,cv2.THRESH_TRUNC)#大于阈值的像素设为阈值，其余保持原值
# ret,thresh4=cv2.threshold(img,127,255,cv2.THRESH_TOZERO)#小于阈值设为 0，其余保持原值
# ret,thresh5=cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)#大于阈值设为 0，其余保持原值
#
# tiles=["Orgin","Binary","Binary_inv","Trunc","ToZero","ToZero_inv"]
# image=[img,thresh1,thresh2,thresh3,thresh4,thresh5]
#
# for i in range (len(image)):
#     plt.subplot(2,3,i+1),plt.imshow(image[i],"gray")
#     plt.title(tiles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()

#endregion

#region 平滑处理
# img=cv2.resize(cv2.imread("images/Study.jpg",cv2.IMREAD_GRAYSCALE),(150,220))
#
# blur=cv2.blur(img,(3,3))#均值滤波，把一个元素周围3*3的元素取平均作为它的值
# box=cv2.boxFilter(img,-1,(3,3),normalize=True)#和均值滤波一样
# box2=cv2.boxFilter(img,-1,(3,3),normalize=False)#把一个元素周围3*3的元素的和作为它的值，大于255取255作为它的值
# aussian=cv2.GaussianBlur(img,(5,5),1)#高斯滤波，卷积核符合高斯分布，1更重视中间的。 越接近0表示越重视中间的值，绝对值越大越模糊
# median=cv2.medianBlur(img,5)#中值滤波，选取中间值为中心元素的值 （好像有效一点）
#
# res=np.hstack((img,blur,box,box2,aussian,median))
# cv2.imshow("unit",res)
# cv2.waitKey(0)
cv2.destroyAllWindows()

#endregion

#region 腐蚀操作

# img=cv2.imread("images/腐蚀操作.jpg")
# #核多大，腐蚀的越快。
# Kernel=np.ones((3,3),np.uint8)#NumPy 创建一个 5×5 的卷积核（Kernel）,所有元素都是1，数据类型是无符号8位数
# erosion=cv2.erode(img,Kernel,iterations=1) #迭代次数为1
# res=np.hstack((img,erosion))
# cv2.imshow("unit",res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# pie=cv2.imread("Images/腐蚀操作2.jpg")
# Kernel2=np.ones((30,30),np.uint8)
# errosion_1=cv2.erode(pie,Kernel2,iterations=1)
# errosion_2=cv2.erode(pie,Kernel2,iterations=2)
# errosion_3=cv2.erode(pie,Kernel2,iterations=3)
# res2=np.hstack((errosion_1,errosion_2,errosion_3))
#
# cv2.imshow("unit2",res2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#endregion

#region 膨胀操作
# img=cv2.imread("images/腐蚀操作.jpg")
# Kernel=np.ones((3,3),np.uint8)#NumPy 创建一个 5×5 的卷积核（Kernel）,所有元素都是1，数据类型是无符号8位数
#
# erosion=cv2.erode(img,Kernel,iterations=1) #迭代次数为1
# dig_dilate=cv2.dilate(erosion,Kernel,iterations=1)#膨胀操作
#
# res=np.hstack((img,erosion,dig_dilate))
# cv2.imshow("unit",res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#endregion
#region 膨胀腐蚀的一些复合运算
# img=cv2.imread("images/腐蚀操作.jpg")
# Kernel=np.ones((3,3),np.uint8)#NumPy 创建一个 5×5 的卷积核（Kernel）,所有元素都是1，数据类型是无符号8位数
#
# erosion=cv2.morphologyEx(img,cv2.MORPH_OPEN,Kernel,iterations=1)#先腐蚀后膨胀
# erosion=cv2.morphologyEx(img,cv2.MORPH_CLOSE,Kernel,iterations=1)#先膨胀后腐蚀
# erosion=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,Kernel,iterations=1)#梯度运算，膨胀减去腐蚀得到边缘轮廓
# erosion=cv2.morphologyEx(img,cv2.MORPH_TOPHAT,Kernel,iterations=1)##礼帽=原始输入-开运算结果,获得刺
# erosion=cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,Kernel,iterations=1)#黑帽，闭运算-原始输入
#
# cv2.imshow("unit",erosion)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#endregion

#region 图像梯度计算
# pie=cv2.imread("Images/Study.jpg",cv2.IMREAD_GRAYSCALE)
#
# dst=cv2.Sobel(pie,-1,1,0,ksize=3)# 进行X方向的梯度计算，核是3。 -1表示输入深度和输出深度是一样的 ，减法得到的负数会被截断成0，
#
# dst2=cv2.Sobel(pie,cv2.CV_64F,1,0,ksize=3)# 减法得到的负数会被截断成0，cv2.CV_64F 这里表示，保留负数 .和上面显示的差不多，只是保留了数值
# sobelx=cv2.convertScaleAbs(dst2)#变成绝对值
#
# dst3=cv2.Sobel(pie,cv2.CV_64F,0,1,ksize=3)# 减法得到的负数会被截断成0，cv2.CV_64F 这里表示，保留负数 .和上面显示的差不多，只是保留了数值
# sobely=cv2.convertScaleAbs(dst3)#变成绝对值
#
# sobelxy=cv2.addWeighted(sobelx,0.5,sobely,0.5,0)#xy和在一起，权重都是0.5.相当于 y=ax1+bx2+c
# wrong=dst=cv2.Sobel(pie,-1,1,1,ksize=3)#不建议这样
#
#
# tiles=["Orgin","Noabs","absX","absY","AddWeight","wrong"]
# image=[pie,dst,sobelx,sobely,sobelxy,wrong]
#
# plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.4, wspace=0.4)  #设plt的间距 hspace是上下，wspace是左右两个的
# for i in range (len(image)):
#     plt.subplot(2,3,i+1),plt.imshow(image[i],"gray")
#     plt.title(tiles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()

#
# img=cv2.resize(cv2.imread("images/Study.jpg",cv2.IMREAD_GRAYSCALE),(150,220))
# # Sobel算子
# sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
# sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
# sobelx = cv2.convertScaleAbs(sobelx)
# sobely = cv2.convertScaleAbs(sobely)
# sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
# #Scharr算子
# scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
# scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
# scharrx = cv2.convertScaleAbs(scharrx)
# scharry = cv2.convertScaleAbs(scharry)
# scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
# #laplacian 算子
# laplacian = cv2.Laplacian(img, cv2.CV_64F)
# laplacian = cv2.convertScaleAbs(laplacian)
#
# res = np.hstack((sobelxy, scharrxy, laplacian))
# Common.showPic(res)
#endregion

#region Canny边缘检测
# img=cv2.resize(cv2.imread("images/Study.jpg",cv2.IMREAD_GRAYSCALE),(150,220))
# v1=cv2.Canny(img,80,150)#极大极小值比较大
# v2=cv2.Canny(img,50,100)#极大极小值比较小
# res = np.hstack((v1, v2))
# Common.showPic(res)

#endregion

#region 轮廓检测
# img=cv2.imread("images/轮廓检测.jpg")
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# cv2.imshow("thresh",thresh)
# cv2.waitKey(0)
# #轮廓检索模式-检索索引，轮廓逼近模式-输出多边形（也可以输出顶点）
# #contours轮廓列表，每个轮廓是一个由点组成的 NumPy 数组，hierarchy轮廓之间的 层级关系
# contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
# #绘制轮廓
# draw_img=img.copy()#绘制轮廓是在原图基础上的
# #-1表示画出所有,其他表示第几个轮廓，用绿线画，线条宽度为3
# res=cv2.drawContours(draw_img,contours,-1,(0,255,0),3)
# Common.showPic(res)
# #轮廓特征
# cnt=contours[0]
# print(type(cnt))
# print(cv2.contourArea(cnt))#获得轮廓面积
# print(cv2.arcLength(cnt,True))#获得周长，true表示闭合的


#先进行阈值处理
# img=cv2.resize(cv2.imread("images/Study.jpg",cv2.IMREAD_GRAYSCALE),(150,220))
# re,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# Common.showPic(re)

#进行模板匹配
# img=cv2.imread("Images/Study.jpg",0)
# template=cv2.imread("Images/StudyTemplate.jpg",0)
# h,w=template.shape[:2]
#
# print(img.shape)
# print(template.shape)
# #下面的1可以用这些代替。NORMED表示加了归一化操作，尽量用这些
# methods=["cv2.TM_CCOEFF","cv2.TM_CCOEFF_NORMED","cv2.TM_CCORR","cv2.TM_CCORR_NORMED","cv2.TM_SQDIFF","cv2.TM_SQDIFF_NORMED"]
# res=cv2.matchTemplate(img,template,1)#模板匹配
# print(res.shape)
# min,max,min_loc,max_loc=cv2.minMaxLoc(res)#最小值，最大值，最小值坐标位置，最大值坐标位置
# print(min,max,min_loc,max_loc)
#

# for meth in methods:
#     img2=img.copy()
#
#     #匹配方法的真值
#     method=eval(meth)
#     print(method)
#     #会返回一个 匹配结果矩阵 res2，它是一个 灰度图像（二维数组），表示模板图像 template 在目标图像 img2 中的匹配程度。
#     res2=cv2.matchTemplate(img2,template,method)
#     min, max, min_loc, max_loc = cv2.minMaxLoc(res2)
#
#     #如果是平方差,或归一化，取最小值
#     if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#     buttom_right=[top_left[0]+w,top_left[1]+h]
#
#     #画矩形
#     cv2.rectangle(img2,top_left,buttom_right,(0,0,255),4)
#
#     plt.subplot(121),plt.imshow(res2,'gray')
#     plt.xticks([]), plt.yticks([])#隐藏坐标轴
#     plt.subplot(122),plt.imshow(img2,'gray' )
#     plt.xticks([]), plt.yticks([])
#     plt.suptitle(meth)
#     plt.show()

#匹配多个对象
# 读取原图和彩色模板
# img_rgb = cv2.imread('mario.jpg')                     # 原图（彩色）
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)  # 转为灰度图
# template = cv2.imread('mario_coin.jpg', 0)            # 模板图（灰度）
# h, w = template.shape[:2]                             # 获取模板尺寸
#
# # 执行模板匹配
# res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
#
# # 设置匹配阈值
# threshold = 0.8
# loc = np.where(res >= threshold)  # 找出匹配度大于阈值的位置
#
# # 在原图上画出所有匹配区域 loc[::-1]：反转坐标顺序.返回结果是yx，改完xy
# # *是解包操作，zip是将多个可迭代对象“打包”成一个个元素对
# for pt in zip(*loc[::-1]):        # 将坐标打包为 (x, y) 点
#     bottom_right = (pt[0] + w, pt[1] + h)
#     cv2.rectangle(img_rgb, pt, bottom_right, (0, 255, 0), 2)
#
# # 显示结果（使用 Matplotlib）
# plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.title('Template Matching Result')
# plt.show()

#金字塔变换
# mg=cv2.imread("Images/Study.jpg",0)
# cv2.imshow("Study",mg)
# cv2.waitKey(0)
# img=cv2.pyrDown(mg)#变小
# Common.showPic(img)
# img2=cv2.pyrUp(mg)#变大
# Common.showPic(img2)

#endrigion


#region 直方图和傅里叶变换
# img=cv2.imread("Images/Study.jpg",0)
# hist=cv2.calcHist([img],[0],None,[256],[0,256])#
# print(hist.shape)
#
# plt.plot(hist,'r')#以红色线画出来 直方图结果
# plt.show() #第一种方式，黑白图，用calcHist计算
#
# plt.hist(img.ravel(),256) # ravel()将图像数组 展平 成一维数组，把所有像素值拉成一条线
# plt.show()#第二种，直接把图像展平
#
# img=cv2.imread("Images/Study.jpg")
# color=("b","g","r")
# for i,col in enumerate(color):
#     histr=cv2.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr,color=col)
#     plt.xlim([0,256])
# plt.show()#第三种，彩色图像，用calcHist计算

#创建mask
# img=cv2.imread("Images/Study.jpg",0)
#
# mask=np.zeros(img.shape[:2],np.uint8)
# mask[100:300,100:400]=255 #创建掩码，白色的是掩码
# Common.showPic(mask)
#
# mask_img=cv2.bitwise_and(img,img,mask=mask)#与操作
# Common.showPic(mask_img)
#
# hist_full=cv2.calcHist([img],[0],None,[256],[0,256])
# hist_mask=cv2.calcHist([img],[0],mask,[256],[0,256])
#
# plt.subplot(221),plt.imshow(img,'gray')
# plt.subplot(222),plt.imshow(mask,'gray')
# plt.subplot(223),plt.imshow(mask_img,'gray')
# plt.subplot(224),plt.plot(hist_full),plt.plot(hist_mask)
# plt.xlim([0,256])
# plt.show()

#均衡化操作
img=cv2.imread("Images/Study.jpg",0)
plt.hist(img.ravel(),256)
plt.show()

equ=cv2.equalizeHist(img)
plt.hist(equ.ravel(),256)
plt.show()

#endregion
