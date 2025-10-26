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
img=cv2.resize(cv2.imread("images/Study.jpg",cv2.IMREAD_GRAYSCALE),(150,220))

blur=cv2.blur(img,(3,3))#均值滤波，把一个元素周围3*3的元素取平均作为它的值
box=cv2.boxFilter(img,-1,(3,3),normalize=True)#和均值滤波一样
box2=cv2.boxFilter(img,-1,(3,3),normalize=False)#把一个元素周围3*3的元素的和作为它的值，大于255取255作为它的值
aussian=cv2.GaussianBlur(img,(5,5),1)#高斯滤波，卷积核符合高斯分布，更重视中间的
median=cv2.medianBlur(img,5)#中值滤波

n,m=(3,3)
b,a=((n-1)//2,(m-1)//2) #保证是整数，不是浮点数
height,width=img.shape

for y in range(height):
    for x in range(width):
        count=0
        for i in range(-a,a+1):
            for j in range(-b,b+1):
                count+=img[y+j,x+i]
        mid = count / (n * m)






#endregion