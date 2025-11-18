#导入工具包
import traceback
from logging import exception

from imutils import contours
import numpy as np
import argparse
import cv2
import imutils
from imutils.text import put_text

import myutils


#设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to input image")
ap.add_argument("-t", "--template", required=True,help="path to template OCR-A image")
args = vars(ap.parse_args())#vars官方推荐的，把参数转换成字典

#指定信用卡类型
FIRST_NUMBER={"3":"Amerian Express",
              "4":"Visa",
              "5":"Mastercard",
              "6":"Discover Card",
              "0":"wrong"}

print(args["image"])

#绘图显示
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#读取一个模板图像
img=cv2.imread(args["template"])
#cv_show("template",img)
#灰度图
ref=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv_show("ref",ref)
#二值图像
ref=cv2.threshold(ref,24,255,cv2.THRESH_BINARY_INV)[1]#大于10设为0，小于设为255
# cv_show("ref",ref)

#计算轮廓
refCnts,hierarchy=cv2.findContours(ref.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#只检测外轮廓，保留坐标点

# cv2.drawContours(img,refCnts,-1,(0,0,255),3)
cv_show("img",img)
print(len(refCnts))
for i, cnt in enumerate(refCnts):
    print(f"轮廓 {i} 的形状:", cnt.shape)
    # cv2.drawContours(img, cnt, -1, (0, 0, 255), 3)
    # cv_show("cnt", img)


refCnts=myutils.sort_contours(refCnts,method="left-to-right")[0]#排序，从左到右，上到下
digits={}

#遍历每一个轮廓
for(i,cnt) in enumerate(refCnts):
    #计算外接矩形并resize合适大小
    (x,y,w,h)=cv2.boundingRect(cnt)
    roi = ref[y:y+h,x:x+w]
    roi=cv2.resize(roi,(57,88))
    # cv_show("ROI",roi)
    #每个数字对应一个模板
    digits[i]=roi

#初始化卷积核
rectKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))#自己指定一个核
squareKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

#读取输入图像，预处理
image=cv2.imread(args["image"])
# cv_show("image",image)
image=myutils.resize(image,width=300)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# cv_show("gray",gray)

#礼帽操作，突出明亮的区域
tophat=cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,rectKernel)
# cv_show("tophat",tophat)

gradX=cv2.Sobel(tophat,cv2.CV_32F,1,0,ksize=1)

gradX=np.absolute(gradX)
(minVal,maxVal)=(np.min(gradX), np.max(gradX))
gradX=255*(gradX-minVal)/(maxVal-minVal)#归一化
gradX=gradX.astype("uint8")

print(np.array(gradX).shape)
# cv_show("gradX",gradX)

#通过闭操作（先膨胀再腐蚀），将数字连在一起
gradX=cv2.morphologyEx(gradX,cv2.MORPH_CLOSE,rectKernel)
# cv_show("gradX2",gradX)
#Thresh_otsy会自动寻找合适的阈值，适合双峰。需要把阈值参数设置为0
thresh=cv2.threshold(gradX,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
# cv_show("thresh1",thresh)

#再来一个闭操作
thresh=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,squareKernel)
# cv_show("thresh3",thresh)

#计算轮廓
threshCnts,hierarchy=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=threshCnts
cur_img=image.copy()
cv2.drawContours(cur_img,cnts,-1,(0,0,255),3)
# cv_show("cur_img",cur_img)
locs=[]

#遍历轮廓
for i,c in enumerate(cnts):
    #计算矩形
    (x,y,w,h)=cv2.boundingRect(c)
    ar=w/float(h)
    print(w,h,ar)
    #选中合适的比值
    if(ar>2.5 and ar<4.0):
        if(w>40 and w<60)and (h>10 and h<30):
            #留下符合的
            # locs.append(c)
            locs.append((x,y,w,h)) #只加找到的轮廓出来的矩形
#符合的从左到右排序
locs=sorted(locs,key=lambda x:x[0])
# locs = sorted(locs, key=lambda c: cv2.boundingRect(c)[0])
output=[]


try:
    #遍历每一个轮廓中的数字
    # for(i,c) in enumerate(locs):
    for (i, (gX, gY, gW, gH)) in enumerate(locs):
        # (gX, gY, gW, gH)=cv2.boundingRect(c)
        #初始化数字列表
        groupOutput=[]

        #根据坐标提取每一个组
        group=gray[gY-5:gY+gH+5,gX-5:gX+gW+5]#让边界扩大点
        # cv_show("group",group)

        #预处理
        group=cv2.threshold(group,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
        # cv_show("group2",group)
        #计算每一组的轮廓
        digitCnts,hierarchy=cv2.findContours(group.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        digitCnts=contours.sort_contours(digitCnts,method="left-to-right")[0]
        #计算每一个组的每一个数值
        for c in digitCnts:
            #找到当前数字的轮廓，resize合适的大小
            (x,y,w,h)=cv2.boundingRect(c)
            roi=group[y:y+h,x:x+w]
            roi=cv2.resize(roi,(57,88))
            # cv_show("roi",roi)
            #计算匹配得分
            scores=[]

            #在模板中计算每一个得分
            for (digit,digitROI) in digits.items():
                #模板匹配
                result=cv2.matchTemplate(roi,digitROI,cv2.TM_CCOEFF)
                (_,score,_,_)=cv2.minMaxLoc(result)
                scores.append(score)

            #得到最合适的数字
            groupOutput.append(str(np.argmax(scores)))
        #在每组框，用红线画出来
        cv2.rectangle(image,(gX-5,gY-5),(gX+gW+5,gY+gH+5),(0,0,255),1)
        cv2.putText(image,"".join(groupOutput),(gX,gY-15),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)
        #得到结果
        output.extend(groupOutput)#把groupOutput加入到output中。output扩展groupOutput

    cv_show("image",image)
    # 打印信用卡类型
    print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
    # 打印完整的信用卡号
    print("Credit Card #: {}".format("".join(output)))


except Exception as e:
    print("错误信息:", e)
    traceback.print_exc()  # 打印完整的堆栈，包括行号

