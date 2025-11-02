import cv2
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i=1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]#用一个最小的矩形，把找到的形状抱起来
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b:b[1][i],reverse=reverse))
    return cnts,boundingBoxes

def resize(image,width=None,height=None,inter=cv2.INTER_AREA):
    dim=None
 # 获取原始图像尺寸
    (h, w) = image.shape[:2]

    # 如果没有指定宽度和高度，返回原图
    if width is None and height is None:
        return image

    # 如果只指定了宽度
    if width is not None and height is None:
        r = width / float(w)
        dim = (width, int(h * r))

    # 如果只指定了高度
    elif height is not None and width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    # 如果同时指定了宽度和高度
    else:
        dim = (width, height)

    # 缩放图像
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized