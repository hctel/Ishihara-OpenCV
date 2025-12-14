import numpy as np
from sklearn.cluster import KMeans
import cv2

PROJECT_NAME = "Gabriel" # Because, during early development, the program read a "71" instead of a "74" on plate Ishihara_9.png, being colorlind in the same way as Gabriel, my boyfriend :3.

def detect_circles(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_8UC1)
    circles = cv2.HoughCircles(blur, method=cv2.HOUGH_GRADIENT, dp=1, minDist=1000, param1 = 60, param2 = 35, minRadius=30, maxRadius = 500)
    if circles is not None:
        #print(circles)
        for circle in circles[0,:]:
            circle = list(map(int, circle))
            cv2.circle(img, (circle[0], circle[1]), circle[2], (255, 255, 255), 2)
        # cv2.imshow("Detected Circles", img)
        x,y = img.shape[:2]
        mask = np.zeros((x,y), dtype=np.uint8)
        cv2.circle(mask, (circle[0], circle[1]), circle[2], (255, 255, 255), -1)
        img = cv2.bitwise_and(img, img, mask=mask)
        # cv2.imshow("Circle Masked Image", img)
    return img, circles

def clustering(img):
    #
    #
    # NOTE: This code was adapted from https://nrsyed.com/2018/03/29/image-segmentation-via-k-means-clustering-with-opencv-python/.
    #
    #
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img = img[:,:,[1]] # Keeps only the Cr channel (red difference)
    if len(img.shape) == 2:
        img.reshape(img.shape[0], img.shape[1], 1)
    reshaped = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    kmeans = KMeans(n_clusters=2, n_init=40, max_iter=500).fit(reshaped)
    clustering = np.reshape(np.array(kmeans.labels_, dtype=np.uint8),
    (img.shape[0], img.shape[1]))
    sortedLabels = sorted([n for n in range(2)],
        key=lambda x: -np.sum(clustering == x))
    kmeansImage = np.zeros(img.shape[:2], dtype=np.uint8)
    for i, label in enumerate(sortedLabels):
        kmeansImage[clustering == label] = 255 * i
    # cv2.imshow('Original vs clustered', kmeansImage)
    return kmeansImage
   
    # pixels = img.reshape((-1, 3))
    # pixels = np.float32(pixels)
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # ret,label,center=cv2.kmeans(pixels,3,None,criteria,10,cv2.KMEANS_PP_CENTERS)
    # A = pixels[label.ravel()==0]
    # B = pixels[label.ravel()==1]    
    
    # # Plot the data
    # plt.scatter(A[:,0],A[:,1])
    # plt.scatter(B[:,0],B[:,1],c = 'r')
    # plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
    # plt.xlabel('Height'),plt.ylabel('Weight')
    # plt.show()

## OLD FUNCTIONS, KEPT FOR INFO ONLY

def coulour_highlight(img):
    rgbcodes={}
    all_rgb_codes = img.reshape(-1, img.shape[-1])
    for rgb in all_rgb_codes:
        if rgb[0] == 0 and rgb[1] == 0 and rgb[2] == 0:
            continue
        rgbcodes[get_rgbstr_from_tuple(tuple(rgb))] = 1
    for rgb in all_rgb_codes:
        if rgb[0] == 0 and rgb[1] == 0 and rgb[2] == 0:
            continue
        rgbcodes[get_rgbstr_from_tuple(tuple(rgb))] += 1
    sorted_rgb = sorted(rgbcodes.items(), key=lambda item: item[1], reverse=True)
    most_common_rgb = sorted_rgb[0][0]
    print("Most common RGB:", most_common_rgb, "with count:", sorted_rgb[0][1])
    color = tuple(int(c) for c in most_common_rgb.split(":"))
    color_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
    print("Most common HSV:", color_hsv)
    if color_hsv[0] < 20:
        lower_bound = np.array([max(0, 0), 100, 100])
        upper_bound = np.array([min(179, color_hsv[0]+20), 255, 255]) 
    else:
        lower_bound = np.array([max(0, color_hsv[0]-20), 0, 0])
        upper_bound = np.array([min(179, color_hsv[0]+20), 255, 255])
    print(f"Bounds: Lower {uint8hsvtohsv(lower_bound)}, Upper {uint8hsvtohsv(upper_bound)}")
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    cv2.imshow("Highlighted Color RGB", mask)
    mask ^= 0xFF
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imshow("Masked Image RGB", cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))
    return mask

def coulour_highlight_hsv(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    all_hsv_codes = img_hsv.reshape(-1, img_hsv.shape[-1])
    hsvcodes={}
    for hsv in all_hsv_codes:
        if hsv[1] == 0 and hsv[2] == 0 or hsv[0] == 0 and hsv[1] == 0 and hsv[2] == 0:
            continue
        hsvcodes[get_rgbstr_from_tuple(tuple(hsv))] = 1
    for hsv in all_hsv_codes:
        if hsv[1] == 0 and hsv[2] == 0 or hsv[0] == 0 and hsv[1] == 0 and hsv[2] == 0:
            continue
        hsvcodes[get_rgbstr_from_tuple(tuple(hsv))] += 1
    sorted_hsv = sorted(hsvcodes.items(), key=lambda item: item[1], reverse=True)
    most_common_hsv = sorted_hsv[0][0]
    print("Most common HSV:", most_common_hsv, "with count:", sorted_hsv[0][1])
    color_hsv = tuple(int(c) for c in most_common_hsv.split(":"))
    if color_hsv[0] < 25:
        lower_bound = np.array([max(0, 0), 100, 100])
        upper_bound = np.array([min(179, color_hsv[0]+25), 255, 255]) 
    else:
        lower_bound = np.array([max(0, color_hsv[0]-25), 0, 0])
        upper_bound = np.array([min(179, color_hsv[0]+25), 255, 255])
    print(f"Bounds: Lower {uint8hsvtohsv(lower_bound)}, Upper {uint8hsvtohsv(upper_bound)}")
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    cv2.imshow("Highlighted Color HSV", mask)
    mask ^= 0xFF
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imshow("Masked Image HSV", cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))
    return mask

def coulour_highlight_bgr(img):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    rgbcodes={}
    all_rgb_codes = img.reshape(-1, img.shape[-1])
    for rgb in all_rgb_codes:
        if rgb[0] == 0 and rgb[1] == 0 and rgb[2] == 0:
            continue
        rgbcodes[get_rgbstr_from_tuple(tuple(rgb))] = 1
    for rgb in all_rgb_codes:
        if rgb[0] == 0 and rgb[1] == 0 and rgb[2] == 0:
            continue
        rgbcodes[get_rgbstr_from_tuple(tuple(rgb))] += 1
    sorted_rgb = sorted(rgbcodes.items(), key=lambda item: item[1], reverse=True)
    most_common_rgb = sorted_rgb[0][0]
    print("Most common RGB:", most_common_rgb, "with count:", sorted_rgb[0][1])
    color = tuple(int(c) for c in most_common_rgb.split(":"))
    color_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
    print("Most common HSV:", color_hsv)
    if color_hsv[0] < 20:
        lower_bound = np.array([max(0, 0), 100, 100])
        upper_bound = np.array([min(179, color_hsv[0]+20), 255, 255]) 
    else:
        lower_bound = np.array([max(0, color_hsv[0]-20), 0, 0])
        upper_bound = np.array([min(179, color_hsv[0]+20), 255, 255])
    print(f"Bounds: Lower {uint8hsvtohsv(lower_bound)}, Upper {uint8hsvtohsv(upper_bound)}")
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    cv2.imshow("Highlighted Color BGR", mask)
    mask ^= 0xFF
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imshow("Masked Image BGR", cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))
    return mask


def coulour_highlight_forcerbg(img, target_rgb):
    color = target_rgb
    color_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
    print("Most common HSV:", color_hsv)
    if color_hsv[0] < 20:
        lower_bound = np.array([max(0, 0), 100, 100])
        upper_bound = np.array([min(179, color_hsv[0]+20), 255, 255]) 
    else:
        lower_bound = np.array([max(0, color_hsv[0]-20), 0, 0])
        upper_bound = np.array([min(179, color_hsv[0]+20), 255, 255])
    print(f"Bounds: Lower {uint8hsvtohsv(lower_bound)}, Upper {uint8hsvtohsv(upper_bound)}")
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    cv2.imshow("Highlighted Color", mask)
    mask ^= 0xFF
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imshow("Masked Image", cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))

def get_rgbstr_from_tuple(rgbtuple):
    return "{}:{}:{}".format(rgbtuple[0], rgbtuple[1], rgbtuple[2])

def uint8hsvtohsv(hsv):
    h = int(360*(hsv[0]/255))
    s = int(100*(hsv[1]/255))
    v = int(100*(hsv[2]/255))
    return (h, s, v)

def dilate_erode(img, size):
    elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    eroded = cv2.dilate(img, elem)
    eroded = cv2.erode(eroded, elem)
    eroded = cv2.GaussianBlur(eroded, (size, size), 0)
    return cv2.bitwise_or(img, eroded)
    
    #return img

if __name__ == "__main__":
    print("Please run Runner.py only.")
