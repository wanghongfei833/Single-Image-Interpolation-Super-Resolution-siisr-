import cv2
img = cv2.imread("image.jpg")
img = cv2.resize(img,(64,64))
cv2.imwrite("image.jpg",img)