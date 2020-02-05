import pytesseract
from PIL import Image
import cv2

img = Image.open('photo_2.jpg')
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray_image,(5,5),0)
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'
result = pytesseract.image_to_string(blur)

with open("result.txt", mode ="w") as file:
 file.write(result)
 print("Check the result.txt")

