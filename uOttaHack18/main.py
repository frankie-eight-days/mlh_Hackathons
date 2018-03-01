from PIL import Image, ImageFilter, ImageEnhance
import pytesseract
import cv2

def enhanceImage():
    photo = Image.open("tester.jpg")
    photo = photo.filter(ImageFilter.MedianFilter())
    photo = photo.filter(ImageFilter.GaussianBlur(3))

    contrast = ImageEnhance.Contrast(photo)
    photo = contrast.enhance(0.9)

    photo = photo.convert('L')
    photo.save('enhancedImage.jpg')

def binImage():
    photo = cv2.imread('enhancedImage.jpg')
    photoGrey = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    retval, thresh0 = cv2.threshold(photoGrey, 253, 255, cv2.THRESH_BINARY)
    gaus = cv2.adaptiveThreshold(photoGrey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    cv2.imwrite('binImage.jpg', gaus)

def imageToText():
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
    return pytesseract.image_to_string(Image.open('binImage.jpg'), lang='eng', boxes=False)


enhanceImage()
binImage()
imageText = imageToText()
print(imageText)