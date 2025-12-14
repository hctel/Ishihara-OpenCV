import pytesseract as tess

tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Extracts a number from the provided image.
def getNumber(img):
    got = tess.image_to_string(img, lang="eng", config='--psm 10 --oem 3 digits')
    valid = len(got) > 0 and got[0].isdigit()
    if valid:
        return got.replace("\n","").replace(" ","")
    else:
        return None

if __name__ == "__main__":
    print("Please run Runner.py only.")