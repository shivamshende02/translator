import cv2
import pytesseract

# Set the path to Tesseract OCR if needed (for Windows users)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Load the image
image = cv2.imread("C:\\Users\\Shivam\\Pictures\\Screenshots\\12.png")  # Replace with your image path

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply OCR with PSM 6 (Assume a single uniform block of text)
custom_config = r'--psm 6'
text = pytesseract.image_to_string(gray, config=custom_config)

# Remove newlines to get text in a single line
single_line_text = " ".join(text.split())

print(single_line_text)
