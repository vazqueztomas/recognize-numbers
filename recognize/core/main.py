import cv2
import pytesseract

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at {image_path}")
    return image

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return thresh_image

def extract_numbers(image):
    custom_config = r'--oem 3 --psm 5 -c tessedit_char_whitelist=0123456789'
    extracted_text = pytesseract.image_to_string(image, config=custom_config)
    numbers_list = [int(part) for part in extracted_text.split() if part.isdigit()]
    return numbers_list

def main(image_path):
    image = load_image(image_path)
    if image is not None:
        preprocessed_image = preprocess_image(image)
        numbers = extract_numbers(preprocessed_image)
        print(f"Numbers found in the image: {numbers}")

if __name__ == "__main__":
    image_path = "recognize/assets/tabla.jpeg"  # Replace with your image path
    main(image_path)