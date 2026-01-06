import cv2
from tkinter import filedialog
from tkinter import Tk

def cartoonize_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray_blur, 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY,
                               blockSize=9, C=9)
    color = cv2.bilateralFilter(img, d=9, sigmaColor=250, sigmaSpace=250)
    return cv2.bitwise_and(color, color, mask=edges)

root = Tk()
root.withdraw()  # Hide the main window

while True:
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    
    if not file_path:  # User cancelled
        break
        
    img = cv2.imread(file_path)
    if img is None:
        print("Error loading image")
        continue
        
    cartoon = cartoonize_image(img)
    
    cv2.imshow("Original", img)
    cv2.imshow("Cartoon", cartoon)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    save = input("Save this result? (y/n): ").lower()
    if save == 'y':
        output_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg")]
        )
        if output_path:
            cv2.imwrite(output_path, cartoon)
            print(f"Saved to {output_path}")

print("Done processing images")