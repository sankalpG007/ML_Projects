import pandas as pd
import pyautogui
import pyperclip
import time
import webbrowser
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def send_whatsapp_messages():
    try:
        # Load Excel file and clean column names
        df = pd.read_excel("contact.xlsx")
        
        # Clean column names by stripping spaces
        df.columns = df.columns.str.strip()
        
        # Now use clean column names
        required_columns = ['phone', 'imagepath', 'audiopath']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Excel file must contain columns: {required_columns}. Found: {list(df.columns)}")

        print(f"Found {len(df)} contacts to process")
        
        # Use Selenium for better control
        driver = webdriver.Chrome()
        driver.get("https://web.whatsapp.com")
        
        # Wait for QR code scan
        print("Please scan the QR code within 60 seconds...")
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.XPATH, '//div[@contenteditable="true"][@data-tab="10"]'))
        )
        print("QR code scanned successfully!")

        for index, row in df.iterrows():
            try:
                phone = str(row['phone']).strip()  # No space now!
                image_path = row['imagepath']      # No space now!
                audio_path = row['audiopath']      # No space now!

                # Validate phone number
                if not phone.isdigit() or len(phone) < 10:
                    print(f"Invalid phone number: {phone}")
                    continue

                print(f"Processing contact: {phone}")
                
                # Open chat with contact
                url = f"https://web.whatsapp.com/send?phone={phone}"
                driver.get(url)
                time.sleep(8)

                # Send image if exists and valid
                if pd.notna(image_path) and os.path.exists(str(image_path)):
                    print(f"Sending image: {image_path}")
                    send_file(driver, str(image_path), "image")
                else:
                    print(f"Image not found or invalid: {image_path}")
                
                # Send audio if exists and valid
                if pd.notna(audio_path) and os.path.exists(str(audio_path)):
                    print(f"Sending audio: {audio_path}")
                    send_file(driver, str(audio_path), "audio")
                else:
                    print(f"Audio not found or invalid: {audio_path}")
                
                time.sleep(3)
                print(f"Completed processing: {phone}\n")
                
            except Exception as e:
                print(f"Error processing contact {phone}: {str(e)}")
                continue

    except Exception as e:
        print(f"Script failed: {str(e)}")
    finally:
        if 'driver' in locals():
            driver.quit()

def send_file(driver, file_path, file_type):
    """Helper function to send files using Selenium"""
    try:
        # Click attach button
        attach_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//div[@title="Attach"]'))
        )
        attach_btn.click()
        time.sleep(2)

        # File input element
        file_input = driver.find_element(By.XPATH, '//input[@type="file"]')
        file_input.send_keys(os.path.abspath(file_path))
        time.sleep(3)

        # Click send button
        send_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//span[@data-icon="send"]'))
        )
        send_btn.click()
        time.sleep(3)
        
    except Exception as e:
        print(f"Error sending {file_type} {file_path}: {str(e)}")

if __name__ == "__main__":
    send_whatsapp_messages()