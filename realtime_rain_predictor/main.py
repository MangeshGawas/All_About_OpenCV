from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import base64
import cv2
import numpy as np
import io
from PIL import Image

# Step 1: Set up Selenium to fetch the radar map
def get_radar_image():
    # Set up the Chrome WebDriver
    driver = webdriver.Firefox()

    try:
        # Open the weather radar page
        url = "https://weather.com/weather/radar/interactive/l/99d81196fbddf2082e857fca0200ed17a70fac5c2b71bce39542e94bc1527886"
        driver.get(url)

        # Wait for the canvas to be visible
        wait = WebDriverWait(driver, 150)  # Increase wait time as needed
        canvas = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'mapboxgl-canvas')))

        # Execute JavaScript to get the canvas data as a base64 string
        canvas_base64 = driver.execute_script("""
            var canvas = document.querySelector('.mapboxgl-canvas');
            return canvas.toDataURL('image/png').substring(22);
        """)
        # Decode the base64 string to an image
        canvas_data = base64.b64decode(canvas_base64)
        image = Image.open(io.BytesIO(canvas_data))
        #save this image 
        image.save('radar_image.png')  # Save the radar image as radar_image.png
        return image
    finally:
        driver.quit()

# Step 2: Image Processing with OpenCV
def process_image(image):
    # Convert the image to BGR format for OpenCV processing
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    print(opencv_image.shape)  # Print shape
    print(opencv_image)        # Print content

    # Convert to HSV color space
    hsv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)

    # Define the green color range in HSV
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    # Create a mask for green areas
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Calculate the percentage of green pixels
    green_percentage = (np.sum(mask > 0) / mask.size) * 100

    return opencv_image, mask, green_percentage

# Step 3: Predict Rainfall
def predict_rainfall(green_percentage):
    threshold = 20  # Define a threshold for green percentage
    if green_percentage > threshold:
        return "Rainfall is likely."
    else:
        return "No rainfall expected."

# Main function to execute the steps
def main():
    radar_image = get_radar_image()
    opencv_image, mask, green_percentage = process_image(radar_image)
    prediction = predict_rainfall(green_percentage)
    print(f"Green Percentage: {green_percentage:.2f}%")
    print(prediction)

    # Display the radar image and mask
    cv2.imshow('Radar Image', opencv_image)
    cv2.imshow('Green Areas', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
