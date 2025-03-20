from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException


class BingImageSearch:
    def __init__(self):
        self.driver = None

    def setup_driver(self):
        options = Options()
        options.add_argument("--headless")
        self.driver = webdriver.Chrome(options=options)

    def scroll_to_bottom(self):
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            try:
                WebDriverWait(self.driver, 10).until(
                    lambda driver: driver.execute_script("return document.body.scrollHeight") > last_height
                )
            except TimeoutException:
                break  # If timeout occurs, break the loop
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def search_images(self, image_path):
        self.setup_driver()
        self.driver.get("https://www.bing.com/visualsearch")
        image_urls = []
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
            )
        except TimeoutException:
            print("Timeout while waiting for file input element.")
            return []

        input_file = self.driver.find_element(By.CSS_SELECTOR, "input[type='file']")
        input_file.send_keys(image_path)

        try:
            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CLASS_NAME, "enableTooltip"))
            )
        except TimeoutException:
            print("Timeout while waiting for focus button to be clickable.")
            return []

        focus_button = self.driver.find_element(By.CLASS_NAME, "enableTooltip")
        focus_button.click()

        self.scroll_to_bottom()

        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "i_results"))
            )
        except TimeoutException:
            print("Timeout while waiting for results container.")
            return []

        results_container = self.driver.find_element(By.ID, "i_results")
        images = results_container.find_elements(By.TAG_NAME, "img")
        image_urls = [img.get_attribute('src').split('?')[0] for img in images if img.get_attribute('src')]
        return image_urls
