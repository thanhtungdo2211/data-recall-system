from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException


class YandexDownloader:
    def __init__(self):
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
                break  
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def search_images(self, image_path):
        self.driver.get("https://yandex.com/images/")
        img_urls = []
        try:
            file_input = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//input[@type='file']"))
            )
            file_input.send_keys(image_path)

            try:
                WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.CLASS_NAME, "CbirNavigation-TabsItem_name_similar-page"))
                ).click()
            except TimeoutException:
                print("Timeout while waiting for the similar images tab to be clickable.")
                return []

            self.scroll_to_bottom()

            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, "serp-item__thumb"))
                )
            except TimeoutException:
                print("Timeout while waiting for images to load.")
                return []

            imgs = self.driver.find_elements(By.CLASS_NAME, "serp-item__thumb")
            img_urls = [img.get_attribute("src") for img in imgs]

        except TimeoutException:
            print("Timeout while waiting for file input element.")
        finally:
            self.driver.quit()

        return img_urls

