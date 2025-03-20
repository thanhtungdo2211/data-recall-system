from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException

class BaiduImageSearcher:
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
        self.driver.get("https://graph.baidu.com/pcpage/index?tpl_from=pc")
        image_urls = []

        try:
            camera = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "graph-d20-search-wrapper-camera"))
            )
            camera.click()
        except TimeoutException:
            print("Timeout while waiting for camera button.")
            self.driver.quit()
            return []

        try:
            file_input = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.XPATH, "//input[@type='file']"))
            )
            file_input[0].send_keys(image_path)
        except TimeoutException:
            print("Timeout while waiting for file input.")
            self.driver.quit()
            return []

        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "graph-similar-list"))
            )
        except TimeoutException:
            print("Timeout while waiting for search results.")
            self.driver.quit()
            return []

        search = self.driver.find_element(By.CLASS_NAME, "graph-similar-list")
        self.scroll_to_bottom()

        try:
            imgs_src = search.find_elements(By.TAG_NAME, "a")
            image_urls = [img_src.find_element(By.TAG_NAME, "img").get_attribute("src") for img_src in imgs_src]
        except TimeoutException:
            print("Timeout while extracting image URLs.")
        finally:
            self.driver.quit()

        return image_urls

