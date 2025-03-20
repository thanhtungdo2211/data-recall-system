import requests
from bs4 import BeautifulSoup
import json
from deep_translator import GoogleTranslator
from urllib.parse import urlencode

class SogouImageScraper:
    def __init__(self, target_language='zh-CN', num_images=1000):
        self.target_language = target_language
        self.num_images = num_images

    def translate_query(self, query):
        return GoogleTranslator(source='auto', target=self.target_language).translate(query)

    def fetch_image_urls(self, query):
        URLs = []
        translated_query = self.translate_query(query)
        encoded_query = urlencode({'query': translated_query})
        
        while len(URLs) < self.num_images:
            url = f"https://pic.sogou.com/pics?{encoded_query}&start={len(URLs)}"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            script_content = soup.find_all('script')[0].string
            
            json_data = script_content.replace('window.__INITIAL_STATE__=', '') 
            json_data = json_data.replace(';(function(){var s;(s=document.currentScript||document.scripts[document.scripts.length-1]).parentNode.removeChild(s);}());', '')
            data = json.loads(json_data)

            images = data.get('searchList', {}).get('searchList', [])
            for image in images:
                URLs.append(image.get('picUrl'))
                if len(URLs) >= self.num_images:
                    break

        return URLs

