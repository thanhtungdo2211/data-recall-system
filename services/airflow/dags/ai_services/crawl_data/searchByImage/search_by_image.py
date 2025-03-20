from searchByImage.bing_crawl_similar_image import BingImageSearch
from searchByImage.yandex_crawl_similar_image import YandexDownloader
from searchByImage.baidu_crawl_image import BaiduImageSearcher
from searchByImage.sogou_crawl_image import SogouImageSearcher
import concurrent.futures
import time

def get_bing_images(image_path):
    return BingImageSearch().search_images(image_path)

def get_yandex_images(image_path):
    return YandexDownloader().search_images(image_path)

def get_baidu_images(image_path):
    return BaiduImageSearcher().search_images(image_path)

def get_sogou_images(image_path):
    return SogouImageSearcher().search_images(image_path)

def run_with_retry(func, args, max_retries=3):
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            result = func(*args)
            end_time = time.time()
            duration = end_time - start_time
            print(f"{func.__name__} succeeded in {duration:.2f} seconds on attempt {attempt+1}")
            return result
        except Exception as e:
            print(f"{func.__name__} failed with {e}, retrying ({attempt+1}/{max_retries})...")
            time.sleep(1)  
    raise Exception(f"{func.__name__} failed after {max_retries} attempts")

def search_by_image(image_path):
    URLS = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(run_with_retry, get_bing_images, (image_path,)): 'Bing',
            # executor.submit(run_with_retry, get_baidu_images, (image_path,)): 'Baidu',
            executor.submit(run_with_retry, get_yandex_images, (image_path,)): 'Yandex',
            executor.submit(run_with_retry, get_sogou_images, (image_path,)): 'Sogou'
        }

        for future in concurrent.futures.as_completed(futures):
            source = futures[future]
            try:
                urls = future.result()
                URLS.extend(urls)
            except Exception as exc:
                print(f"{source} generated an exception: {exc}")

    print(f"Total URLs collected: {len(URLS)}")
    return URLS
