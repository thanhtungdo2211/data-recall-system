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

# def search_by_image(image_path : str):
#     URLS = []
#     with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
#         futures = {
#             executor.submit(run_with_retry, get_bing_images, (image_path,)): 'Bing',
#             # executor.submit(run_with_retry, get_baidu_images, (image_path,)): 'Baidu',
#             executor.submit(run_with_retry, get_yandex_images, (image_path,)): 'Yandex',
#             executor.submit(run_with_retry, get_sogou_images, (image_path,)): 'Sogou'
#         }

#         for future in concurrent.futures.as_completed(futures):
#             source = futures[future]
#             try:
#                 urls = future.result()
#                 URLS.extend(urls)
#             except Exception as exc:
#                 print(f"{source} generated an exception: {exc}")

#     print(f"Total URLs collected: {len(URLS)}")
#     return URLS

def search_by_image(image_path: str, max_urls: int = None, engine_limits: dict = None):
    """
    Search for similar images across multiple search engines.
    
    Args:
        image_path: Path to the image file to search for
        max_urls: Maximum total URLs to return (None for unlimited)
        engine_limits: Dict of {engine_name: limit} to control URLs per engine (None for unlimited)
    
    Returns:
        List of image URLs found
    """
    URLS = []
    engine_counts = {}
    engine_limits = engine_limits or {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(run_with_retry, get_bing_images, (image_path,)): 'Bing',
            executor.submit(run_with_retry, get_baidu_images, (image_path,)): 'Baidu',
            executor.submit(run_with_retry, get_yandex_images, (image_path,)): 'Yandex',
            executor.submit(run_with_retry, get_sogou_images, (image_path,)): 'Sogou'
        }

        for future in concurrent.futures.as_completed(futures):
            source = futures[future]
            try:
                urls = future.result()
                
                # Apply per-engine limit if specified
                if source in engine_limits and engine_limits[source] is not None:
                    original_count = len(urls)
                    urls = urls[:engine_limits[source]]
                    engine_counts[source] = {"collected": original_count, "used": len(urls)}
                else:
                    engine_counts[source] = {"collected": len(urls), "used": len(urls)}
                
                URLS.extend(urls)
                
                # Check if we've hit the total limit
                if max_urls and len(URLS) >= max_urls:
                    # Cancel any pending futures
                    for pending_future in [f for f in futures if not f.done()]:
                        pending_future.cancel()
                    break
                    
            except Exception as exc:
                print(f"{source} generated an exception: {exc}")
                engine_counts[source] = {"collected": 0, "used": 0, "error": str(exc)}

    # Apply overall limit
    if max_urls and len(URLS) > max_urls:
        URLS = URLS[:max_urls]

    # Print detailed stats
    total_collected = sum(stats["collected"] for stats in engine_counts.values())
    print(f"URLs collected: {total_collected}, URLs used: {len(URLS)}")
    for engine, stats in engine_counts.items():
        print(f"  {engine}: collected {stats['collected']}, used {stats['used']}")
    
    return URLS