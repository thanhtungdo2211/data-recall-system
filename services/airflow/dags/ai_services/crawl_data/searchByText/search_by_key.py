from searchByText.Sougou_search import SogouImageScraper
import concurrent.futures
import time

def search_by_text_Sougou(query : str, num_images : int):
    return SogouImageScraper(num_images=num_images).fetch_image_urls(query)

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

def search_by_text(keyword : str, num_images : int):
    URLS = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(run_with_retry, search_by_text_Sougou, (keyword, num_images,)): 'Sougou',
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
