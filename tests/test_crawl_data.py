import sys
sys.path.insert(1, '.')
sys.path.insert(1, 'services/airflow/dags/')
sys.path.insert(1, 'services/airflow/dags/ai_services/crawl_data/')

from ai_services.crawl_data.searchByImage.search_by_image import search_by_image
from ai_services.crawl_data.searchByText.search_by_key import search_by_text


# def crawl_urls_data(image_path: str = None, text: str = None):
#     if image_path:
#         return search_by_image(image_path)
#     elif text:
#         return search_by_text(text)
#     else:
#         return "Please provide either image_path or text"

# Search for images related to "human"
human_image_urls = search_by_text("human")
print(f"Found {len(human_image_urls)} human images")

# First 5 URLs:
for url in human_image_urls[:5]:
    print(url)