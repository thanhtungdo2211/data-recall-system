from searchByImage.search_by_image import search_by_image
from searchByText.search_by_key import search_by_text

# Search for images related to "human"
# human_image_search_by_text_urls = search_by_text("human")
# print(f"Found {len(human_image_search_by_text_urls)} human images")

# Search for images related to "human" using an image
human_image_search_by_image_urls = search_by_image("/mnt/d/Personal/Programing/PersonalProjects/data-recall-system/services/airflow/dags/ai_services/crawl_data/human.jpg")

# Write all URLs to a text file
with open("human_image_urls.txt", "w") as f:
    for url in human_image_search_by_image_urls:
        f.write(url + "\n")
        
print(f"Found {len(human_image_search_by_image_urls)} human images")

# First 5 URLs:
# for url in human_image_search_by_text_urls[:5]:
#     print("URL search by text",url)

for url in human_image_search_by_image_urls[:5]:
    print("URL search by image",url)