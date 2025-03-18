import google.generativeai as genai
import matplotlib.patches as patches
import time
from PIL import Image

class CompareImage:
    def __init__(self, api):
        self.init_model(api)
    def __call__(self, source_image_path, target_image_path, bboxes):
        start = time.time()
        result = {
        'bboxes': [],
        'labels': []
        }
        source_image = Image.open(source_image_path)
        target_image = Image.open(target_image_path)

        cropped_images = self.crop_bboxes_pil(target_image, bboxes)

        for i in range(0,len(cropped_images)):
            time.sleep(5)
            res = self.compare_object(source_image, cropped_images[i])
            if "true" in res :

                result["bboxes"].append(bboxes[i])
                result["labels"].append("")

        print(f"Time to compare : {time.time() - start}")
        return result["bboxes"]
    
    def init_model(self, api):
        genai.configure(api_key=api)
        self.model_compare = genai.GenerativeModel('gemini-2.0-flash-exp')
        
    def compare_object(self, image1, image2):
        prompt = """
        Xác định và so sánh vật thể chính trong Ảnh 1 và Ảnh 2 bằng cách thực hiện các bước sau:

        1. Kiểm tra Ảnh 1: Tìm một vật thể duy nhất chiếm phần lớn diện tích ảnh. Nếu không có vật thể nào rõ ràng hoặc vật thể quá nhỏ so với nền, ghi nhận "không có vật thể chính" và dừng lại, trả về false. Nếu có, xác định đó là gì.

        2. Kiểm tra Ảnh 2: Lặp lại bước trên cho Ảnh 2 – tìm một vật thể duy nhất chiếm phần lớn diện tích ảnh tìm kiếm sao cho chỉ có duy nhất một vật thể. Nếu không có, ghi nhận "không có vật thể chính" và dừng lại, trả về false. Nếu có, xác định đó là gì.

        3. Nếu cả hai ảnh đều có vật thể chính: Trích xuất các đặc điểm tổng quát của vật thể trong Ảnh 1 (ví dụ: hình dạng, màu sắc, không tính kích thước) và làm tương tự cho vật thể trong Ảnh 2.

        4. So sánh các đặc điểm: Xem xét liệu hình dạng và màu sắc (hoặc các đặc điểm khác) của hai vật thể có tương tự nhau không (đảm bảo đối tượng trong ảnh 2 có đầy đủ các đặc điểm của đối tượng trong ảnh 1 ). Nếu tương tự, trả về true; nếu khác biệt rõ ràng, trả về false.

        5. Đảm bảo kết quả cuối cùng là một từ duy nhất: true hoặc false.

        Ví dụ 1:

        Ảnh 1: Nền trắng với chấm đỏ nhỏ ở góc → Vật thể quá nhỏ, không chiếm phần lớn → Dừng, trả về false.

        Ảnh 2: Nền trắng với chấm đỏ lớn ở góc → Không cần kiểm tra tiếp vì Ảnh 1 đã thất bại.

        Kết quả: false

        Ví dụ 2:

        Ảnh 1: Một người trên nền trắng → Có vật thể chính (người). Đặc điểm: hình người, không xét màu.

        Ảnh 2: Một người trên đường → Có vật thể chính (người). Đặc điểm: hình người, không xét màu.

        So sánh: Hình dạng giống nhau (đều là người) → Trả về true.

        Kết quả: true

        Ví dụ 3:

        Ảnh 1: Xe hơi màu xanh trên đường phố → Có vật thể chính (xe hơi). Đặc điểm: hình xe 4 bánh, màu xanh.

        Ảnh 2: Xe đạp màu đỏ trong công viên → Có vật thể chính (xe đạp). Đặc điểm: hình xe 2 bánh, màu đỏ.

        So sánh: Hình dạng khác (4 bánh vs 2 bánh), màu khác (xanh vs đỏ) → Trả về false.

        Kết quả: false

        Ví dụ 4:

        Ảnh 1: Nền trống không có vật thể → Không có vật thể chính → Dừng, trả về false.

        Ảnh 2: Cây xanh trong vườn → Không cần kiểm tra tiếp vì Ảnh 1 đã thất bại.

        Kết quả: false
        """
        request = self.model_compare.generate_content([prompt, image1, image2])
        return request.text
    
    def crop_bboxes_pil(self, image, bboxes):
        """
        Crop bounding boxes from an image and return a list of cropped images as PIL JpegImageFile objects.

        Parameters:
        - image (PIL.Image.Image): Input image from which the bounding boxes will be cropped.
        - bboxes (list of tuples): List of bounding boxes, where each bbox is a tuple
        (x_min, y_min, x_max, y_max).

        Returns:
        - cropped_images (list of PIL.JpegImagePlugin.JpegImageFile): List of cropped images corresponding to the bounding boxes.
        """
        cropped_images = []
        # print(bboxes)
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox

            # Ensure bbox coordinates are within image boundaries and x_max > x_min, y_max > y_min
            x_min = max(0, int(x_min))
            y_min = max(0, int(y_min))
            x_max = min(image.width, int(x_max)) # Changed image.height to image.width
            y_max = min(image.height, int(y_max)) # Changed image.width to image.height
            
            if x_max > x_min and y_max > y_min:
                # Crop the image
                cropped = image.crop((x_min, y_min, x_max, y_max))
                cropped_images.append(cropped)
            else:
                print(f"Invalid bbox: {bbox}, skipping.")  # Log invalid bboxes

        return cropped_images