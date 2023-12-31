import base64
import io
import unittest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, RequestFactory
from PIL import Image

from .Konsol6.main import find_robot
from .views import home, process_image

class HomeViewTest(TestCase):

    # Home Page'in doğru template'i döndürüp döndürmedğini kontrol et
    def test_home_view_uses_correct_template(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'home.html')

class ProcessViewTest(TestCase):
    def setUp(self):
        self.factory = RequestFactory()

    # POST yerine GET request gönderilirse kullanınıcın ana sayfaya yönlendirilip yönlendirilmediğini kontrol et
    def test_process_get_request_redirects_home(self):
        response = self.client.get('/api/process')
        self.assertRedirects(response, '/')

    # Görsel içermeyen request gelirse kullanınıcın ana sayfaya yönlendirilip yönlendirilmediğini kontrol et
    def test_process_post_request_without_file_redirects_home(self):
        response = self.client.post('/api/process')
        self.assertRedirects(response, '/')
        
    # Geçersiz max alan içeren request gelirse kullanınıcın ana sayfaya yönlendirilip yönlendirilmediğini kontrol et
    def test_process_post_request_with_invalid_max_area_redirects_home(self):
        image = Image.new('RGB', (100, 100))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")

        response = self.client.post('/api/process', {'file': buffer, 'minArea': 10, 'maxArea': 5, 'minCircularity': 0.5,
                                                      "gaussianBlurKernelSize": 51, "thresholdMin": 192, "thresholdMax": 193})
        self.assertRedirects(response, '/')

    # Hatasız bir isteğe doğru bir şekilde yanı verildiğini kontrol et
    def test_process_post_request_with_valid_parameters_returns_processed_image(self):
        image = Image.new('RGB', (100, 100))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")

        file_to_upload = SimpleUploadedFile("test_image.jpg", buffer.getvalue())

        response = self.client.post('/api/process', {'file': file_to_upload, 'minArea': 10, 'maxArea': 100, 'minCircularity': 0.5,
                                                     "gaussianBlurKernelSize": 51, "thresholdMin": 192, "thresholdMax": 193})

        self.assertIn("processed_image", response.context)

    # Geçersiz dosya formatı içeren istek geldiğinde kullanınıcın ana sayfaya yönlendirilip yönlendirilmediğini kontrol et
    def test_process_post_request_with_invalid_file_format_redirects_home(self):
        text_file = SimpleUploadedFile("test.txt", b"Invalid file format")

        response = self.client.post('/api/process', {'file': text_file, 'minArea': 10, 'maxArea': 100, 'minCircularity': 0.5,
                                                     "gaussianBlurKernelSize": 51, "thresholdMin": 192, "thresholdMax": 193})

        self.assertRedirects(response, '/')

    # Eksik parametre içeren istek geldiğinde kullanınıcın ana sayfaya yönlendirilip yönlendirilmediğini kontrol et
    def test_process_post_request_with_missing_parameters_redirects_home(self):
            
            image = Image.new('RGB', (100, 100))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")

            response = self.client.post('/api/process', {'file': buffer, 'minArea': 10, 'maxArea': 100,
                                                         "gaussianBlurKernelSize": 51, "thresholdMin": 192, "thresholdMax": 193})

            self.assertRedirects(response, '/')

    # Response template'ine 64'lük tabanda kodlanmış resimnib başarıyla aktraıldığını kontrol et
    def test_processed_image_view_template_has_base64_image(self):
         
        image = Image.new('RGB', (100, 100))
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        file_to_upload = SimpleUploadedFile("test_image.jpg", buffer.getvalue())

        response = self.client.post('/api/process', {'file': file_to_upload, 'minArea': 10, 'maxArea': 100, 'minCircularity': 0.5,
                                                     "gaussianBlurKernelSize": 51, "thresholdMin": 192, "thresholdMax": 193})

        self.assertIn('processed_image', response.context)

        processed_image = response.context['processed_image']
        self.assertTrue(processed_image)

        self.assertIsInstance(processed_image, str)

    # 64'lük tabanda kodlanan resmin hatasız bir şekilde kodlandığını kontrol et
    def test_process_post_request_returns_valid_base64_encoded_image(self):
            image = Image.new('RGB', (100, 100))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")

            file_to_upload = SimpleUploadedFile("test_image.jpg", buffer.getvalue())

            response = self.client.post('/api/process', {'file': file_to_upload, 'minArea': 10, 'maxArea': 100, 'minCircularity': 0.5,
                                                         "gaussianBlurKernelSize": 51, "thresholdMin": 192, "thresholdMax": 193})

            encoded_image = response.context['processed_image']

            try:
                decoded_image = base64.b64decode(encoded_image)
                Image.open(io.BytesIO(decoded_image)) 
            except Exception as e:
                self.fail(f"Decoding or opening the processed image failed: {e}")

    # Doğru parametreler içeren isteğe karşılık doğru tempale'in gönderildiğini kontrol et
    def test_process_post_request_returns_correct_template_after_getting_valid_inputs(self):
         
        image = Image.new('RGB', (100, 100))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")

        file_to_upload = SimpleUploadedFile("test_image.jpg", buffer.getvalue())

        response = self.client.post('/api/process', {'file': file_to_upload, 'minArea': 10, 'maxArea': 100, 'minCircularity': 0.5,
                                                     "gaussianBlurKernelSize": 51, "thresholdMin": 192, "thresholdMax": 193})

        self.assertTemplateUsed(response, "processed_image.html")

    # Döndürülen template'in doğru formatta bir img tag içerdiğini kontrol et
    def test_process_post_request_returned_template_contains_correct_image_tag(self):
        image = Image.new('RGB', (100, 100))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")

        file_to_upload = SimpleUploadedFile("test_image.jpg", buffer.getvalue())

        response = self.client.post('/api/process', {'file': file_to_upload, 'minArea': 10, 'maxArea': 100, 'minCircularity': 0.5,
                                                     "gaussianBlurKernelSize": 51, "thresholdMin": 192, "thresholdMax": 193})

        self.assertIn('<img style="display: block;" src="data:image/jpeg;base64,', response.content.decode("utf8"))

    # Gelen istek çift bir gaussian blur kernel büyüklüğü içeriyorsa kullanınıcın ana sayfaya yönlendirilip yönlendirilmediğini kontrol et
    def test_process_post_request_with_even_blur_kernel_size_redirects_home(self):
        image = Image.new('RGB', (100, 100))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        file_to_upload = SimpleUploadedFile("test_image.jpg", buffer.getvalue())

        response = self.client.post('/api/process', {'file': file_to_upload, 'minArea': 10, 'maxArea': 100, 'minCircularity': 0.5,
                                                    "gaussianBlurKernelSize": 50, "thresholdMin": 192, "thresholdMax": 193})

        self.assertRedirects(response, '/')

    # Sistemin büyük görselleri sorunsuzca işleyebildiğini kontrol et
    def test_process_post_request_with_large_image_returns_processed_image(self):
        image = Image.new('RGB', (5000, 5000))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")

        file_to_upload = SimpleUploadedFile("large_image.jpg", buffer.getvalue())

        response = self.client.post('/api/process', {'file': file_to_upload, 'minArea': 10, 'maxArea': 100, 'minCircularity': 0.5,
                                                    "gaussianBlurKernelSize": 51, "thresholdMin": 192, "thresholdMax": 193})

        self.assertIn("processed_image", response.context)

    # Sistemin çok küçük görselleri sorunsuzca işleyebildiğini kontrol et
    def test_process_post_request_with_very_small_image_returns_processed_image(self):
        image = Image.new('RGB', (1, 1))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")

        file_to_upload = SimpleUploadedFile("large_image.jpg", buffer.getvalue())

        response = self.client.post('/api/process', {'file': file_to_upload, 'minArea': 10, 'maxArea': 100, 'minCircularity': 0.5,
                                                    "gaussianBlurKernelSize": 51, "thresholdMin": 192, "thresholdMax": 193})

        self.assertIn("processed_image", response.context)

    # JPG olmayan ancak PNG gibi eşdeğer resim formatlarının da başarıyla işlenebildiğini kontrol et
    def test_process_post_request_with_valid_non_jpeg_image_format_returns_processed_image(self):
    
        image = Image.new('RGB', (500, 500))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        file_to_upload = SimpleUploadedFile("large_image.jpg", buffer.getvalue())

        response = self.client.post('/api/process', {'file': file_to_upload, 'minArea': 10, 'maxArea': 100, 'minCircularity': 0.5,
                                                    "gaussianBlurKernelSize": 51, "thresholdMin": 192, "thresholdMax": 193})

        self.assertIn("processed_image", response.context)

if __name__ == '__main__':
    unittest.main()
