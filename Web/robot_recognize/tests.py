import base64
import io
import unittest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, RequestFactory
from PIL import Image

from .Konsol6.main import find_robot
from .views import home, process_image

class HomeViewTest(TestCase):
    def test_home_view_uses_correct_template(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'home.html')

class ProcessViewTest(TestCase):
    def setUp(self):
        self.factory = RequestFactory()


    def test_process_get_request_redirects_home(self):
        response = self.client.get('/api/process')
        self.assertRedirects(response, '/')

    def test_process_post_request_without_file_redirects_home(self):
        response = self.client.post('/api/process')
        self.assertRedirects(response, '/')

    def test_process_post_request_with_invalid_max_area_redirects_home(self):
        image = Image.new('RGB', (100, 100))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")

        response = self.client.post('/api/process', {'file': buffer, 'minArea': 10, 'maxArea': 5, 'minCircularity': 0.5})
        self.assertRedirects(response, '/')

    def test_process_post_request_returns_processed_image(self):
        image = Image.new('RGB', (100, 100))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")

        file_to_upload = SimpleUploadedFile("test_image.jpg", buffer.getvalue())

        response = self.client.post('/api/process', {'file': file_to_upload, 'minArea': 10, 'maxArea': 100, 'minCircularity': 0.5})

        self.assertIn("processed_image", response.context)

    def test_process_post_request_with_invalid_file_format_redirects_home(self):
        text_file = SimpleUploadedFile("test.txt", b"Invalid file format")

        response = self.client.post('/api/process', {'file': text_file, 'minArea': 10, 'maxArea': 100, 'minCircularity': 0.5})

        self.assertRedirects(response, '/')

    def test_process_post_request_with_missing_parameters_redirects_home(self):
            
            image = Image.new('RGB', (100, 100))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")

            response = self.client.post('/api/process', {'file': buffer, 'minArea': 10, 'maxArea': 100})

            self.assertRedirects(response, '/')

    def test_processed_image_view_template_has_base64_image(self):
         
        image = Image.new('RGB', (100, 100))
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        file_to_upload = SimpleUploadedFile("test_image.jpg", buffer.getvalue())

        response = self.client.post('/api/process', {'file': file_to_upload, 'minArea': 10, 'maxArea': 100, 'minCircularity': 0.5})

        self.assertIn('processed_image', response.context)

        processed_image = response.context['processed_image']
        self.assertTrue(processed_image)

        self.assertIsInstance(processed_image, str)

    def test_process_post_request_returns_valid_base64_encoded_image(self):
            image = Image.new('RGB', (100, 100))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")

            file_to_upload = SimpleUploadedFile("test_image.jpg", buffer.getvalue())

            response = self.client.post('/api/process', {'file': file_to_upload, 'minArea': 10, 'maxArea': 100, 'minCircularity': 0.5})

            encoded_image = response.context['processed_image']

            try:
                decoded_image = base64.b64decode(encoded_image)
                Image.open(io.BytesIO(decoded_image)) 
            except Exception as e:
                self.fail(f"Decoding or opening the processed image failed: {e}")

    def test_process_returns_correct_template_after_getting_valid_inputs(self):
         
        image = Image.new('RGB', (100, 100))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")

        file_to_upload = SimpleUploadedFile("test_image.jpg", buffer.getvalue())

        response = self.client.post('/api/process', {'file': file_to_upload, 'minArea': 10, 'maxArea': 100, 'minCircularity': 0.5})

        self.assertTemplateUsed("processed_image.html")


if __name__ == '__main__':
    unittest.main()
