from django.shortcuts import render
from django.http import JsonResponse
from .forms import ImageForm
from PIL import Image
import cv2
import numpy as np
import base64

def home(request):
   return render(request, "home.html")

def process_image(request):
   if request.method == 'POST':

      image = request.FILES["file"].read()

      encoded = base64.b64encode(image).decode("utf-8")

      #nump_data = np.asarray(image)

      #cv2.imwrite("./temp/temp_file.jpg", nump_data)


      # You can pass the processed image to the template or perform additional processing
      return render(request, 'processed_image.html', {'processed_image': encoded})

   else:
      return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

def process_image_function(uploaded_image):
    # Example: Resize the image using the Pillow library
    img = Image.open(uploaded_image)
    img.thumbnail((300, 300))  # Resize to a maximum dimension of 300x300 pixels
    return img