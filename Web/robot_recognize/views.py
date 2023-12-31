from django.shortcuts import render, redirect
from django.core.exceptions import ValidationError
from django.core.validators import validate_image_file_extension
from .Konsol6.main import find_robot
import cv2
import numpy as np
import base64

def home(request):
   return render(request, "home.html")


def process_image(request):

   #Istek ancak POST request ise işlenir
   if request.method == 'POST':

      try:
         validate_image_file_extension(request.FILES["file"])

      except:
         return redirect("home")

      #Eğer bir görsel gelmediyse kullanıcıyı home page'e yönlendir
      if not request.FILES.get("file"):
         return redirect("home")

      #Formdan gelen resmi al
      image = request.FILES["file"].read()

      #Formdan gelen parametreleri al ve uygun veri tiplerine çevir
      minArea = int(request.POST["minArea"])
      maxArea = int(request.POST["maxArea"])
      minCircularity = float(request.POST["minCircularity"])
      gaussianBlurKernelSize = int(request.POST["gaussianBlurKernelSize"])
      thresholdMin = int(request.POST["thresholdMin"])
      thresholdMax = int(request.POST["thresholdMax"])

      #Eğer parametrelerde eksik varsa kullanıcıyı home page'e yönlendir.
      if minArea is None or maxArea is None or minCircularity is None or gaussianBlurKernelSize is None or thresholdMin is None or thresholdMax is None:
         return redirect("home")

      # Gaussianblur kernel büyüklüğü çift sayı olamaz öyle girildeiyse kullanıcıyı ana sayfaya yönlendir
      if gaussianBlurKernelSize % 2 == 0:
         return redirect("home")

      # Maksimum değerler minimumlardan küçük gönderilirse işlemi durdur ve kullanıcıyıdan yeniden girdi al
      if maxArea < minArea or thresholdMax < thresholdMin:
         return redirect("home")

      #Görseli opencv'nin kullanabileceği bir formata getir
      image_for_cv = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)


      #Görselde analizi yap ve sonucu değişkene kaydet
      image_to_encode = find_robot(image_for_cv, minArea, maxArea, minCircularity, gaussianBlurKernelSize, thresholdMin, thresholdMax)

      #Resmi jpg formatına çevir
      _, buffer = cv2.imencode(".jpg", image_to_encode)

      #Resme link vermek yerine direk göndermek için resmi 64'lük tabana çevir.
      #Taryıcı bu 64 tabanda gelen veriyi tekrar görsele frontende çevirecektir. 
      encoded = base64.b64encode(buffer).decode("utf-8")

      #Kullanıcıya resmi içeren response'u gönder
      return render(request, 'processed_image.html', {'processed_image': encoded})

   #POST'tan farklı bir şey ise kullanıcıyı Home Page'e yönlendir
   else:
      return redirect("home")
