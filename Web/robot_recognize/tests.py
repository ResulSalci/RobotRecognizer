from django.test import TestCase
from django.http import HttpRequest
from .views import home


class HomePageTest(TestCase):

    def test_home_page_returns_correct_template(self):
        response = self.client.get("/")
        self.assertContains(response, "<title>Robot Recognizer</title>")
        self.assertContains(response, "<html>")
        self.assertContains(response, "</html>")
        self.assertTemplateUsed(response, "home.html")