import unittest
import numpy as np

def get_number_from_filename(filename):
    file_number = filename.split('.')[0]
    if not file_number:
        return None
    try:
        number = int(file_number)
        if number <= 0:  # 0 veya negatif sayılar için None döndür
            return None
        elif (number >=109):
            return None
        else:
            return number
    except ValueError:
        return None
    
def get_color_from_file(number):
    if(number >= 1 and number <= 2):
        return np.array([[24,16,56], [87,91,171]])
    elif(number >= 3 and number <= 4):
        return np.array([[59,30,95], [89,94,163]])
    elif(number >= 5 and number <= 8):
        return np.array([[30,30,34], [89,142,166]])
    elif(number >= 9 and number <= 12):
        return np.array([[54,47,50], [98,94,255]])
    elif(number >= 13 and number <= 14):
        return np.array([[38,53,117], [56,100,255]])
    elif(number >= 15 and number <= 22):
        return np.array([[0,36,81], [66,110,255]])
    elif(number >= 23 and number <= 27):
        return np.array([[30,16,113], [80,64,255]])
    elif(number >= 28 and number <= 49):
        return np.array([[78,17,169], [99,55,255]])
    elif(number >= 50 and number <= 69):
        return np.array([[44,16,86], [105,78,212]])
    elif(number >= 70 and number <= 79):
        return np.array([[64,3,163], [150,50,255]])
    elif(number >= 80 and number <= 89):
        return np.array([[20,0,11], [139,131,75]])
    elif(number >= 90 and number <= 94):
        return np.array([[65,0,42], [131,150,190]])
    elif(number >= 95 and number <= 108):
        return np.array([[100,11,56], [132,122,166]])
    else:
        return None

class TestGetColorFromFileAndGetFileName(unittest.TestCase):
    
    # Geçerli sayılar için testler
    def test_valid_numbers(self):
        self.assertTrue(np.array_equal(get_color_from_file(1), np.array([[24,16,56], [87,91,171]])))
        self.assertTrue(np.array_equal(get_color_from_file(5), np.array([[30,30,34], [89,142,166]])))
        self.assertTrue(np.array_equal(get_color_from_file(49), np.array([[78,17,169], [99,55,255]])))
        self.assertTrue(np.array_equal(get_color_from_file(108), np.array([[100,11,56], [132,122,166]])))
    
    # Geçersiz sayılar için testler
    def test_invalid_numbers(self):
        self.assertEqual(get_color_from_file(0), None)
        self.assertEqual(get_color_from_file(109), None)
        self.assertEqual(get_color_from_file(-1), None)
    
    # Sınır değerleri için testler
    def test_boundary_numbers(self):
        self.assertTrue(np.array_equal(get_color_from_file(2), np.array([[24,16,56], [87,91,171]])))
        self.assertTrue(np.array_equal(get_color_from_file(4), np.array([[59,30,95], [89,94,163]])))
        self.assertTrue(np.array_equal(get_color_from_file(108), np.array([[100,11,56], [132,122,166]])))
    
    # Geçerli değerler için test
    def test_valid_filenames(self):
        self.assertEqual(get_number_from_filename("1.txt"), 1)
        self.assertEqual(get_number_from_filename("50.jpg"), 50)
        self.assertEqual(get_number_from_filename("108.png"), 108)
    # Geçersiz Değerler için Test
    def test_invalid_filenames(self):
        self.assertEqual(get_number_from_filename("0.txt"), None)
        self.assertEqual(get_number_from_filename(".txt"), None)
        self.assertEqual(get_number_from_filename("file.txt"), None)
        self.assertEqual(get_number_from_filename("109.doc"), None)
        self.assertEqual(get_number_from_filename("-1.pdf"), None)
    # Sınır değerler için Test
    def test_boundary_filenames(self):
        self.assertEqual(get_number_from_filename("1.doc"), 1)
        self.assertEqual(get_number_from_filename("108.csv"), 108)

if __name__ == '__main__':
    unittest.main()