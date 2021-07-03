import unittest

class CropperMethods(unittest.TestCase):

    def test_cropper_create(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_cropper_run(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())


if __name__ == '__main__':
    unittest.main()