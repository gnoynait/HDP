import data
import unittest

class TestData(unittest.TestCase):
    def test_RandomBlob(self):
        blob = data.RandomBlob(2)
        x = blob.nextBatch(3)
        print x
    def test_FixSizeData(self):
        d = data.FixSizeData(10, 2)
        d.beforeFirst()
        while d.nextBatch(2) > 0:
            print d.value()
if __name__ == '__main__':
    unittest.main()
       
