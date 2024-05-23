import os
# from loguru import logger as log
from common.timeit import time_it
from invoice_rec import InvoiceRec
import cv2
import numpy as np


@time_it
def main(img_file):
    img = cv2.imdecode(np.frombuffer(img_file, dtype=np.uint8), cv2.IMREAD_COLOR)
    #img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), cv2.IMREAD_COLOR)
    invoiceRec = InvoiceRec(img)
    result = invoiceRec()
    return result

#main(r'test/1/fe8b4dd018952a0752b6b7ceff9e9d7.jpg')