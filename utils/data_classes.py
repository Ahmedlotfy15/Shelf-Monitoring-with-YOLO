import numpy as np



class Shelf:
    def __init__(self, box):
        box_int = box.numpy().astype(np.int32)
        self.p1 = (box_int[0][0], box_int[0][1])  
        self.p2 = (box_int[1][0], box_int[1][1])
        self.p3 = (box_int[2][0], box_int[2][1])
        self.p4 = (box_int[3][0], box_int[3][1])


class Product:
    def __init__(self, box):
        points = box.numpy().astype(np.int32)
        self.p1 = (points[0], points[1])  
        self.p2 = (points[2], points[3])  
        


