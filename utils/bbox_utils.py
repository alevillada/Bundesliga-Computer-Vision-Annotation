

def get_center_of_bbox(bbox):
    """
        takes in a bounding box and returns the center of the bounding box
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) // 2), int((y1 + y2) // 2)

def get_width_of_bbox(bbox):
    """
        takes in a bounding box and returns the width of the bounding box
    """
    # same as saying x2 - x1
    return bbox[2] - bbox[0]

def get_euclidean_distance(p1, p2):
    """
        calculates the euclidean distance between two points
    """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def measure_xy_distance(p1, p2):
    return p1[0] - p2[0], p1[1] - p2[1]