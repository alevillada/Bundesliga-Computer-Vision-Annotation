import cv2

def get_video_info(video_path):
    """
        takes in a video path and returns the list of frames - each image in the video
    """
    cap = cv2.VideoCapture(video_path) 
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames
