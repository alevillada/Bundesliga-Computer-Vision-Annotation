from ultralytics import YOLO # type: ignore
import supervision as sv
import pickle
import os
import sys
import cv2
import numpy as np
import pandas as pd
sys.path.append('../')
from utils import get_center_of_bbox, get_width_of_bbox

class Tracker:
    """
        This class helps us identiy and track bounding boxes in our model
    """
    
    def __init__(self, model_path:str):
        """
            model_path: str: the path to the model weights
        """
        self.model = YOLO(model_path)
        
        # note: the parameters were fine tuned for this example.
        self.tracker = sv.ByteTrack(frame_rate=24,
                       minimum_matching_threshold=0.95,
                       lost_track_buffer= 80,
                       minimum_consecutive_frames=12,
                       track_activation_threshold=.50) 
    
    
    def interpolate_ball_positions(self, ball_tracks):
        """
            Use interpolation, or insertion of missing positional values, into the ball tracks. This will fill in the missing bbox positions for the ball. pandas provides a simple way to do this.
            
            Steps: 
                1. Create a pandas dataframe with the ball positions
                2. Use pandas interpolate method to fill in the missing values
        """
        ball_positions = []

        # note: since the model is detecting after 12 frames, this has to be taken into account or it will cause error. 
        # Extract ball positions with validation - filling in missing values with None placeholders
        for x in ball_tracks:
            bbox = x.get(1, {}).get('bbox')
            # check if the data is in the right format
            if bbox and isinstance(bbox, list) and len(bbox) == 4:
                ball_positions.append(bbox)
            # add placeholder for missing values
            else:
                ball_positions.append([None, None, None, None])
        
        # create a pandas dataframe - Remember we are using x1,y1,x2,y2 format
        ball_pos_df = pd.DataFrame(ball_positions, columns=["x1","y1","x2","y2"])
        
        # Interpolate the missing values
        ball_pos_df = ball_pos_df.interpolate()
        ball_pos_df = ball_pos_df.bfill()
        
        ball_positions = [{1: {'bbox': x} }for x in ball_pos_df.to_numpy().tolist()]
        
        return ball_positions
    
        
    def detect_frames(self, frames):
        """
            Detects the bounding boxes in the frames
        """
        # create a batch size of 16
        batch_size = 8
        detections = []
        count = 0
        
        for i in range(0, len(frames), batch_size):
            # get the batch
            batch = frames[i:i+batch_size]
            batch_detections = self.model.predict(batch, conf=0.5)
            
            # adds it to the detections list
            detections += batch_detections
        return detections
    
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
            this function will check for the existence of a pickle file and read the tracks from it. 
            
            If the file does not exist, it will run the model and save the tracks to a pickle file
        
            frames: list: a list of frames
            read_from_stub: bool: whether to read the tracks from the pickle file
            stub_path: str: the path to the pickle file
            
            returns: dict: a dictionary of the tracks
            
            , read_from_stub=False, stub_path=None
        """
        # if read_from_stub is True, we will read the tracks from the pickle file and return it
        # This will save us the time of re-running the model
        if (read_from_stub == True) and (stub_path is not None) and (os.path.exists(stub_path)):
            # finds the pickle file and returns the tracks
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
                print(f"Tracks load was a success")
            return tracks
        
        detections = self.detect_frames(frames)
        
        tracks ={
            "players": [],
            "referees": [],
            "ball": []
        }
        
        
        for frame_num, detection in enumerate(detections):
            # changing the class names from football-player-detection-1/data.yaml to make it more easily readible
            # extract names and values
            cls_names = detection.names
            # exchange their positions
            cls_names_inv = {v:k for k,v in cls_names.items()}
            
            # convert to supervision detection format
            detections_supervision = sv.Detections.from_ultralytics(detection)
            
            # goalkeepers aren't being detected accurately in the model, this could just be because of the size of the dataset, therefore we will change goalkeeper status {1: 'goalkeeper'} to {2: 'person'}
            for object_ind, class_id in enumerate(detections_supervision.class_id):
                # each object_ind is a frame
                # each frame has multiple class_ids for the objects detected
                if cls_names[class_id] == "goalkeeper":
                    detections_supervision.class_id[object_ind] = cls_names_inv["player"]
                    
    
            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detections_supervision)
            
            # print(detection_with_tracks)
            # store the bounding box information
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_with_tracks:
                # check output for indexing 
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3].tolist()
                track_id = frame_detection[4].tolist()
                
                print(f"Frame {frame_num}, Track ID: {track_id}, Class ID: {cls_id}")
                
                # save the boudning box information to the appropriate track dictionary
                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                    
                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox} 
            
            for frame_detection in detections_supervision:
                # check output for indexing 
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3].tolist()
                
                # save the boudning box information to the appropriate track dictionary
                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}
        
        
        # if there is no pickle file, save the tracks to a pickle file
        # this will save us the time of re-running the model   
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)
                print(f"{tracks}\nTracks saved to {stub_path}\nTracks save was a success")
                
        return tracks
        
        
    def draw_oval(self, frame, bbox, color, track_id=None):
        """
            Function to draw an oval around the players
        """
        # we want the center of the bottom y to create the oval
        y2 = int(bbox[3]) # bottom of the bounding box
        x_center, _ = get_center_of_bbox(bbox)
        width = get_width_of_bbox(bbox)
    
        # use cv2 to create the visual annotations
        cv2.ellipse(frame,
                    center= (x_center, y2),
                    axes= (int(width), int(0.40*width)), # determines the size of the oval
                    angle= 0.0,
                    startAngle= -45,
                    endAngle= 235,
                    color= tuple(color),
                    thickness= 2,
                    lineType= cv2.LINE_4)
        
        # add track id annotation at the bottom of the oval, inside a rectangle.
        rectangle_width = 40
        rectangle_height= 20
        x1_rect = x_center - rectangle_width// 2
        x2_rect = x_center + rectangle_width// 2
        y1_rect = (y2 - rectangle_height//2) + 15 # bottom of the bounding box
        y2_rect = (y2 + rectangle_height//2) + 15 
        
        # only if theres a track id
        if track_id is not None:
            cv2.rectangle(frame, 
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            # more adjustments - Since my track_id is still giving high numbers (above 100), i want to change the postion of the text for those high numbers 
            x1_rect_text = x1_rect + 12
            if track_id > 99:
                x1_rect_text -= 15
            
            cv2.putText(frame, 
                        f"{track_id}",
                        (int(x1_rect_text), int(y1_rect + 15)),
                        cv2.FONT_HERSHEY_COMPLEX, # font type
                        0.6, # font size
                        (0,0,0), # font color
                        2 # font thickness
                        )
        
        return frame
    
    
    def draw_traingle(self, frame, bbox, color):
        """
            This function draws a trangle to move around with the ball. 
        """
        y = int(bbox[1])
        
        # top of the bounding box, so it can float above the ball
        x,_ = get_center_of_bbox(bbox) # just grab x of the center 
        
        # math coordiantes for the trangle
        trignale_points = np.array([[x,y-5],
                                   [x-8, y-17],
                                   [x+8, y-17]])
        
        # this will draw and fill the triangle with the color
        cv2.drawContours(frame, [trignale_points], 0, color, cv2.FILLED)
        
        # this will draw the outline of the triangle
        cv2.drawContours(frame, [trignale_points], 0, (0,0,0), 2)
        
        return frame
    
    
    def draw_team_possession(self, frame, frame_num, team_ball_possession):
        """
            This function draws the team ball possession calculation on each frame of the video.
            
            tracks: dict: the tracks dictionary
            frame: int: the frame number
            player_possession: int: the player with the ball on a given frame
            
            returns: frame: the annotated frame
        """
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350,850), (1900, 970), (255,255,255), cv2.FILLED)
        alpha = 0.4 # transparency factor
        
        # combine rectangle and alpha
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # convert the team ball possession to a numpy array
        # Up to the current frame, calculate the team possession
        team_ball_control_percentage_to_frame = np.array(team_ball_possession[:frame_num+1])
        
        # calculate possession
        team_1_ball_control = team_ball_control_percentage_to_frame[team_ball_control_percentage_to_frame == 1].shape[0]
        team_2_ball_control = team_ball_control_percentage_to_frame[team_ball_control_percentage_to_frame == 2].shape[0]
        
        # skip: if there is no registered ball control
        if team_1_ball_control + team_2_ball_control == 0:
            team_1_possesion = 0
            team_2_possesion = 0
        
        else:
            team_1_possesion = team_1_ball_control / (team_1_ball_control + team_2_ball_control)
            team_2_possesion = team_2_ball_control / (team_1_ball_control + team_2_ball_control)
        
        # annotate the frame with the team possession
        cv2.putText(frame, 
                    f"Team 1: {team_1_possesion*100:.2f}%",
                    (1400, 900),
                    cv2.FONT_HERSHEY_COMPLEX, # font type
                    1, # font size
                    (0,0,0), # font color
                    2 # font thickness
                    )
        cv2.putText(frame, 
                    f"Team 2: {team_2_possesion*100:.2f}%",
                    (1400, 950),
                    cv2.FONT_HERSHEY_COMPLEX, # font type
                    1, # font size
                    (0,0,0), # font color
                    2 # font thickness
                    )
        
        return frame
    
    
    def draw_camera_movement(self, frame,frame_num, camera_movement_per_frame):
        """
            Contains all the camera movement annotations
            
            frame: int: the frame we are working on
            camera_movement_per_frame: list: the camera movement per frame
            
            returns: frame: the annotated frame
        """
        overlay = frame.copy()
        
        # draw the white rectangle for camera movement
        cv2.rectangle(overlay, (60,63), (560, 123), (255,255,255), cv2.FILLED)
        alpha = 0.6 # transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # extract the camera movement for the current frame
        # annoate the frame with the camera movement
        x_movement, y_movement = camera_movement_per_frame[frame_num]
        frame = cv2.putText(frame,f"Camera Movement X: {x_movement:.2f}", (65,103), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
        # frame = cv2.putText(frame,f"Camera Movement Y: {y_movement:.2f}", (65,123), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
        
        return frame
    
        
    def draw_annotations(self, frames, tracks, output_path, team_ball_possession, camera_movement_per_frame):
        """
            Function to annotate and create visual representation of the tracks
        """
        # note: new approach to save add annotations to the video one frame at a time
        video_format = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, video_format, 24, (frames[0].shape[1],frames[0].shape[0]))
           
        
        for frame_num, frame in enumerate(frames):
            # copy the video frames to avoid overwriting the original frames
            frame = frame.copy()
            
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            
            # Draw the refs oval like fifa -yellow
            for _, referee in referee_dict.items():
                frame = self.draw_oval(frame, referee["bbox"], (0,255,255))
            
            # Draw the players oval like fifa
            # ! dependent on team color assignment ! defualt is red
            for track_id, player in player_dict.items():
                team_colors = player.get("team_color", (0,0,255)) # default is red
                frame = self.draw_oval(frame, player["bbox"], team_colors, track_id)
                
                # draw a triangle on the player with the ball
                # if the player has the ball then do the following, if not return False. 
                if player.get("has_ball", False):
                    frame = self.draw_traingle(frame, player["bbox"], (0,0,225))
                
            # Draw the ball tringle like fifa -blue
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (255,0,0))
            
            # Draw team possession calculations
            frame = self.draw_team_possession(frame, frame_num, team_ball_possession)

            frame = self.draw_camera_movement(frame, frame_num, camera_movement_per_frame)
            
            out.write(frame)
            
            del frame
        
        out.release()
        print("output video saved")
            