from utils import get_video_info
from trackers import Tracker
from team_picker import TeamPicker
from player_ball_assigner import PlayerBallAssigner
from team_possession import TeamPossession
from camera_movement_estimator import CameraMovementEstimator
import cv2

def main():
    
    # Read the video
    video_frames = get_video_info("data/08fd33_4_original.mp4") # version 1
    # video_frames = get_video_info("data/kaggle_test2.mp4") # version 2
    
    # initializae the tracker
    tracker = Tracker("models/Yolov5x/weights/best.pt")
    
    # get the object tracks 
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path="stubs/track_stubs_final_Yolov5x.pkl")
    
    # add camera movement estiamtor
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,read_from_stub=True, stub_path="stubs/camera_movement_yolov5x.pkl")
    
    # Draw Camera Movement 
    
    # Interpolate Ball Postitions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    # Assign the players to the teams
    # note: because i am using Bytetrack minimum consecutive frames is 12, the first 12 frames are skipped from the tracks. These will appear as empyt dict in the tracks['players']. Therefore the first player is at index 12
    team_picker = TeamPicker()
    team_picker.assign_team_color(video_frames[13],
                                  tracks["players"][13])
    
    # loop over every player in the tracks and assign team
    for frame_num, player_tracks in enumerate(tracks['players']):
        # skip the first 12 frames
        if frame_num < 12:
            continue
    
        for player_id, track in player_tracks.items():
            team = team_picker.pick_team_player(video_frames[frame_num],
                                                track['bbox'],
                                                player_id)
            # assign the team to the player 
            # note: this creates new keys in the tracks['players'] dict
            # Hardcoded annotation for player 24 and 31 - they are the goalkeepers
            
            if player_id == 24: 
                tracks['players'][frame_num][player_id]['team'] = 1
                tracks['players'][frame_num][player_id]['team_color'] = team_picker.team_colors[1]
                
            elif player_id == 31:
                tracks['players'][frame_num][player_id]['team'] = 2
                tracks['players'][frame_num][player_id]['team_color'] = team_picker.team_colors[2]

            else: 
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = team_picker.team_colors[team] 
            
    
    # Assign ball possession
    player_assigner = PlayerBallAssigner()
    team_possession = TeamPossession()
    
    # go thru each player and assign the ball to the player that is closest to the ball
    for frame_num, player_tracks in enumerate(tracks['players']):
        # skip the first 12 frames
        if frame_num < 12:
            continue
        
        # note: 1 is the track id for ball
        ball_bbox = tracks['ball'][frame_num][1]['bbox'] 
        player_possession = player_assigner.assign_ball_possession(player_tracks, ball_bbox)
        
        # we skip if the player_possesion is -1
        if player_possession != -1:
            tracks['players'][frame_num][player_possession]['has_ball'] = True
            team_possession.note_possession(tracks["players"], frame_num, player_possession)
        else:
            team_possession.note_possession(tracks["players"], frame_num, player_possession) # dosent need to be hardcoded, can be removed
            
        
    print(team_possession.team_possession)
    
    # Draw annotations on the output video
    output_path = "output/RESULTS/Final_Yolov5x.mp4"
    tracker.draw_annotations(video_frames, tracks,output_path,team_possession.team_possession, camera_movement_per_frame)
    
    
if __name__ == "__main__":
    main() 