import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_euclidean_distance
import numpy as np


class PlayerBallAssigner:
    """
        This class contains the logic for the player and ball assigner. 
        This identifies which player is closest to the ball in a given frame.
    """
    
    def __init__(self):
        # The maxium amount of pixels a player can be from the ball to be considered in possession of the ball
        self.max_player_ball_distance = 60
        
    
    def assign_ball_possession(self, players, ball_bbox):
        """
            Assigns the ball to the player that is closest to the ball.
            
            1. Get the center of the ball bbox
            2. Get the center of the player bbox
            3. Calculate the euclidean distance between the players bottom bbox corners and the ball
            4. If the distance is less than the max_player_ball_distance, assign the ball to the player
        """
        # get the center of the ball bbox
        ball_position  = get_center_of_bbox(ball_bbox)
        
        # We always wants the closes player
        # intialize the comparison variables
        min_distance = float('inf')
        assigned_player = -1
        
        for player_id, player in players.items():
            # Get the player's bounding box coordinates
            x1, y1, x2, y2 = player['bbox']
            
            # Define bottom corners of the player's bbox
            bottom_left = (x1, y2)
            bottom_right = (x2, y2)
            
            # Calculate distances from bottom corners to the ball center
            distance_left = get_euclidean_distance(bottom_left, ball_position)
            distance_right = get_euclidean_distance(bottom_right, ball_position)
            
            # Select the minimum distance
            distance = min(distance_left, distance_right)
            
            # Update assigned player if conditions are met
            # if the player is 60 pixels or less from the ball and closer than the previous player
            if distance < self.max_player_ball_distance and distance < min_distance:
                min_distance = distance
                assigned_player = player_id
                
        return assigned_player
            
            
        