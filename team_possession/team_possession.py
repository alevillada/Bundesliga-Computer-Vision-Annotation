import numpy as np

class TeamPossession:
    """
        This class is responsible for calculating the team possession.
    """
    
    def __init__(self):
        self.team_possession = []
        
        
    def note_possession(self, tracks, frame_num, player_possession):
        """
            This function notes the possession of the ball.
            
            tracks: dict
            frame: int
            player_possession: int
            
            1. If the assigned player is -1, then the team possession is the same as the previous frame.
            2. If the assigned player is not -1, then the team possession is the same as the assigned player.
            
            Since each player is assigned to a team in the previous step, we can use the assigned player to determine the team possession.
        """
        # Handle the first frame explicitly
        if len(self.team_possession) == 0:
            # Default to team 0 possession for the first frame
            if player_possession == -1:
                self.team_possession.append(0)  # Default team value
            else:
                self.team_possession.append(tracks[frame_num][player_possession]["team"])
            return self.team_possession

        if player_possession == -1:
            # No player has possession, carry over the last possession
            # self.team_possession.append(self.team_possession[-1])
            # No player has possession, set to 0
            self.team_possession.append(0)
        else:
            # Player has possession, record their team
            self.team_possession.append(tracks[frame_num][player_possession]["team"])

        return self.team_possession
        