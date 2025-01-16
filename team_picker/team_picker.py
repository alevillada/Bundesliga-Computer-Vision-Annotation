# object that will pick the teams based on color 
from sklearn.cluster import KMeans


class TeamPicker:
    """
        Object the will pick the teams based on their kit color. Specifically their jersey color.
        
        uses kmeans clustering - check color_kmeans_development/color_assigment.py for the how of this.
    """
    
    def __init__(self):
        self.team_colors = {}
        
        # helps keep track of the teams - by [player_id: team]
        self.player_team_dict = {}
        
    
    
    def get_clustering_model(self, image):
        """
            Returns the kmeans clustering model that will be used to get the color of the player's jersey.
        """
        # Rehsape image for kmeans
        image_2d = image.reshape(-1, 3)
        
        # perform kmeans clustering - random state is set to 42 for reproducibility
        kmeans = KMeans(n_clusters=2, random_state= 42).fit(image_2d)
        return kmeans 
    
    
    def assign_team_color(self,frame, player_detections):
        """
            Identifies the team colors. 
            
            What colors are in the video and assigns them to the teams.
            
            1. Create an array of the player colors
            2. Perform kmeans clustering on the player colors (all the players in the video in a given frame)
            3. Retrieve each player jersey color and add it to the player_colors array
            4. Perform kmeans clustering on the player_colors array to get the team colors - kmeans clusters 2 colors.
            5. Assign the team colors to the team_colors dictionary (team_colors['1'] and team_colors['2'])
        """
        
        player_colors = []
        player_ids = []
        
        for player_id, player_detections in player_detections.items():
            # get the bbox of the player
            bbox = player_detections['bbox']
            
            # this function is basically what was worked on in color_kmeans_development/color_assigment.py
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
            player_ids.append(player_id)
        
        # do kmenas clustering on the player colors to get two color labels per team
        KMeans_results = KMeans(n_clusters=2, random_state= 42).fit(player_colors)
        self.team_kmeans_results = KMeans_results
        
        # note: This is for non-hardcoded
        # # Assign the team colors
        self.team_colors[1] = KMeans_results.cluster_centers_[0]
        self.team_colors[2] = KMeans_results.cluster_centers_[1]
    

    
    def get_player_color(self, frame, bbox):
        """
            use kmeans clustering to get the color of the player's jersey
            
            note: for more details go to color_kmeans_development/color_assigment.py
            
            1. Get the cropped image of the player
            2. Select only the top half of the image, the players jersey.
            3. Initialize the kmeans clustering object by calling on the function
            4. Get the cluster labels for each pixel
            5. Reshape the cluster labels to the shape of the image
            6. Figure out the player color by getting the opposite cluster of the corner clusters
            
            returns: player_color - the color of the player's jersey
        """
        # Get the cropped image of the player
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]\
            
        # Select only the top half of the image, the players kit.
        player_jersey = image[0:int(image.shape[0]/2), :]
        
        # initialize the kmeans clustering object by calling on the function
        kmeans = self.get_clustering_model(player_jersey)
        
        # get the cluster labels for each pixel
        cluster_labels = kmeans.labels_
        
        # reshape the cluster labels to the shape of the image
        cluster_labels = cluster_labels.reshape(player_jersey.shape[0], player_jersey.shape[1])
        
        # since most of our images are in the same format - the middle of the bbox being the player, using the corner clusters of the image could be a good way to determine the player shirt color - since the player shirt will be the other cluster color
        corner_clusters = [cluster_labels[0, 0], cluster_labels[0, -1], cluster_labels[-1, 0], cluster_labels[-1, -1]]

        # most common cluster in the corners will be the non player cluster
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        
        # The opposite cluster will be the player cluster
        player_clusters = 1 - non_player_cluster
        
        
        # player jersey color
        player_color = kmeans.cluster_centers_[player_clusters]
    
        return player_color
        

    def pick_team_player(self, frame, player_bbox, player_id):
        """
            Assigns the player to the team dependent on the color of the player's jersey.
            
            1. Cehcks if the player has already been assigned a team
            2. If not, get the player's color
            3. Use the kmeans model to predict the team color for that player
            4. Assign the player to the team
            
            returns: team_color_id - the team id (1 or 2) the player is assigned to
        """
        # ! Manually add Goalkeepers to respective teams using the player_id !
        # if player_id == 24:
        #     self.player_team_dict[24] = 1
            
        # elif player_id == 31:
        #     self.player_team_dict[31] = 2
        
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        
        player_color = self.get_player_color(frame, player_bbox)
        
        team_color_id = self.team_kmeans_results.predict(player_color.reshape(1, -1))[0]
        team_color_id += 1
        
        self.player_team_dict[player_id] = team_color_id
        
        return team_color_id