class laneMemory: 
    # Creation 29.08.2024 
    # Last Changed: 4.4.2025 -- removed unused function 
    # PURP: Defines a snapshot of a processed lane
    # VAR: leftExist, rightExist, leftLane, rightLane
    # may contain methods in handling comparison between two lane Memories 
    # AIM:  to help fix some on the inconsistencies with the current lane detection

    def __init__(self, pLeftExist, pRightExist, pLeftLane, pRightLane):
        self.leftExist = pLeftExist
        self.rightExist = pRightExist
        self.leftLane = pLeftLane
        self.rightLane = pRightLane