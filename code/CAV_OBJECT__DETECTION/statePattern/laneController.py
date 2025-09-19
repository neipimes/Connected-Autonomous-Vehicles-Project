#STATE PATTERN
#WILL BE ASSIGNED EITHER oneLaneState or twoLaneState 
from statePattern import oneLaneState as ols
from statePattern import twoLaneState as tls
from statePattern import correctionState as cs
from statePattern import turningState as ts
class laneController:

    def __init__(self):
        self.onelanestate    = ols.oneLaneState(self)
        self.twolanestate    = tls.twoLaneState(self)
        self.correctionstate = cs.correctionState(self)
        self.turningstate    = ts.turningState(self)
        self.state           = self.twolanestate 

    #Change the state of the objects held by lanestat3e
    def changeState(self):
        self.state.changeState() 

    #To tell us what state it is in
    def getState(self):
        return self.state.getState()
    
    #Calls an unique process depending on the state 
    def proccess(self, frame, scale, model, df, midX, laneCenter, newMemory, cameras): 
        return self.state.proccess(frame, scale, model, df, midX, laneCenter, newMemory, cameras)
    #Temp: get speed assigned to each state class 
    def getSpeed(self):
        return self.state.getSpeed()

