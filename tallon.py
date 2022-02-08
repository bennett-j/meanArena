# tallon.py
#
# The code that defines the behaviour of Tallon. This is the place
# (the only place) where you should write code, using access methods
# from world.py, and using makeMove() to generate the next move.
#
# Written by: Simon Parsons
# Last Modified: 12/01/22

import world
import random
import utils
from utils import Directions, Pose
import config

from enum import Enum

import numpy as np

# this would work if x=0 and y=0 set??
"""
def createPose(self, x, y):
    self.x = x
    self.y = y

Pose.__init__ = createPose
"""

def createPose(x,y):
    p = Pose() 
    p.x = x
    p.y = y
    return p




# Representation of moves
class Moves(Enum):
    AHEAD = 0
    LEFT = 1
    RIGHT  = 2


class MeanieTransitionModel():
    def __init__(self, grid):
        # save grid instance as attrubute
        self.g = grid

    def getTransitionProbs(self, mLoc, tLoc):
        """Given Tallon's current location, accounting for Tallon's possible moves,
        what is the probability distribution of states a meanie may travel to.
        """
         
        # meanie motion model
        self.model = {}

        # repeat for every move Tallon could make to get possible locations
        # meanie could move to and probability of going to that state.
        # assumes equal likelyhood of each of Tallon's moves
        for direction in self.g.actions:
            # get state Tallon will be in if action executed
            tState1 = self.g.findState(tLoc, direction)
            # save current mLoc
            # x = mLoc.x
            # y = mLoc.y

            # behaviour varies upon whether Tallon within sense distance
            if utils.separation(mLoc, tState1) < config.senseDistance:
                # moveToTallon
                # If same x-coordinate, move in the y direction
                if mLoc.x == tState1.x:
                    y1 = self.reduceDifference(mLoc.y, tState1.y)
                    self.updateModel(self.g.findState(createPose(mLoc.x, y1)), 1.0)

                # If same y-coordinate, move in the x direction
                elif mLoc.y == tState1.y:
                    x1 = self.reduceDifference(mLoc.x, tState1.x)
                    self.updateModel(self.g.findState(createPose(x1, mLoc.y)), 1.0)

                # If x and y both differ, approximate a diagonal
                # approach by randomising between moving in the x and
                # y direction.
                else:
                    y1 = self.reduceDifference(mLoc.y, tState1.y)
                    self.updateModel(self.g.findState(createPose(mLoc.x, y1)), 0.5)
                    x1 = self.reduceDifference(mLoc.x, tState1.x)
                    self.updateModel(self.g.findState(createPose(x1, mLoc.y)), 0.5)
            
            else:
                # makeRandomMove
                # P(N),S,E,W = 1/6, P(stay)=1/3
                # if direction hits a wall then it stays

                # probability of staying in current state
                self.updateModel(self.g.findState(mLoc), 2/6)
                # probability of NSWE
                # if one action is into a wall and returns this state, it will be added via the method
                for a in self.g.actions:
                    s = self.g.findState(mLoc, a)
                    self.updateModel(s, 1/6)
        
        return self.normaliseModel(self.model)

    def normaliseModel(self, model):
        # normalise
        sumPs = sum(model.values())
        for s in model:
            #print((s.x, s.y), ": ", model[s], end=', ')
            model[s] /= sumPs
            #print(model[s])
        return model

    def updateModel(self, s, p):
        """If state exists in model, probability is added. 
        If the state doesn't exit, an entry is added with corresponding probability.
        """
        if s in self.model:
            self.model[s] += p
        else:
            self.model[s] = p
  
    # Move value towards target. FOR MEANIE
    def reduceDifference(self, value, target):
        if value < target:
            return value+1
        elif value > target:
            return value-1
        else:
            return value


class Grid():
    def __init__(self):
        # initialise attributes
        self.states = set()
        self.actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        
        # populate states
        # create a state for every grid square
        for x in range(config.worldLength):
            for y in range(config.worldBreadth):
                # create state and add to set
                state = createPose(x,y)
                self.states.add(state)

    def findState(self, pose, direction=None, move=None):
        """Supply a pose (needn't be a state). Return the state if *move* is made in the intended *direction*. 
        If neither defined, returns the state from self.states at equivalent location.
        """
        # calculate new coordinates according to direction
        if direction is not None:
            
            x = pose.x
            y = pose.y

            # correct actual direction according to move left, right, ahead
            if move is not None:
                direction = self.correctDirection(direction, move)
            
            # update coordinate
            if direction == Directions.SOUTH:
                y += 1
                
            if direction == Directions.NORTH:
                y -= 1
                    
            if direction == Directions.EAST:
                x += 1
                    
            if direction == Directions.WEST:
                x -= 1

            # turn location into a state
            newPose = createPose(x,y)

        else:
            newPose = createPose(pose.x, pose.y)

        # find the state that is at this location
        # (for efficiency do this first rather than checking state exists in grid)
        for s in self.states:
            if utils.sameLocation(newPose, s):
                return s
        
        # if the state hasn't been found in the grid, Tallon has hit a wall
        # Tallon stays in same location, so return state of current location
        for s in self.states:
            if utils.sameLocation(pose, s):
                return s

        # if we get to here and still haven't found a matching state then something is wrong           
        raise IndexError("Something is wrong. The supplied pose was not found in states.")

    def correctDirection(self, direction, move):
        # get actual direction - i.e. if direction = north but move = left then return west
        if direction == Directions.NORTH:
            if move == Moves.AHEAD:
                return direction
            elif move == Moves.LEFT:
                return Directions.WEST
            elif move == Moves.RIGHT:
                return Directions.EAST
            else:
                raise ValueError("Move ", move, " is not valid.")
        
        if direction == Directions.SOUTH:
            if move == Moves.AHEAD:
                return direction
            elif move == Moves.LEFT:
                return Directions.EAST
            elif move == Moves.RIGHT:
                return Directions.WEST
            else:
                raise ValueError("Move ", move, " is not valid.")
        
        if direction == Directions.WEST:
            if move == Moves.AHEAD:
                return direction
            elif move == Moves.LEFT:
                return Directions.SOUTH
            elif move == Moves.RIGHT:
                return Directions.NORTH
            else:
                raise ValueError("Move ", move, " is not valid.")

        if direction == Directions.EAST:
            if move == Moves.AHEAD:
                return direction
            elif move == Moves.LEFT:
                return Directions.NORTH
            elif move == Moves.RIGHT:
                return Directions.SOUTH
            else:
                raise ValueError("Move ", move, " is not valid.")

###################################                
    

class Tallon():

    def __init__(self, arena, gamma):

        # Make a copy of the world an attribute, so that Tallon can
        # query the state of the world
        self.gameWorld = arena

        self.gamma = gamma

        self.g = Grid()
        self.meanieModel = MeanieTransitionModel(self.g)

        self.terminals = set()
        self.transitions = {}
        self.reward = {}

        if config.partialVisibility:
            self.savedPits = set()
            self.savedBonuses = set()
            self.exploredStates = set()
            self.bonusesRemaining = config.numberOfBonuses
            self.fullyExplored = False

    def makeMove(self):
        # This is the function you need to define

        

        self.refreshGrid()
        #self.dispReward()
        U = self.valueIteration(self.gamma)

        tLoc = self.gameWorld.getTallonLocation()
        tState = self.g.findState(tLoc)

        # for debugging
        #pi = self.optimalPolicy(U)
        #self.displayGrids(U,pi)
        
        #self.dispReward()
        ######

        

        # we get the move which maximises the expected utility from this state
        # essentially computing the policy just for one space        
        move = max(self.A(tState), key=lambda a: self.expectedUtility(U,tState,a))

        #debugging 
        #print("Tallon Location: ", end='')
        #tLoc.print()
        #print("Move: ", move)

        return move

    def refreshGrid(self):
        # clear terminals, transitions, reward
        self.terminals = set()
        self.transitions = {}
        self.reward = {}

        # find location of items
        bLoc = self.gameWorld.getBonusLocation()
        pLoc = self.gameWorld.getPitsLocation()
        mLoc = self.gameWorld.getMeanieLocation()
        tLoc = self.gameWorld.getTallonLocation()

        # update saved locations
        if config.partialVisibility:
            # if we collected a bonus on the last move, remove it from the saved bonus set
            if self.gameWorld.justGrabbed():
                for bonus in self.savedBonuses:
                    if utils.sameLocation(tLoc, bonus):
                        self.savedBonuses.remove(bonus)
                        break

            # save new pit and bonus locations and update from saved sets
            # not concerned with saving meanie locations since they move
            for b in bLoc: self.savedBonuses.add(b)
            for p in pLoc: self.savedPits.add(p)
            bLoc = list(self.savedBonuses)
            pLoc = list(self.savedPits)

            # add any visible locations to the saved state
            if not self.fullyExplored:
                for s in self.g.states:
                    if utils.separation(s, tLoc) <= config.visibilityLimit:
                        self.exploredStates.add(s) # won't add duplicates
                
                # determine if world fully explored
                # saves computation on future iterations
                if len(self.g.states) == len(self.exploredStates):
                    self.fullyExplored = True
                    print("Fully Explored")

        # if a pit or meanie, add to terminal states
        terminalLocs = pLoc + mLoc # join lists
        for s in self.g.states:
            if utils.containedIn(s, terminalLocs):
                self.terminals.add(s)
    
        # create tallon transistion model
        # transition model - goal: P(s'|s,a) or T(s,a,s')
        # Stored as a list of pairs for probability and s' for each s, a
        # If a = None (i.e. terminal state) return state probability = 1
        for state in self.g.states:
            self.transitions[state] = {}
            # if state is terminal define T here?
            for action in self.A(state):
                self.transitions[state][action] = self.calcT(state, action) 

        # calculate rewards
        # define rewards structure
        rEmpty = 1/config.scoreInterval
        rBonus = config.bonusValue
        rPit = -1 #-5
        rMeanie = -10 #-10,-1 # balance between being afraid so not going near and being so afraid it runs into walls
        
        if config.partialVisibility:
            # reward for unexplored is available bonus points divided amongst all unexplored spaces
            # once all rewards collected no incentive to explore over empty space reward
            # does not consider pits exist in unexplored space, changing average
            if not self.fullyExplored:
                rUnexplored = rEmpty + (rBonus * self.bonusesRemaining)/(len(self.g.states)-len(self.exploredStates))
            else:
                rUnexplored = rEmpty # this won't be needed

        # populate initial reward grid
        for state in self.g.states:
            reward = 0
            if utils.containedIn(state, mLoc):
                # meanie first to overwrite bonus if on same square
                # it is a terminal state because if you move to meanie's location it will stay there
                reward = rMeanie/2 #rMeanie
            elif utils.containedIn(state, pLoc):
                reward = rPit  
            elif utils.containedIn(state, bLoc):
                reward = rBonus + rEmpty
            else:
                if config.partialVisibility:
                    if self.fullyExplored or utils.containedIn(state, self.exploredStates):
                        reward = rEmpty
                    else:
                        reward = rUnexplored

            self.reward[state] = reward
        
        # world is non-stationary; estimate possible future locations of meanies
        # weighted by their likelihood for any location Tallon could move to and update rewards
        collated = {}
        for meanieLoc in mLoc:
            # returns normalised probability, state pairs for one meanie
            probs = self.meanieModel.getTransitionProbs(meanieLoc, tLoc)
        
            for s in probs:
                if s in collated:
                    collated[s] += probs[s]
                else:
                    collated[s] = probs[s]
                
        for s in collated:
            self.reward[s] = collated[s] * rMeanie
       

    def dispReward(self):
        # show reward grid
        grid = np.empty((10,10))
        for state in self.g.states:
            grid[state.y, state.x] = round(self.reward[state],3)
            
        print("↓ y+   → x+")
        print(grid)
    
    def displayGrids(self, U, pi):
        
        self.dispReward()

        # show utility grid
        Ugrid = np.empty((10,10))
        for state in self.g.states:
            Ugrid[state.y, state.x] = round(U[state],1)
            
        print("↓ y+   → x+")
        print(Ugrid)

        # show policy grid
        Pgrid = np.empty((10,10),dtype='U')
        m = {Directions.NORTH:"▲", Directions.SOUTH:"▼",
             Directions.WEST:"◄", Directions.EAST:"►", None:"•"} #30 31 16 17
        for state in self.g.states:
            a = pi[state]
            Pgrid[state.y][state.x] = m[a]
                       
        print("↓ y+   → x+")
        print(Pgrid)

    def valueIteration(self,gamma):
        """Return U(s) for all s as a dictionary of {s: value} pairs"""

        # fig 16.6
        U1 = {s: 0 for s in self.g.states}
        #gamma = 0.9
        epsilon = 0.001
        threshold = epsilon * (1 - gamma) / gamma  #Eqn 16.12

        i = 0
        while True:
            i+=1
            U = U1.copy()
            delta = 0
            for s in self.g.states:
                # max selects max value from list
                # if a = None should all still work, returning prob 1 of staying state
                U1[s] = self.R(s) + gamma * max(sum(p * U[s1] for (p, s1) in self.T(s,a)) for a in self.A(s))
                # keep track of largest delta and keep iterating until done
                delta = max(delta, abs(U1[s] - U[s]))
            if delta <= threshold:
                # print(i)
                return U # why U not U1?

    def expectedUtility(self, U, state, action):
        return sum(p * U[s1] for (p, s1) in self.T(state, action))

    def optimalPolicy(self, U):
        pi = {}
        for s in self.g.states:
            pi[s] = max(self.A(s), key=lambda a: self.expectedUtility(U,s,a))
        return pi
            
        
    def calcT(self, state, action):
        """Returns a list of probability, state pairs. 
        Probability of transitioning to state given current state and desired action.
        """
        if action == None:
            return [(1.0, state)]
        
        else:
            pAhead = config.directionProbability
            pSide = (1-pAhead)/2
            
            return [(pAhead, self.g.findState(state, action, Moves.AHEAD)),
                    (pSide, self.g.findState(state, action, Moves.LEFT)),
                    (pSide, self.g.findState(state, action, Moves.RIGHT))]
    
    #
    # access methods
    #

    def A(self, state):
        """Return a list of actions available from this state. The actions are the same for all states except absorbing states."""
        if state in self.terminals:
            return [None]
        else:
            return self.g.actions

    def R(self, state):
        return self.reward[state]

    def T(self, state, action):
        return self.transitions[state][action]

    
# for testing
if __name__ == "__main__":
    from world import World
    gameWorld = World()
    player = Tallon(gameWorld)
    player.createMDP()
    #player.meanieT()
    player.dispReward()