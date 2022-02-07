# tallon.py
#
# The code that defines the behaviour of Tallon. This is the place
# (the only place) where you should write code, using access methods
# from world.py, and using makeMove() to generate the next move.
#
# Written by: Simon Parsons
# Last Modified: 12/01/22

from statistics import mean
import this
from venv import create
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

def updateMmm(mmm, s, p):
    if s in mmm:
        mmm[s] += p
    else:
        mmm[s] = p
    return mmm


# Representation of moves
class Moves(Enum):
    AHEAD = 0
    LEFT = 1
    RIGHT  = 2


class Tallon():

    def __init__(self, arena):

        # Make a copy of the world an attribute, so that Tallon can
        # query the state of the world
        self.gameWorld = arena

        # What moves are possible.
        self.moves = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]

           
    def createMDP(self):
        #############
        # MDP stuff #
        #############

        self.states = set()
        self.reward = {}
        self.terminals = set()
        self.actions = self.moves # for naming continuity

        # define rewards, states, terminals
        rEmpty = 1/config.scoreInterval
        rBonus = config.bonusValue
        rPit = -1 #-5
        rMeanie = -10 #-10,-1 # balance between being afraid so not going near and being so afraid it runs into walls

        bLoc = self.gameWorld.getBonusLocation()
        pLoc = self.gameWorld.getPitsLocation()
        mLoc = self.gameWorld.getMeanieLocation()

        

        for x in range(config.worldLength):
            for y in range(config.worldBreadth):
                state = createPose(x,y)
                reward = 0
                if utils.containedIn(state, mLoc):
                    # meanie first to overwrite bonus if on same square
                    # it is a terminal state because if you move to meanie's location it will stay there
                    reward = rMeanie/2 #rMeanie
                    self.terminals.add(state)
                elif utils.containedIn(state, pLoc):
                    reward = rPit
                    self.terminals.add(state)
                elif utils.containedIn(state, bLoc):
                    reward = rBonus # + rEmpty
                else:
                    reward = rEmpty

                self.states.add(state)
                self.reward[state] = reward

        meanieProbs = self.meanieT()
        for m in meanieProbs:
            for s in m:
                self.reward[s] = m[s] * rMeanie

        # transition model - goal: P(s'|s,a) or T(s,a,s')
        # Stored as a list of pairs for probability and s' for each s, a
        # If a = None (i.e. terminal state) return state probability = 1
        self.transitions = {}
        for state in self.states:
            self.transitions[state] = {}
            # if state is terminal define T here?
            for action in self.A(state):
                self.transitions[state][action] = self.calcT(state, action)

        #self.U = self.valueIteration()
        #self.pi = self.optimalPolicy()

    def dispReward(self):
        # show reward grid
        grid = np.empty((10,10))
        for state in self.states:
            grid[state.y, state.x] = round(self.reward[state],3)
            
        print("↓ y+   → x+")
        print(grid)
    
    def displayGrids(self, U, pi):
        
        self.dispReward()

        # show utility grid
        Ugrid = np.empty((10,10))
        for state in self.states:
            Ugrid[state.y, state.x] = round(U[state],1)
            
        print("↓ y+   → x+")
        print(Ugrid)

        # show policy grid
        Pgrid = np.empty((10,10),dtype='U')
        m = {Directions.NORTH:"▲", Directions.SOUTH:"▼",
             Directions.WEST:"◄", Directions.EAST:"►", None:"•"} #30 31 16 17
        for state in self.states:
            a = pi[state]
            Pgrid[state.y][state.x] = m[a]
                       
        print("↓ y+   → x+")
        print(Pgrid)

    def valueIteration(self):
        """Return U(s) for all s as a dictionary of {s: value} pairs"""

        # fig 16.6
        U1 = {s: 0 for s in self.states}
        gamma = 0.9
        epsilon = 0.001
        threshold = epsilon * (1 - gamma) / gamma  #Eqn 16.12

        i = 0
        while True:
            i+=1
            U = U1.copy()
            delta = 0
            for s in self.states:
                # max selects max value from list
                # if a = None should all still work, returning prob 1 of staying state
                U1[s] = self.R(s) + gamma * max(sum(p * U[s1] for (p, s1) in self.T(s,a)) for a in self.A(s))
                # keep track of largest delta and keep iterating until done
                delta = max(delta, abs(U1[s] - U[s]))
            if delta <= threshold:
                print(i)
                return U # why U not U1?

    def expectedUtility(self, U, state, action):
        return sum(p * U[s1] for (p, s1) in self.T(state, action))

    def optimalPolicy(self, U):
        pi = {}
        for s in self.states:
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
            
            return [(pAhead, self.whatState(state, action, Moves.AHEAD)),
                    (pSide, self.whatState(state, action, Moves.LEFT)),
                    (pSide, self.whatState(state, action, Moves.RIGHT))]

    def whatState(self, state, direction, move=None):
        """Return the state if *move* is made in the intended *direction*"""
        
        if move is not None:
            direction = self.correctDirection(direction, move)

        x = state.x
        y = state.y

        if direction == Directions.SOUTH:
            y += 1
            
        if direction == Directions.NORTH:
            y -= 1
                
        if direction == Directions.EAST:
            x += 1
                
        if direction == Directions.WEST:
            x -= 1


        # The following process doesn't seem most efficient
        # could use limits instead of utils.containedIn (but less general)
        # inefficency subject of how states are saved and checking they're equal

        newState = createPose(x,y)

        # check if contained in
        if not utils.containedIn(newState, self.states):
            # if Tallon has hit a wall, he stays in same location
            return state

        # find the state that is at this location
        for s in self.states:
            if utils.sameLocation(newState, s):
                return s
        



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
    #
    # access methods
    #

    def A(self, state):
        """Return a list of actions available from this state. The actions are the same for all states except absorbing states."""
        if state in self.terminals:
            return [None]
        else:
            return self.actions

    def R(self, state):
        return self.reward[state]

    def T(self, state, action):
        return self.transitions[state][action]

    def getState(self, location):
        """Return state that is equivalent to location. Location supplied as a pose"""
               
        for s in self.states:
            if utils.sameLocation(location, s):
                return s

    ##################
    #  MEANIE MODEL  #
    ##################
    def meanieT(self):
        mLocs = self.gameWorld.getMeanieLocation()
        tLoc0 = self.gameWorld.getTallonLocation()
        # TODO: make a 'this state' and use throughout for tLoc

        print("Tallon Location: ", end = '')
        tLoc0.print()

        meanieProbs = []

        for mLoc in mLocs:
            
            print("Meanie Location: ", end = '')
            mLoc.print()

            # TODO this is a bodge
            mX0 = mLoc.x
            mY0 = mLoc.y

            # meanie motion model
            mmm = {}

            # repeat for every move Tallon could make to get possible locations
            # meanie could move to and probability of going to that state.
            # assumes equal likelyhood of each of Tallon's moves
            for a in self.actions:
                tLoc = self.whatState(self.getState(tLoc0), a)
                mLoc = createPose(mX0, mY0)

                if utils.separation(mLoc, tLoc) < config.senseDistance:
                    # moveToTallon
                    # If same x-coordinate, move in the y direction
                    if mLoc.x == tLoc.x:
                        mLoc.y = self.reduceDifference(mLoc.y, tLoc.y)
                        mmm = updateMmm(mmm, self.getState(mLoc), 1.0)

                    # If same y-coordinate, move in the x direction
                    elif mLoc.y == tLoc.y:
                        mLoc.x = self.reduceDifference(mLoc.x, tLoc.x)
                        mmm = updateMmm(mmm, self.getState(mLoc), 1.0)       
                    # If x and y both differ, approximate a diagonal
                    # approach by randomising between moving in the x and
                    # y direction.
                    else:
                        y = mLoc.y
                        x = mLoc.x
                        y1 = self.reduceDifference(y, tLoc.y)
                        mmm = updateMmm(mmm, self.getState(createPose(x, y1)), 0.5)
                        x1 = self.reduceDifference(x, tLoc.x)
                        mmm = updateMmm(mmm, self.getState(createPose(x1, y)), 0.5)
                else:
                    # makeRandomMove
                    # P(N),S,E,W = 1/6, P(stay)=1/3
                    # if direction hits a wall then it stays

                    # probability of staying in current state
                    mmm = updateMmm(mmm, self.getState(mLoc), 2/6)
                    # probability of NSWE
                    # if one action is into a wall and returns this state, it will be added via the method
                    for a in self.actions:
                        s = self.whatState(self.getState(mLoc), a)
                        mmm = updateMmm(mmm, s, 1/6)
            
              
           
            

            # normalise mmm 
            sumPs = sum(mmm.values())
            for s in mmm:
                print((s.x, s.y), ": ", mmm[s], end=', ')
                mmm[s] /= sumPs
                print(mmm[s])

            meanieProbs.append(mmm)

        return meanieProbs
  
    # Move value towards target. FOR MEANIE
    def reduceDifference(self, value, target):
        if value < target:
            return value+1
        elif value > target:
            return value-1
        else:
            return value

    # move methods  
    def makeMove(self):
        # This is the function you need to define

        self.createMDP()
        U = self.valueIteration()

        # for debugging
        pi = self.optimalPolicy(U)
        self.displayGrids(U,pi)
        ######

        tLoc = self.gameWorld.getTallonLocation()
        tState = self.getState(tLoc)

        # we get the move which maximises the expected utility from this state
        # essentially computing the policy just for one space        
        move = max(self.A(tState), key=lambda a: self.expectedUtility(U,tState,a))
        return move

        '''
        # For now we have a placeholder, which always moves Tallon
        # directly towards any existing bonuses. It ignores Meanies
        # and pits.
        # 
        # Get the location of the Bonuses.
        allBonuses = self.gameWorld.getBonusLocation()

        # if there are still bonuses, move towards the next one.
        if len(allBonuses) > 0:
            nextBonus = allBonuses[0]
            myPosition = self.gameWorld.getTallonLocation()
            # If not at the same x coordinate, reduce the difference
            if nextBonus.x > myPosition.x:
                return Directions.EAST
            if nextBonus.x < myPosition.x:
                return Directions.WEST
            # If not at the same y coordinate, reduce the difference
            if nextBonus.y < myPosition.y:
                return Directions.NORTH
            if nextBonus.y > myPosition.y:
                return Directions.SOUTH

        # if there are no more bonuses, Tallon doesn't move
        '''

# for testing
if __name__ == "__main__":
    from world import World
    gameWorld = World()
    player = Tallon(gameWorld)
    player.createMDP()
    #player.meanieT()
    player.dispReward()