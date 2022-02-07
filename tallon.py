# tallon.py
#
# The code that defines the behaviour of Tallon. This is the place
# (the only place) where you should write code, using access methods
# from world.py, and using makeMove() to generate the next move.
#
# Written by: Simon Parsons
# Last Modified: 12/01/22

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
        rPit = -5
        rMeanie = -10

        bLoc = self.gameWorld.getBonusLocation()
        pLoc = self.gameWorld.getPitsLocation()
        mLoc = self.gameWorld.getMeanieLocation()

        for x in range(config.worldLength):
            for y in range(config.worldBreadth):
                state = createPose(x,y)
                
                if utils.containedIn(state, bLoc):
                    reward = rBonus
                elif utils.containedIn(state, pLoc):
                    reward = rPit
                    self.terminals.add(state)
                elif utils.containedIn(state, mLoc):
                    reward = rMeanie
                    self.terminals.add(state)
                else:
                    reward = rEmpty

                self.states.add(state)
                self.reward[state] = reward

        # show reward grid
        grid = np.empty((10,10))
        for state in self.states:
            grid[state.y, state.x] = self.reward[state]
            
        print("↓ y+   → x+")
        print(grid)

        # transition model - goal: P(s'|s,a) or T(s,a,s')
        # Stored as a list of pairs for probability and s' for each s, a
        # If a = None (i.e. terminal state) return state probability = 1
        self.transitions = {}
        for state in self.states:
            self.transitions[state] = {}
            # if state is terminal define T here?
            for action in self.A(state):
                self.transitions[state][action] = self.calcT(state, action)

        U = self.valueIteration()

        # show utility grid
        Ugrid = np.empty((10,10))
        for state in self.states:
            Ugrid[state.y, state.x] = round(U[state],1)
            
        print("↓ y+   → x+")
        print(Ugrid)

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

    def whatState(self, state, direction, move):
        """Return the state if *move* is made in the intended *direction*"""
        
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

    # move methods  
    def makeMove(self):
        # This is the function you need to define

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


# for testing
if __name__ == "__main__":
    from world import World
    gameWorld = World()
    player = Tallon(gameWorld)