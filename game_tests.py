# game.py
#
# The top level loop that runs the game until Tallon eventually loses.
#
# run this using:
#
# python3 game.py
#
# Written by: Simon Parsons
# Last Modified: 12/01/22

from world import World
from tallon  import Tallon
#from tallon_original import Tallon
from arena import Arena
import utils
import time
import numpy as np
import sys



# How we set the game up. Create a world, then connect player and
# display to it.

#display = Arena(gameWorld)

# Uncomment this for a printout of world state at the start
#utils.printGameState(gameWorld)

# Now run...
def runGame(gamma):

    gameWorld = World()
    player = Tallon(gameWorld, gamma)

    while not(gameWorld.isEnded()):
        gameWorld.updateTallon(player.makeMove())
        gameWorld.updateMeanie()
        gameWorld.updateClock()
        gameWorld.addMeanie()
        gameWorld.updateScore()
        #display.update()
        # Uncomment this for a printout of world state every step
        #utils.printGameState(gameWorld)
        #time.sleep(1)

    return gameWorld.getScore()

orig_stdout = sys.stdout
f = open('out.txt', 'w')
sys.stdout = f



iterations = 100
gammas = [0.5, 0.6, 0.7, 0.75, 0.85, 0.9, 0.95, 0.98]
data = []

for g in gammas:
    
    tStart = time.time_ns()
    sm = 0
    for i in range(iterations):
        sm += runGame(g)
    
    sys.stdout = orig_stdout
    print("Gamma: ", g)
    print("Execution Time: ", time.time_ns() - tStart)
    av = sm/iterations
    print("Average Score: ", av)
    sys.stdout = f
    data.append([g, av])
    
sys.stdout = orig_stdout
np.savetxt('data_bench', data, delimiter=',')
f.close()

