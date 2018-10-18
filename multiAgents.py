# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

#check
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        prevFood = currentGameState.getFood()
        currentPos = list(successorGameState.getPacmanPosition())
        maxDistance = -10000000
        distance = 0
        foodList = prevFood.asList()
        gGhostDist = []

        if action == Directions.STOP:
            return float("-inf")

        for states in newGhostStates:
           if states.getPosition() == tuple(currentPos) and (states.scaredTimer == 0):
               return float("-inf")
        # Find the distance of all the foods to the pacman
        for food in foodList:
            distance = -1 * (manhattanDistance(food, currentPos))


            if (distance > maxDistance):
                maxDistance = distance

        return maxDistance

        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        def max_value(gameState, depth):
            actList = gameState.getLegalActions(0)  # Get actions of pacman
            if len(actList) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)

            val = -(float("inf"))
            goAction = None

            for thisAction in actList:
                successorValue = min_value(gameState.generateSuccessor(0, thisAction), 1, depth)[0]
                if (successorValue > val):
                    val, goAction = successorValue, thisAction
            return (val, goAction)

        def min_value(gameState, agentID, depth):
            actList = gameState.getLegalActions(agentID)  # Get the actions of the ghost
            if len(actList) == 0:
                return (self.evaluationFunction(gameState), None)
            val = float("inf")
            goAction = None

            for thisAction in actList:
                if (agentID == gameState.getNumAgents() - 1):
                    successorValue = max_value(gameState.generateSuccessor(agentID, thisAction), depth + 1)[0]
                else:
                    successorValue = min_value(gameState.generateSuccessor(agentID, thisAction), agentID + 1, depth)[0]
                if (successorValue < val):
                    val, goAction = successorValue, thisAction
            return (val, goAction)

        return max_value(gameState, 0)[1]
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = 0
        alpha = float('-inf')
        beta = float('inf')
        return self.getMaxValue(gameState, alpha, beta, depth)[1]

    def getMaxValue(self, gameState, alpha, beta, depth, agent=0):
        actions = gameState.getLegalActions(agent)

        if not actions or gameState.isWin() or depth >= self.depth:
            return self.evaluationFunction(gameState), Directions.STOP

        successorCost = float('-inf')
        successorAction = Directions.STOP

        for action in actions:
            successr = gameState.generateSuccessor(agent, action)
            costaction = self.getMinValue(successr, alpha, beta, depth, agent + 1)[0]

            if costaction > successorCost:
                successorCost = costaction
                successorAction = action

            if successorCost > beta:
                return successorCost, successorAction

            alpha = max(alpha, successorCost)

        return successorCost, successorAction

    def getMinValue(self, gameState, alpha, beta, depth, agent):
        actions = gameState.getLegalActions(agent)

        if not actions or gameState.isLose() or depth >= self.depth:
            return self.evaluationFunction(gameState), Directions.STOP

        successorCost = float('inf')
        successorAction = Directions.STOP

        for action in actions:
            successr = gameState.generateSuccessor(agent, action)

            costaction = 0

            if agent == gameState.getNumAgents() - 1:
                costaction = self.getMaxValue(successr, alpha, beta, depth + 1)[0]
            else:
                costaction = self.getMinValue(successr, alpha, beta, depth, agent + 1)[0]

            if costaction < successorCost:
                successorCost = costaction
                successorAction = action

            if successorCost < alpha:
                return successorCost, successorAction

            beta = min(beta, successorCost)

        return successorCost, successorAction

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        " Max Value "

        def max_value(gameState, depth):
            " Cases checking "
            actnList = gameState.getLegalActions(0)  # Get actions of pacman
            if len(actnList) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)

            " Initializing the value of v and action to be returned "
            val = -(float("inf"))
            goAction = None

            for thisAction in actnList:
                successorValue = exp_value(gameState.generateSuccessor(0, thisAction), 1, depth)[0]

                if (val < successorValue):
                    val, goAction = successorValue, thisAction

            return (val, goAction)

        " Exp Value "

        def exp_value(gameState, agentID, depth):
            " Cases checking "
            actnList = gameState.getLegalActions(agentID)  # Get the actions of the ghost
            if len(actnList) == 0:
                return (self.evaluationFunction(gameState), None)

            " Initializing the value of v and action to be returned "
            val = 0
            goAction = None

            for thisAction in actnList:
                if (agentID == gameState.getNumAgents() - 1):
                    successorValue = max_value(gameState.generateSuccessor(agentID, thisAction), depth + 1)[0]
                else:
                    successorValue = exp_value(gameState.generateSuccessor(agentID, thisAction), agentID + 1, depth)[0]

                probability = successorValue / len(actnList)
                val += probability

            return (val, goAction)

        return max_value(gameState, 0)[1]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    ghostList = currentGameState.getGhostStates()
    foods = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    # Return based on game state
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")
    # Populate foodDistList and find minFoodDist
    foodDistList = []
    for each in foods.asList():
        foodDistList = foodDistList + [util.manhattanDistance(each, pacmanPos)]
    minFoodDist = min(foodDistList)
    # Populate ghostDistList and scaredGhostDistList, find minGhostDist and minScaredGhostDist
    ghostDistList = []
    scaredGhostDistList = []
    for each in ghostList:
        if each.scaredTimer == 0:
            ghostDistList = ghostDistList + [util.manhattanDistance(pacmanPos, each.getPosition())]
        elif each.scaredTimer > 0:
            scaredGhostDistList = scaredGhostDistList + [util.manhattanDistance(pacmanPos, each.getPosition())]
    minGhostDist = -1
    if len(ghostDistList) > 0:
        minGhostDist = min(ghostDistList)
    minScaredGhostDist = -1
    if len(scaredGhostDistList) > 0:
        minScaredGhostDist = min(scaredGhostDistList)
    # Evaluate score
    score = scoreEvaluationFunction(currentGameState)
    # Distance to closest food
    score = score + (-1.5 * minFoodDist)
    # Distance to closest ghost
    score = score + (-2 * (1.0 / minGhostDist))
    # Distance to closest scared ghost
    score = score + (-2 * minScaredGhostDist)
    # Number of capsules
    score = score + (-20 * len(capsules))
    # Number of food
    score = score + (-4 * len(foods.asList()))
    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

