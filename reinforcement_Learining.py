# -*- coding: UTF-8 -*-
import numpy as np
import turtle

# turtle  패키지를 사용할경우 아래 코드를 주석처리 해주십시오. 
import matplotlib.pyplot as plt  
# world height
WORLD_HEIGHT = 10
# world width
WORLD_WIDTH = 10


startState = [0, 0] #lower left corner 
goalState = [9, 9] #lower right corner

# all possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
actions = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

# reward for each action in each state
# 이 부분 역시 GUI 단에서는 reward를 받으면 알기 때문에 그냥 환경이 만들어지면 계산을 하여 줄 수 있다. 
actionRewards = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
actionRewards[:, :, :] = -1.0
actionRewards[0:7, 4, ACTION_RIGHT] = -101.0 #dive into cliff 
actionRewards[0:7, 6, ACTION_LEFT] = -101.0 #dive into cliff 
actionRewards[8, 5, ACTION_UP] = -101.0 #dive into cliff 
actionRewards[9, 8, ACTION_RIGHT] = 100000

# initial state action pair values
Q = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
accumulatedRewards = 0.0

# set up destinations for each action in each state
# 이 부분이 GUI 단에서 만들어져서 들어와야한다. 어짜피 실제 GUI와의 interactive situation에서는
# 자연스럽게 nextState와 거기서의 reward가 정해지므로 table이 필요 없다. 
# 실험 상황에서만 코드를 돌려보기 위해 필요하여 만들었을 뿐
# [next_y, next_x] = actionDestinaton[current_y][current_x][action]
actionDestination = []
for i in range(0, WORLD_HEIGHT):
    actionDestination.append([])
    for j in range(0, WORLD_WIDTH):
        destinaion = dict()
        destinaion[ACTION_UP] = [max(i - 1, 0), j]
        destinaion[ACTION_LEFT] = [i, max(j - 1, 0)]
        destinaion[ACTION_RIGHT] = [i, min(j + 1, WORLD_WIDTH - 1)]
        if i == 8 and j == 5:
            destinaion[ACTION_UP] = startState
        elif 0 <= i <= 7 and j == 6:
            destinaion[ACTION_LEFT] = startState
        elif 0 <= i <= 7 and j == 4:
            destinaion[ACTION_RIGHT] = startState
    
        destinaion[ACTION_DOWN] = [min(i + 1, WORLD_HEIGHT - 1), j]
        actionDestination[-1].append(destinaion)


# choose an action based on epsilon greedy algorithm
def chooseAction(state, Q):
    # probability for exploration
    EPSILON = 0.1
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(actions)
    else:
        return np.argmax(Q[state[0], state[1], :]) # 처음에는 무조건 0방향

# current reward, 우리는 GUI로부터 받아야 한다. 
# reward = actionRewards[currentState[0]][currentState[1]][currentAction] 
def sarsa(currentState, reward, Q, accumulatedRewards,actionDestination):
    #
    # step size (learning rate)
    stepSize = 0.5
    # gamma for Q-Learning and Expected Sarsa
    GAMMA = 1
    
    #1 Choose current action
    currentAction = chooseAction(currentState, Q)
    nextState = actionDestination[currentState[0]][currentState[1]][currentAction]
    #2 choose newAction from the newState
    #################################################################################
    # your code here ###############################################################
    nextAction = chooseAction(nextState, Q)
    #################################################################################
    
    #3 Sarsa update
    #################################################################################
    # your code here ###############################################################
    Q[currentState[0]][currentState[1]][currentAction] = \
    Q[currentState[0]][currentState[1]][currentAction] + \
    stepSize * (reward + GAMMA * Q[nextState[0]][nextState[1]][nextAction] - \
    Q[currentState[0]][currentState[1]][currentAction])
    #################################################################################
        
    #5 Compute the accumulated reward for comparison (plotting)        
    accumulatedRewards += reward 
        
    return currentAction, Q, accumulatedRewards

Nepisode = 500
score = np.zeros(Nepisode)

def fromgui(state,action,actionDestination,actionRewards):
    # GUI 에서는 현재 state와 action이 주어졌을 때, nextState와 reward를 알 수 있고 그냥 이걸 return하면 되는데 
    # 현재 여기서만 당장 돌려야하니까 actionDestination table을 만들어서 거기서 뽑아올 뿐이다. 
    # reward도 마찬가지
    nextState = actionDestination[state[0]][state[1]][action]
    reward = actionRewards[state[0]][state[1]][action] 
    return nextState, reward

    # perform 20 independent runs
runs = 20
testStateTable = ""
mouse_movemnet = []
for run in range(0, runs):
    stateActionValuesSarsa = np.copy(Q)
    for e in range(Nepisode):
        currentState = startState
        mouse_movemnet.append(currentState)
        reward = 0
        accumulatedRewards = 0
        while currentState != goalState:
            currentAction, stateActionValuesSarsa, accumulatedRewards = sarsa(currentState, reward, stateActionValuesSarsa, accumulatedRewards,actionDestination)
            # From GUI we should retrieve the reward and the next currentState
            nextState, reward = fromgui(currentState,currentAction,actionDestination,actionRewards)
            currentState = nextState
            mouse_movemnet.append(currentState)
        score[e] += max(accumulatedRewards, -100)
    # turtle 패키지를 이용하여 움직임을 보고싶을경우, matplotlib을 주석처리하고 아래 코드를 주석제거 해주십시요
    """ 
    myTurtle = turtle.Turtle()   
    myTurtle.speed(9)
    myTurtle.screen.screensize(500, 500)
    myTurtle.goto(-250, -250)
    travel = 0
    for points in mouse_movemnet:
        myTurtle.goto((points[0]*50)-250, ((points[1])*50)-250)
        if points[0] == 0 and points[1] == 0:
            print("%d 번 이동후 목적지 도착"%(travel))
            travel = 0
        travel += 1
    """
# averaging over independt runs
score /= runs

# draw reward curves
# turtle 패키지를 사용할경우아래 코드들을 주석처리 해주십시요 
plt.figure(1)
plt.plot(score, label='Sarsa')
plt.xlabel('Episodes')
plt.ylabel('Sum of rewards during episode')
plt.legend()
plt.show()