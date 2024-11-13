import gym
import numpy as np  
from tqdm import tqdm

##################
# this function learns the state value function by using the Monte Carlo method
##################
# inputs: 
##################
# env - OpenAI Gym environment 
# stateNumber - number of states
# numberOfEpisodes - number of simulation episodes
# discountRate - discount rate 
##################
# outputs:
##################
# valueFunctionEstimate - final learned value of the state value function vector 
##################



def MonteCarloLearnStateValueFunction(env,stateNumber,numberOfEpisodes,discountRate):
    import numpy as np
    
    # sum of returns for every state
    sumReturnForEveryState=np.zeros(stateNumber)
    # number of visits of every state
    numberVisitsForEveryState=np.zeros(stateNumber)
    
    # estimate of the state value function vector
    valueFunctionEstimate=np.zeros(stateNumber)
    
    ###########################################################################
    # START - episode simulation
    ###########################################################################
    for indexEpisode in tqdm(range(numberOfEpisodes), total=numberOfEpisodes):
        # this list stores visited states in the current episode
        visitedStatesInEpisode=[]
        # this list stores the return in every visited state in the current episode
        rewardInVisitedState=[]
        (currentState,prob)=env.reset()
        visitedStatesInEpisode.append(currentState)
        
        
        ###########################################################################
        # START - single episode simulation
        ###########################################################################
        # here we randomly generate actions and step according to these actions
        # when the terminal state is reached, this loop breaks
        while True:
            
            # select a random action
            randomAction= env.action_space.sample()
            
            # explanation of "env.action_space.sample()"
            # Accepts an action and returns either a tuple (observation, reward, terminated, truncated, info)
            # https://www.gymlibrary.dev/api/core/#gym.Env.step
            # format of returnValue is (observation,reward, terminated, truncated, info)
            # observation (object)  - observed state
            # reward (float)        - reward that is the result of taking the action
            # terminated (bool)     - is it a terminal state
            # truncated (bool)      - it is not important in our case
            # info (dictionary)     - in our case transition probability
            # env.render()
            
            # here we step and return the state, reward, and boolean denoting if the state is a terminal state
            (currentState, currentReward, terminalState,_,_) = env.step(randomAction)          
            
            # append the reward
            rewardInVisitedState.append(currentReward)
            
            # if the current state is NOT terminal state 
            if not terminalState:
                visitedStatesInEpisode.append(currentState)   
            # if the current state IS terminal state 
            else: 
                break
            # explanation of IF-ELSE:
            # let us say that a state sequence is 
            # s0, s4, s8, s9, s10, s14, s15
            # the vector visitedStatesInEpisode is then 
            # visitedStatesInEpisode=[0,4,8,10,14]
            # note that s15 is the terminal state and this state is not included in the list
            
            # the return vector is then
            # rewardInVisitedState=[R4,R8,R10,R14,R15]
            # R4 is the first entry, meaning that this is the reward going
            # from state s0 to s4. That is, the rewards correspond to the reward
            # obtained in the destination state
        
        ###########################################################################
        # END - single episode simulation
        ###########################################################################
        
        
        # how many states we visited in an episode    
        numberOfVisitedStates=len(visitedStatesInEpisode)
            
        # this is Gt=R_{t+1}+\gamma R_{t+2} + \gamma^2 R_{t+3} + ...
        Gt=0
        # we compute this quantity by using a reverse "range" 
        # below, "range" starts from len-1 until second argument +1, that is until 0
        # with the step that is equal to the third argument, that is, equal to -1
        # here we do everything backwards since it is easier and faster 
        # if we go in the forward direction, we would have to 
        # compute the total return for every state, and this will be less efficient
        
        for indexCurrentState in range(numberOfVisitedStates-1,-1,-1):
                
            stateTmp=visitedStatesInEpisode[indexCurrentState] 
            returnTmp=rewardInVisitedState[indexCurrentState]
              
            # this is an elegant way of summing the returns backwards 
              
            Gt=discountRate*Gt+returnTmp
              
            # below is the first visit implementation 
            # here we say that if visitedStatesInEpisode[indexCurrentState] is 
            # visited for the first time, then we only count that visit
            # here also note that the notation a[0:3], includes a[0],a[1],a[2] and it does NOT include a[3]
            if stateTmp not in visitedStatesInEpisode[0:indexCurrentState]:
                # note that this state is visited in the episode
                numberVisitsForEveryState[stateTmp]=numberVisitsForEveryState[stateTmp]+1
                # add the sum for that state to the total sum for that state
                sumReturnForEveryState[stateTmp]=sumReturnForEveryState[stateTmp]+Gt
            
    
    ###########################################################################
    # END - episode simulation
    ###########################################################################
    
    # finally we need to compute the final estimate of the state value function vector
    for indexSum in range(stateNumber):
        if numberVisitsForEveryState[indexSum] !=0:
            valueFunctionEstimate[indexSum]=sumReturnForEveryState[indexSum]/numberVisitsForEveryState[indexSum]
        
    return valueFunctionEstimate
            


##################
# this function computes the state value function by using the iterative policy evaluation algorithm
##################
# inputs: 
##################
# env - environment 
# valueFunctionVector - initial state value function vector
# policy - policy to be evaluated - this is a matrix with the dimensions (number of states)x(number of actions)
#        - p,q entry of this matrix is the probability of selection action q in state p
# discountRate - discount rate 
# maxNumberOfIterations - max number of iterations of the iterative policy evaluation algorithm
# convergenceTolerance - convergence tolerance of the iterative policy evaluation algorithm
##################
# outputs:
##################
# valueFunctionVector - final value of the state value function vector 
##################

def evaluatePolicy(env,valueFunctionVector,policy,discountRate,maxNumberOfIterations,convergenceTolerance):
    import numpy as np
    convergenceTrack=[]
    for iterations in range(maxNumberOfIterations):
        convergenceTrack.append(np.linalg.norm(valueFunctionVector,2))
        valueFunctionVectorNextIteration=np.zeros(env.observation_space.n)
        for state in env.P:
            outerSum=0
            for action in env.P[state]:
                innerSum=0
                for probability, nextState, reward, isTerminalState in env.P[state][action]:
                    #print(probability, nextState, reward, isTerminalState)
                    innerSum=innerSum+ probability*(reward+discountRate*valueFunctionVector[nextState])
                outerSum=outerSum+policy[state,action]*innerSum
            valueFunctionVectorNextIteration[state]=outerSum
        if(np.max(np.abs(valueFunctionVectorNextIteration-valueFunctionVector))<convergenceTolerance):
            valueFunctionVector=valueFunctionVectorNextIteration
            print('Iterative policy evaluation algorithm converged!')
            break
        valueFunctionVector=valueFunctionVectorNextIteration       
    return valueFunctionVector

##################



##################
# this function visualizes and saves the state value function 
##################
# inputs: 
##################
# valueFunction - state value function vector to plot
# reshapeDim - reshape dimension
# fileNameToSave - file name to save the figure
def grid_print(valueFunction,reshapeDim,fileNameToSave):
    import seaborn as sns
    import matplotlib.pyplot as plt  
    ax = sns.heatmap(valueFunction.reshape(reshapeDim,reshapeDim),
                     annot=True, square=True,
                     cbar=False, cmap='Blues',
                     xticklabels=False, yticklabels=False)
    plt.savefig(fileNameToSave,dpi=600)
    plt.show()
##################


def create_policy_from_value_function(env, value_function):
    policy = np.zeros((env.observation_space.n, env.action_space.n))
    for state in range(env.observation_space.n):
        q_values = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for prob, next_state, reward, done in env.P[state][action]:
                q_values[action] += prob * (reward + discountRate * value_function[next_state])
        best_action = np.argmax(q_values)
        policy[state, best_action] = 1.0
    return policy

def test_policy(env, policy, test_episodes=100):
    wins = 0
    for _ in range(test_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = np.argmax(policy[state])
            state, reward, done, _, _ = env.step(action)
        if state == env.observation_space.n - 1:
            wins += 1
    return wins / test_episodes


# create an environment 

# note here that we create only a single hole to makes sure that we do not need
# a large number of simulations
# generate a custom Frozen Lake environment
desc=["SFFF", "FFFF", "FFFF", "HFFG"]

# here we render the environment- use this only for illustration purposes
# env=gym.make('FrozenLake-v1', desc=desc, map_name="4x4", is_slippery=True,render_mode="human")

# uncomment this and comment the previous line in the case of a large number of simulations
env=gym.make('FrozenLake-v1', desc=desc, map_name="4x4", is_slippery=True)

# number of states in the environment
stateNumber=env.observation_space.n

# number of simulation episodes
numberOfEpisodes=10000

# discount rate
discountRate=1
# estimate the state value function by using the Monte Carlo method
estimatedValuesMonteCarlo=MonteCarloLearnStateValueFunction(env,stateNumber=stateNumber,numberOfEpisodes=numberOfEpisodes,discountRate=discountRate)


# for comparison compute the state value function vector by using the iterative policy 
# evaluation algorithm

# select an initial policy
# initial policy starts with a completely random policy
# that is, in every state, there is an equal probability of choosing a particular action
initialPolicy=(1/4)*np.ones((16,4))

# initialize the value function vector
valueFunctionVectorInitial=np.zeros(env.observation_space.n)
# maximum number of iterations of the iterative policy evaluation algorithm
maxNumberOfIterationsOfIterativePolicyEvaluation=1000
# convergence tolerance 
convergenceToleranceIterativePolicyEvaluation=10**(-6)

# the iterative policy evaluation algorithm
valueFunctionIterativePolicyEvaluation=evaluatePolicy(env,valueFunctionVectorInitial,initialPolicy,1,maxNumberOfIterationsOfIterativePolicyEvaluation,convergenceToleranceIterativePolicyEvaluation)

# plot the result
grid_print(valueFunctionIterativePolicyEvaluation,reshapeDim=4,fileNameToSave='iterativePolicyEvaluationEstimated.png')

# plot the result
grid_print(estimatedValuesMonteCarlo,reshapeDim=4,fileNameToSave='monteCarloEstimated.png')

env=gym.make('FrozenLake-v1', desc=desc, map_name="4x4", is_slippery=True, render_mode='human')

# Créer une politique à partir des valeurs d'état
policy = create_policy_from_value_function(env, estimatedValuesMonteCarlo)

# Tester la politique et calculer le taux de victoire
win_rate = test_policy(env, policy, test_episodes=1000)

print(f"Win rate: {win_rate * 100}%")


# 8x8
# desc = ["SFFFFFFF", 
#         "FFFFFFFF", 
#         "FFFFFFFF", 
#         "FFFFFFFF", 
#         "FFFFFFFF", 
#         "FFFFFFFF", 
#         "FFFFFFFF", 
#         "HFFFFFFG"]

# env = gym.make('FrozenLake-v1', desc=desc, map_name="8x8", is_slippery=True)

# stateNumber = env.observation_space.n
# numberOfEpisodes = 10000
# discountRate = 1

# estimatedValuesMonteCarlo = MonteCarloLearnStateValueFunction(env, stateNumber=stateNumber, numberOfEpisodes=numberOfEpisodes, discountRate=discountRate)

# initialPolicy = (1/4) * np.ones((64, 4))
# valueFunctionVectorInitial = np.zeros(env.observation_space.n)
# maxNumberOfIterationsOfIterativePolicyEvaluation = 1000
# convergenceToleranceIterativePolicyEvaluation = 10**(-6)

# valueFunctionIterativePolicyEvaluation = evaluatePolicy(env, valueFunctionVectorInitial, initialPolicy, 1, maxNumberOfIterationsOfIterativePolicyEvaluation, convergenceToleranceIterativePolicyEvaluation)

# grid_print(valueFunctionIterativePolicyEvaluation, reshapeDim=8, fileNameToSave='iterativePolicyEvaluationEstimated_8x8.png')
# grid_print(estimatedValuesMonteCarlo, reshapeDim=8, fileNameToSave='monteCarloEstimated_8x8.png')
    
