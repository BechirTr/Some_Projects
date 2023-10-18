'''
This code was implemented in the course MAP670C Reinfocement Learning. 
This code was done to implement the algorithm proposed in the paper " Safe and efficient off-policy reinforcement learning "
'''

# Packages import
import numpy as np
import gym
import random


# Function to compute cs parameter depending on the algorithm. 
def compute_cs(algo, mu, pi, Lambda): 
    # Possible values for algo are IS, Off-policy, TB, Retrace
    if algo == "IS":
        return pi/mu
    if algo == "Off-policy":
        return Lambda
    if algo == "TB":
        return Lambda * pi
    if algo == "Retrace":
        return Lambda * min(1, pi/mu)

# The simulator that runs the algorithms
def simulator(env, algo= "Retrace", epsilon= 0.5, decay_rate= 0.99, total_episodes= 10, max_steps= 100, Lambda= 1, gamma= 1): 
    ''' function that simulate the training of an algorithm.
    
    inputs:
      env: a gym object containing the environment.
      algo: a string in this list  ["IS", "Off-policy", "TB", "Retrace"], represent the algorithme used to compute the Q-value.
      epsilon: a float between 0 and 1 for the epsilon-greedy part.
      decay_rate: a flot between 0 and 1 it represent the speed of decay of the epsilion term in the epsilon-greedy part.
      total_episodes: a positive intger represent the number of episodes to be simulated.
      max_steps: a positive intger represent the number of steps taken at each episode.
      Lambda: a float represnt the update in the Q-values.
      gamma: discount factor.

    outputs: 
      rewards: a list of rewards at each episode.
      Q: the Q values matrix.
    '''
    # Environment parameters
    n_actions= env.action_space.n # number of actions
    n_states = env.observation_space.n # number of states

    Q = np.zeros((n_states, n_actions)) # Initialize the Q matrix
    rewards = [] # List of rewards
    
    '''         The algorithm  '''
    # Simulate different episodes
    for episode in range(total_episodes):
        # Reset the environment
        state = env.reset()
        step = 0
        done = False # Boolean variable, in case the game is over to stop looping over the steps.
        total_rewards = 0
        cs = 1 
        # strat playing/interacting    
        for step in range(max_steps):
            ''' epsilon greedy part'''
            # Compute the random variable 
            t = random.uniform(0, 1)
            # We initialize the probabilty mu 
            mu = 1 
            # Exploitation choose the most likely action to be done.
            if t > epsilon:
                action = np.argmax(Q[state,:])
                mu = 1 - epsilon
            # Exploration choose a random action.
            else:
                action = env.action_space.sample()
                mu = epsilon / (n_actions - 1) 
            
            # Compute the new state, it's reward and whenever the game is over or not.
            new_state, reward, done, info = env.step(action)
            a_star = np.argmax(Q[new_state])
            probabilities = [epsilon / n_actions] * n_actions
            probabilities[a_star] += 1 - epsilon

            sum_prb = 0
            # compute cs
            cs  *=  compute_cs(algo= algo, mu= mu,
                            pi= probabilities[a_star], Lambda= Lambda) 
            
            # compute the Expectation over pi of Q.
            for i in range(n_actions):
                sum_prb += probabilities[i] * Q[new_state, i]
            
            # Update the Q(x,a) using the equation in (3) in the paper and (1) in our report. 
            Q[state, action] += (gamma**step) * cs * (reward + gamma * sum_prb - Q[state,action])
            
            # We add the new reward
            total_rewards += reward
            
            # We switch to the new state
            state = new_state
            # If done (if we're dead) : finish episode
            if done == True: 
                break
            
        # Reduce The epsilon value since with time we prefer more exploitation than exploration. 
        epsilon *= decay_rate
        rewards.append(total_rewards)

    return rewards, Q