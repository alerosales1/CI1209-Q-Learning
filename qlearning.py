import random
import numpy as np

class QLearningAgent():
    def __init__(self, env, epsilon=0.01, alpha=0.8, gamma=0.2):
        self.epsilon = epsilon # exploration
        self.alpha = alpha   # learning rate
        self.gamma = gamma   # discount
        self.env = env
        self.q_table = np.zeros((env.getStateSpaceSize(),
                env.getActionSpaceSize()))

    def getAction(self, state):
        actions = self.env.getLegalActions(state)

        "*** SEU CODIGO AQUI ***"
        prob = np.random.rand(1)[0]

        if prob < self.epsilon :
            return random.choice(actions)

        else:
            max_action = -1;
            max_q = -np.inf

            for action in actions:
                q = self.q_table[state, action]
                if q > max_q:
                    max_q = q
                    max_action = action
            return max_action            

        return random.choice(actions)

    def update(self, state, action, reward, next_state):
        "*** SEU CODIGO AQUI ***"
        amostra = reward + self.gamma * np.max(self.q_table[next_state,:])
        self.q_table[state,action] = (1 - self.alpha) * self.q_table[state,action] + self.alpha * amostra
        
