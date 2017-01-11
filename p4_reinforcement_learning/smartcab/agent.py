import random
import numpy as np
from sys import argv
from ast import literal_eval
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from operator import itemgetter
from copy import deepcopy

class TrialLogger(object):
    def __init__(self, active=False):
        self.__active = active
        self.trial = 0
        self.deadline = None
        self.s = None
        self.a = None
        self.r = 0
        self.q_state = ''
        self.s_prime = None
        self.a_prime = None
        self.q_table_visits = {}

    @property
    def trial(self):
        return self._trial

    @trial.setter
    def trial(self, value):
        if hasattr(self,'_trial'):
            if self.__active: print 'Logger_QTable: ' + str(self.q_table_visits)
        self._trial = value

    @property
    def s(self):
        return self._s

    @s.setter
    def s(self, value):
        if hasattr(self, '_s'):
            state_action = value + (str(self.a),)
            if state_action in self.q_table_visits: self.q_table_visits[state_action] += 1
            else: self.q_table_visits[state_action] = 1
        self._s = value

    def get_variables(self):
        return sorted([attr for attr in vars(self) if not attr.startswith("_" + self.__class__.__name__)])

    def get_var_values(self):
        values = []
        for variable in self.get_variables():
            val = self.__dict__[variable]
            if isinstance(val, float): values.append(str(round(val, 2)))
            elif isinstance(val, dict): values.append(str({k: round(v, 2) for k, v in val.items()}))
            else: values.append(str(val))
        return values

    def move_completed(self):
        print 'Logger_QTable_State: ' + str(self.s)
        print 'Logger_QTable_Action: ' + str(self.a)
        print 'Logger_QTable_SA ' + str(self.q_state)

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # TODO: Initialize any additional variables here
        self.q_table_type = 'default' # default, 128 states, or compact, 12 states
        self.action_policy = 'glie' # algorithm for choosing actions: glie or softmax
        self.epsilon = 1 # set 0 to force argmax policy
        self.tau = 1 
        self.epsilon_decay = False
        self.tau_decay = True
        self.alpha = 1
        self.gamma = 1
        if len(argv) > 1: self.set_params_from_cli()
        if len(argv) > 3: self.set_more_params_from_cli()
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.q_table = self.init_q_table()
        self.trial = -1 # is updated to 0 after reset
        self.logger = TrialLogger(active=True)

    def set_params_from_cli(self):
        try:
            self.alpha, self.gamma = float(argv[1]), float(argv[2])
        except IndexError as e:
            raise type(e)(e.message + ' -- set alpha and gamma parameters')
        except ValueError as e:
            raise type(e)(e.message + ' -- alpha and gamma should be either ints or floats')

    def set_more_params_from_cli(self):
        try:
            self.epsilon, self.epsilon_decay = float(argv[3]), literal_eval(argv[4])
            self.q_table_type, self.action_policy = argv[5], argv[6]
        except IndexError as e:
            raise type(e)(e.message + ' -- set epsilon, epsilon_decay, q_table_type, action_policy')

    def init_q_table(self):
        if self.q_table_type == 'default': return self.init_default_q_table()
        if self.q_table_type == 'compact': return self.init_compact_q_table()
        else: raise Exception('Unknown Q-Table type! Choose between default and compact.')

    def init_compact_q_table(self):
        valid_actions = self.env.valid_actions
        next_waypoints = ['right', 'left', 'forward']
        intersection_states = ['green_safe_to_move', 'green_yield_oncoming', 'red_yield', 'red_safe_to_move_right']
        all_states = [(next_waypoint, intersection_state)
                     for next_waypoint in next_waypoints
                     for intersection_state in intersection_states]
        zero_reward_actions = { action: reward for action in valid_actions for reward in len(valid_actions) * [0] }
        return { state: deepcopy(zero_reward_actions) for state in all_states }

    def init_default_q_table(self):
        valid_actions = self.env.valid_actions
        next_waypoints = ['right', 'left', 'forward', None]
        light_states = ['red', 'green']
        oncoming_traffic_states = [None, 'right', 'left', 'forward']
        left_traffic_states = [None, 'right', 'left', 'forward']
        all_states = [(next_waypoint, light_state, oncoming_traffic_state, left_traffic_state)
                     for next_waypoint in next_waypoints
                     for light_state in light_states
                     for oncoming_traffic_state in oncoming_traffic_states
                     for left_traffic_state in left_traffic_states]
        zero_reward_actions = { action: reward for action in valid_actions for reward in len(valid_actions) * [0] }
        return { state: deepcopy(zero_reward_actions) for state in all_states }

    def policy(self, state):
        if self.action_policy == 'glie': return self.glie_policy(state)
        if self.action_policy == 'softmax': return self.softmax_policy(state)
        else: raise Exception('Unknown action policy! Choose between glie and softmax.')

    def glie_policy(self, state):
        # Implementing epsilon-greedy policy
        actions = self.q_table[state].items()
        random.shuffle(actions)
        if random.random() > self.epsilon:
            best_action = max(actions, key=itemgetter(1))[0]
        else:
            zeroed_actions = [k for k,v in actions if v == 0]
            if zeroed_actions: best_action = zeroed_actions[0]
            else: best_action = random.choice(actions)[0]
        return best_action

    def softmax_policy(self, state):
        actions = self.env.valid_actions
        prob_t = len(actions) * [0]
        for a in range(0, 4):
            act = actions[a]
            prob_t[a] = np.exp(self.q_table[state][act] / float(self.tau))
        prob_t = np.true_divide(prob_t, sum(prob_t))
        return actions[self.weighted_choice(prob_t)]

    def weighted_choice(self, weights):
        totals = np.cumsum(weights)
        norm = totals[-1]
        throw = np.random.rand() * norm
        return np.searchsorted(totals, throw)

    def intersection_state(self, light, oncoming, left):
        if light == 'green' and oncoming in [None, 'left']: return 'green_safe_to_move'
        if light == 'green' and oncoming in ['right', 'forward']: return 'green_yield_oncoming'
        if light == 'red' and left == 'forward': return 'red_yield'
        if light == 'red' and left in [None, 'left', 'right']: return 'red_safe_to_move_right'
        else: raise Exception('Unknown inputs for intersection state calculation!')

    def prepare_state(self, inputs):
        light, oncoming, left = inputs['light'], inputs['oncoming'], inputs['left']
        if self.q_table_type == 'default':
            return (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'])
        elif self.q_table_type == 'compact':
            intersection_state = self.intersection_state(inputs['light'], inputs['oncoming'], inputs['left'])
            return (self.next_waypoint, intersection_state)
        else:
            raise Exception('Unknown Q-Table type! Choose between default and compact.')

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.trial += 1
        self.logger.trial = self.trial
        if self.epsilon_decay: self.decay_epsilon(self.trial)
        if self.tau_decay: self.decay_tau(self.trial)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.logger.deadline = deadline

        # TODO: Update state
        state = self.prepare_state(inputs)
        self.state = state
        self.state_history.append(state)

        # TODO: Select action according to your policy
        action = self.policy(state)

        if t > 0:
            previous_state = self.state_history[-2]
            previous_action = self.action_history[-1]
            previous_reward = self.reward_history[-1]
            log_data = [previous_action, previous_state, previous_reward, self.q_table[previous_state], action, state]
            self.logger.a, self.logger.s, self.logger.r, self.logger.q_state, self.logger.a_prime, self.logger.s_prime = log_data
            self.update_q_table(previous_state, previous_action, previous_reward, state, action)

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.action_history.append(action)
        self.reward_history.append(reward)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def decay_epsilon(self, t):
        if t < 1: self.epsilon = 1
        else: self.epsilon = 0

    def decay_tau(self, t):
        if t < 1: self.tau = 1.0
        else: self.tau = 1.0 / t

    def update_q_table(self, previous_state, previous_action, previous_reward, state, action):
        s, a, r, a_prime, s_prime = previous_state, previous_action, previous_reward, action, state
        max_q_value = max(self.q_table[s_prime].values())
        self.q_table[s][a] = (1 - self.alpha) * self.q_table[s][a] + self.alpha * (r + self.gamma * max_q_value)
        self.logger.move_completed()

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent

    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials
    
    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
