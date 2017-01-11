# argv[1]: the output file name
# argv[2]: current simulation number (to concatenate multiple simulations in one file)

from sys import argv
from collections import OrderedDict
import re

sim = 0

def init_run_metrics():
    return OrderedDict([('trial', None), ('deadline', None), ('errors', 0), ('right', 0),
                        ('left', 0), ('forward', 0), ('None', 0), ('goal_reached', False),
                        ('total_reward', 0.0), ('end_trial_deadline', None), 
                        ('simulation', sim), ('qtable', '')])

def process_file(f):
    trial_state = init_run_metrics()
    for line in f:
        line = line.strip()
        if line.startswith('Simulator.run()'):
            if trial_state['trial'] != None: 
                append_trial_results(trial_state)
                trial_state = init_run_metrics()
                trial_state['trial'] = re.search(r'Trial (\d+)', line).group(1)
            else:
                trial_state['trial'] = 0
        elif line.startswith('Environment.reset()'):
            trial_state['deadline'] = re.search(r'deadline = (.*$)', line).group(1)
        elif line.startswith('LearningAgent.update()'):
            reward = float(re.search(r'reward = (.*$)', line).group(1))
            deadline = int(re.search(r'deadline = (\d+)', line).group(1))
            action = re.search(r'action = (\w+)', line).group(1)
            if reward < 0: trial_state['errors'] += 1
            trial_state['total_reward'] += reward
            trial_state['end_trial_deadline'] = deadline
            trial_state[action] += 1
            if reward == 12.0: trial_state['goal_reached'] = True
        elif line.startswith('Logger_QTable:'):
            trial_state['qtable'] = re.search('Logger_QTable: (.*)$', line).group(1)
    append_trial_results(trial_state)

def append_trial_results(t):
    print '\t'.join(map(str, t.values()))

def main(file):
    with open(file) as f:
        if sim == 0: print '\t'.join(init_run_metrics().keys())
        process_file(f)

if __name__ == '__main__':
    sim = int(argv[2])
    main(argv[1])