import numpy as np
import matplotlib.pyplot as plt
import copy

from src.game import Game 
from src.learning_rule import MardenMoodRule
from src.plot_utils import HistoryAnalysisMixin

class UnifiedLearning(HistoryAnalysisMixin):
    """
    Implements the algorithm Unified Learning Framework for a multi-agent game with finite horizon and two players.
    """
    def __init__(self, game, T, learning_rule, save=False, save_path=None, no_override=False):
        self.T = T          # number of learning iterations
        self.learning_rule = learning_rule

        if learning_rule.norm_rewards:
            self.game = self._normalize_rewards(game, learning_rule.reward_prec)
        else:
            self.game = game

        #output parameters
        self._save = save
        self._save_path = save_path
        self._no_override =no_override


        # Definition of variables

        # Q[player][stage h][state_index][action_pl1][action_pl2]
        self.Q = np.zeros((self.game.N, self.game.H + 1, len(self.game.s_map[2]), len(self.game.actions), len(self.game.actions)))

        # V[player][stage h][state_index]
        self.V = np.zeros((self.game.N, self.game.H + 2, len(self.game.s_map[2])))
        
        # a[stage h][state_index] -> (a1, a2)
        self.a = {}

        # xi[stage h][state_index] -> (xi_1, xi_2)
        self.hidden = {}

        # Save cronology of the state s1 to check convergence
        self.V_history = []             # V-value just of player 0 (we have symmetric games)
        self.s1_action_history = []     # pair of actions taken by both players

    def _normalize_rewards(self,game: Game, prec: int) -> Game:
        g = copy.deepcopy(game)
        
        max_val = 0.0
        for stage_data in g.rewards.values():
            for reward_matrix in stage_data.values():
                max_val = max(max_val, np.max(np.abs(reward_matrix)))

        if max_val == 0:
            return g  # nothing to normalize

        normalized = {}
        for stage, stage_data in g.rewards.items():
            normalized[stage] = {}
            for state, reward_matrix in stage_data.items():
                normalized[stage][state] = np.round(reward_matrix / max_val,prec)
        g.rewards = normalized

        return g

    def _initialize(self):
        """ Initialisation of Q-values, actions and hidden variables. """
        
        actions = self.game.actions

        # Q values are initialised to rewards
        for h in range(1, self.game.H + 1):
            for s_str, reward_matrix in self.game.rewards[h].items():
                s_idx = self.game.s_map[h][s_str]
                for a1 in actions:
                    for a2 in actions:
                        reward = reward_matrix[a1, a2]

                        for i in range(self.game.N):
                            self.Q[i, h, s_idx, a1, a2] = reward[i]

        # Actions and hidden variables are initialised randomly
        for h in range(1, self.game.H + 1):
            self.a[h] = {}
            self.hidden[h] = {}
            for _, s_idx in self.game.s_map[h].items():

                a1_rand = np.random.choice(actions)
                a2_rand = np.random.choice(actions)
                self.a[h][s_idx] = [a1_rand, a2_rand]
                
                if isinstance(self.learning_rule, MardenMoodRule):
                    hidd1_rand = np.random.choice(['C','D'])
                    hidd2_rand = np.random.choice(['C','D'])
                    self.hidden[h][s_idx] = [hidd1_rand, hidd2_rand]
                else:
                    self.hidden[h][s_idx] = [0.0, 0.0]


    def run(self):
        """ Run of the main learning cycle. """
        self._initialize()

        for t in range(self.T):

            V_t = np.copy(self.V)
            self.V_history.append(V_t[0, 1, 0])

            for h in range(self.game.H, 0, -1):
                
                # Actor: computes new actions and new auxiliary variables for all the states in stage h, using Q^(t)
                new_action_h = {}   #a_h_t_plus
                new_hidden_h = {}

                for s_str, s_idx in self.game.s_map[h].items():
                    current_a = self.a[h][s_idx]
                    current_hid = self.hidden[h][s_idx]

                    q_vals = self.Q[:, h, s_idx, :, :]
                             
                    new_action_h[s_idx], new_hidden_h[s_idx] = self.learning_rule.update_vars(current_a, current_hid, self.game.N, self.game.actions, q_vals)


                # Critic: updates V_{i,h} and Q_{i,h} for all the states in stage h, based on the new actions
                for s_str, s_idx in self.game.s_map[h].items():
                                        
                    # V-values update
                    t_joint_action = self.a[h][s_idx] 
                    for i in range(self.game.N):
                        q_val_t = self.Q[i, h, s_idx, t_joint_action[0], t_joint_action[1]]
                        
                        # Calcola la media mobile
                        if t == 0:
                           self.V[i, h, s_idx] = q_val_t
                        else:
                           old_v = V_t[i, h, s_idx]
                           self.V[i, h, s_idx] = (t / (t + 1)) * old_v + (1 / (t + 1)) * q_val_t

                    # Q-values update
                    for i in range(self.game.N):
                        for a1 in self.game.actions:
                            for a2 in self.game.actions:
                                expected_V = 0
                                if h < self.game.H: # Per h=1, calcola il valore atteso da h=2
                                    next_s_str = self.game.transition(a1, a2)
                                    next_s_idx = self.game.s_map[h + 1][next_s_str]
                                    expected_V = self.V[i, h + 1, next_s_idx]
                                
                                reward = self.game.rewards[h][s_str][a1,a2]
                                self.Q[i, h, s_idx, a1, a2] = reward[i] + expected_V

                # Save variables new values
                self.a[h] = new_action_h  
                self.hidden[h] = new_hidden_h              
            
            # Save history of the initial state
            action_in_s1 = self.a[1][0]         # actions taken in h=1, s_idx=0

            self.s1_action_history.append(action_in_s1)

    # def run_experiments(self, num_runs):
    #     all_runs = []

    #     for _ in range(num_runs):
    #         self.reset()    #DA DEFINIRE, CAPIRE COME
    #         self.run()   # esegue una traiettoria, riempiendo self.s1_action_history
    #         all_runs.append(self.s1_action_history.copy())

    #     return all_runs
