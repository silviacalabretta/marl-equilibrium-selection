import numpy as np
import matplotlib.pyplot as plt
import numbers
import os
import re


def generate_plot(save, save_path=None, default_name="plot.png", no_override=False):
    if save:
        if save_path is None:
            save_path = os.path.join("out", default_name)
        
        save_plot(save_path=save_path, no_override=no_override)
    else:
        plt.show()

def save_plot(save_path, no_override=False):
        """
        Auxiliary function to handle saving or showing plots.
        save_path can be:   True -> plot is saved in OUTPUT_DIR/filename
                            string -> plot is saved in save_path (it is expected to contain the filename)
        no_override(bool):  if True, checks for files with same filename and adds an index to it to avoid overrides
                            if False, overrides existing file with the same filename
        """

        final_path = save_path
        if no_override:
            final_path = _find_unique_path(final_path)

        output_dir = os.path.dirname(final_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving figure to {final_path}")
        plt.savefig(final_path)
        plt.close() 
        return

def _find_unique_path(path):
    """
    Check if a file already exists and returns a unique path adding a unique index
    Example: plot.png -> plot1.png -> plot2.png
    """
    
    directory = os.path.dirname(path)
    base_filename, extension = os.path.splitext(os.path.basename(path))
    
    # if directory doesn't exist, no need to check for duplicates
    if not os.path.exists(directory):
        return path

    # looks for base_filename + number, Pattern: base_filename (number)? .extension
    pattern = re.compile(r"^" + re.escape(base_filename) + r"(_(\d+))?" + re.escape(extension) + r"$")
    
    existing_files = [f for f in os.listdir(directory) if pattern.match(f)]
    
    if not existing_files:
        return path

    # if there are duplicates, find the highest index
    max_index = 0
    for filename in existing_files:
        match = re.search(r"((\d+))" + re.escape(extension) + r"$", filename)
        if match:
            index = int(match.group(1))
            if index > max_index:
                max_index = index
        elif filename == base_filename + extension:
            max_index = max(max_index, 0)
    
    next_index = max_index + 1

    unique_filename = f"{base_filename}_{next_index}{extension}"
    return os.path.join(directory, unique_filename)




class HistoryAnalysisMixin:
    """
    Provides storage and plotting utilities for learning histories.
    Requires:
        self.V_history : list[float]
        self.runs_history : list[list[tuple]]
        self.T : int
        self._save : bool
        self._save_path : str
        self._no_override : bool
    """

    def print_results(self):
        """ For each stage and state, prints the final V-values and Q-values learnt by player 0, and the pair of actions learnt. """
        print("\n--- Learnt Values  (Player 0)---")
        for h in range(1, self.game.H + 1):
            print(f"\n--- Stage h={h} ---")
            for s_str, s_idx in self.game.s_map[h].items():
                print(f"  State '{s_str}':")
                print(f"    V-value: {self.V[0, h, s_idx]:.4f}")
                print("    Q-values:")
                q_matrix = self.Q[0, h, s_idx, :, :]
                print("         a2=0    a2=1")
                print(f"    a1=0 [{q_matrix[0,0]:.2f}]  [{q_matrix[0,1]:.2f}]")
                print(f"    a1=1 [{q_matrix[1,0]:.2f}]  [{q_matrix[1,1]:.2f}]")
                print(f"    Learnt joint action (Policy): {[int(x) for x in self.a[h][s_idx]]}")

                
    def plot_convergence(self, save=None, save_path=None, no_override=None):
        """Plot convergence of V(s1)."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.V_history)
        plt.xlabel("Iteration (t)")
        plt.ylabel("V(s1)")
        plt.title("Convergence of the V-value in the initial state 's1'")
        plt.grid(True)

        save, save_path, no_override = self._plot_options(save, save_path, no_override)
        generate_plot(
            save=save,
            save_path=save_path,
            default_name="value_convergence.png",
            no_override=no_override
        )


    def plot_policy_evolution(self, history, params, save=None, save_path=None, no_override=None):
        """
        Plot policy evolution for actions (0,0) and (1,1) from stored runs_history.
        """
        runs = np.array(self._normalize_runs(history))
        num_runs = len(runs)
        T = self.T

        freq_00 = np.zeros((num_runs, T))
        freq_11 = np.zeros((num_runs, T))

        for i in range(num_runs):
            count_00 = 0
            count_11 = 0
            for t in range(T):
                action = tuple(runs[i, t])
                if action == (0, 0):
                    count_00 += 1
                elif action == (1, 1):
                    count_11 += 1

                freq_00[i, t] = count_00 / (t + 1)
                freq_11[i, t] = count_11 / (t + 1)

        mean_freq_00 = np.mean(freq_00, axis=0)
        mean_freq_11 = np.mean(freq_11, axis=0)

        lower_bound_00 = np.percentile(freq_00, 20, axis=0)
        upper_bound_00 = np.percentile(freq_00, 80, axis=0)
        lower_bound_11 = np.percentile(freq_11, 20, axis=0)
        upper_bound_11 = np.percentile(freq_11, 80, axis=0)

        plt.figure(figsize=(12, 7))

        plt.plot(mean_freq_00, label='Prob(a=(0,0) | s1)')
        plt.plot(mean_freq_11, label='Prob(a=(1,1) | s1)')
        if num_runs > 1:
            plt.fill_between(range(T), lower_bound_00, upper_bound_00, alpha=0.2)
            plt.fill_between(range(T), lower_bound_11, upper_bound_11, alpha=0.2)

        if len(params) == 1:
            eps, = params
            plt.title(f"Evolution of policy in state s1, eps={eps}")
        else:
            eps, c = params
            plt.title(f"Evolution of policy in state s1, eps={eps}, c={c}")

        plt.xlabel("Iteration (t)")
        plt.ylabel("Empirical probability of actions")
        plt.ylim(0, 1)
        plt.xlim(0, T)
        plt.legend()
        plt.grid(True)

        save, save_path, no_override = self._plot_options(save, save_path, no_override)
        generate_plot(
            save=save,
            save_path=save_path,
            default_name="policy_evolution.png",
            no_override=no_override
        )


    def _normalize_runs(self, history):
        """
        Accepts:
            - single trajectory: List[List]
            - list of trajectories: List[List[List]]
        Returns:
            List[List[List]]
        """
        if len(history) == 0:
            raise ValueError("Empty history.")

        if isinstance(history[0][0], (numbers.Number,str)):
            return [history]
        return history
    
    def _plot_options(self, save, save_path, no_override):
        if save is None:
            save = self._save
        if save_path is None:
            save_path = self._save_path
        if no_override is None:
            no_override = self._no_override
        return save, save_path, no_override
