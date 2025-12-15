import numpy as np


class CellularAutomata:
    def __init__(self, size: int, seed: int | None = None):
        self.size = size
        self.seed = seed

        np.random.seed(self.seed)

    @staticmethod
    def get_rule_binary(rule_number: int) -> np.ndarray:
        return np.array([int(x) for x in f"{rule_number:08b}"])

    @staticmethod
    def apply_rule(rule: np.ndarray, left: int, center: int, right: int) -> int:
        index = 7 - (4 * left + 2 * center + right)
        return int(rule[index])

    @staticmethod
    def swap_odd_columns(matrix: np.ndarray) -> np.ndarray:
        rows, cols = matrix.shape
        if cols < 3:
            return matrix

        new_matrix = matrix.copy()

        odd_cols = new_matrix[:, 1::2]
        odd_cols_swaped = np.roll(odd_cols, shift=1, axis=1)

        new_matrix[:, 1::2] = odd_cols_swaped

        return new_matrix

    @staticmethod
    def swap_random_columns(arr: np.ndarray) -> np.ndarray:
        if arr.shape[1] < 2:
            return arr

        cols = np.random.choice(range(arr.shape[1]), 2,replace=False)
        arr[:, cols[0]], arr[:, cols[1]] = arr[:, cols[1]].copy(), arr[:, cols[0]].copy()

        return arr

    def init_history(self, steps: int, init_type: str) -> np.ndarray:
        if init_type == "single":
            history = np.zeros((steps, self.size), dtype=int)
            history[0, self.size // 2] = 1
        elif init_type == "random":
            history = np.random.choice([0, 1], size=(steps, self.size))
        else:
            raise ValueError("Invalid init_type")
        return history

    def generate_automaton(self, rule_number: int, steps=50, init_type="single",
                           swap_columns=False, swap_percentage=0.0, slow_walking=False, perception_noise=False,
                           perception_noise_rand_threshold=0.0, action_noise=False,
                           action_noise_rand_threshold=0.0,
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        rule = CellularAutomata.get_rule_binary(rule_number)
        history = self.init_history(steps, init_type)
        history_2 = history.copy()
        history_agent_without_errors = history.copy()

        donation_sum = 0
        for t in range(1, steps):
            new_state = np.zeros(self.size, dtype=int)
            for i in range(self.size):  # Periodic boundary conditions
                left = int(history[t - 1, (i - 1) % self.size])
                center = int(history[t - 1, i])
                right = int(history[t - 1, (i + 1) % self.size])

                if perception_noise:
                    if np.random.random() < perception_noise_rand_threshold:
                        left = int(not left)

                    if np.random.random() < perception_noise_rand_threshold:
                        right = int(not right)

                new_state[i] = CellularAutomata.apply_rule(rule, left, center, right)

                donation_sum = donation_sum + new_state[i]
                history_agent_without_errors[t] = new_state

                if action_noise and (new_state[i] == 1) and (np.random.random() < action_noise_rand_threshold):
                    new_state[i] = 0

            history[t] = new_state
            history_2[t] = new_state

            if swap_columns and swap_percentage > 0.0:
                num_swap = int(swap_percentage)

                for jj in range(num_swap):
                    history = CellularAutomata.swap_random_columns(history)
                    history_agent_without_errors = CellularAutomata.swap_random_columns(history_agent_without_errors)

            if slow_walking:
                history = CellularAutomata.swap_odd_columns(history)

        return history, history_2, history_agent_without_errors, donation_sum
