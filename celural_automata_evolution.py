import numpy as np

from celural_automata import CellularAutomata
from donations_data import calc_donation


class CellularAutomataEvolution(CellularAutomata):
    def __init__(self, size: int, seed: int | None = None):
        super().__init__(size, seed)

    @staticmethod
    def swap_odd_list_elements(list_of_elements):
        odd_values = [list_of_elements[i] for i in range(len(list_of_elements)) if i % 2 != 0]

        for i, item in enumerate(odd_values):
            if (2 * i + 3) < len(list_of_elements):
                list_of_elements[2 * i + 3] = item
            else:
                list_of_elements[1] = item
        return list_of_elements

    @staticmethod
    def generate_integer_list(values: list[int], proportions: list[float], total_length: int) -> list[int]:
        if len(values) != len(proportions):
            raise ValueError("Length of values and proportions must be equal.")

        if abs(sum(proportions) - 1.0) > 1e-6:
            raise ValueError("Proportions must sum to 1.0")

        counts = [int(p * total_length) for p in proportions]
        # Adjust in case rounding causes the total to be off
        while sum(counts) < total_length:
            counts[counts.index(min(counts))] += 1
        while sum(counts) > total_length:
            counts[counts.index(max(counts))] -= 1

        result = []
        for value, count in zip(values, counts):
            result.extend([value] * count)
        np.random.shuffle(result)
        return result

    def generate_automaton(self, steps=50, init_type="single",
                           swap_columns=False, swap_percentage=0.0, slow_walking=False, perception_noise=False,
                           perception_noise_rand_threshold=0.0, action_noise=False,
                           action_noise_rand_threshold=0.0,
                           rules_as_agents: list[int] = None,
                           fatigue=False,
                           f_parameter=0
                           ) -> (np.ndarray, int, np.ndarray, np.ndarray, np.ndarray):

        bin_rule_as_agents = [CellularAutomata.get_rule_binary(item) for item in rules_as_agents]
        donation_sum = 0

        history = super().init_history(steps, init_type)
        history_2 = history.copy()

        history_donations_for_recipients = np.zeros((steps, self.size), dtype=float)

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

                # APPLY RULE
                rule = bin_rule_as_agents[i]
                new_state[i] = CellularAutomata.apply_rule(rule, left, center, right)

                if fatigue and t > f_parameter and new_state[i] == 1 and np.sum(history[t - f_parameter:t, i],
                                                                                axis=0) == f_parameter:
                    new_state[i] = 0
                    left_donation = 0.0
                    right_donation = 0.0
                else:
                    agent_rule = rules_as_agents[i]
                    left_donation, right_donation = calc_donation(left, center, right, agent_rule)

                if i == 0:
                    history_donations_for_recipients[t, self.size - 1] = left_donation
                    history_donations_for_recipients[t, 1] = right_donation
                elif (i + 1) == self.size:
                    history_donations_for_recipients[t, self.size - 2] = left_donation
                    history_donations_for_recipients[t, 0] = right_donation
                else:
                    history_donations_for_recipients[t, i - 1] = left_donation
                    history_donations_for_recipients[t, i + 1] = right_donation

                donation_sum = donation_sum + new_state[i]

                if action_noise and (new_state[i] == 1) and (np.random.random() < action_noise_rand_threshold):
                    new_state[i] = 0

            history[t] = new_state
            history_2[t] = new_state

            if swap_columns and swap_percentage > 0.0:
                num_swap = int(swap_percentage)

                for jj in range(num_swap):
                    cols2swap = np.random.choice(range(history.shape[1]), 2, replace=False)

                    history_agent_1 = history[:, cols2swap[0]].copy()
                    history_agent_2 = history[:, cols2swap[1]].copy()

                    history[:, cols2swap[0]] = history_agent_2
                    history[:, cols2swap[1]] = history_agent_1

                    # donation history for recipients
                    history_recipient_1 = history_donations_for_recipients[:, cols2swap[0]].copy()
                    history_recipient_2 = history_donations_for_recipients[:, cols2swap[1]].copy()

                    history_donations_for_recipients[:, cols2swap[0]] = history_recipient_2
                    history_donations_for_recipients[:, cols2swap[1]] = history_recipient_1

                    agent_1 = bin_rule_as_agents[cols2swap[0]]
                    agent_2 = bin_rule_as_agents[cols2swap[1]]

                    bin_rule_as_agents[cols2swap[0]] = agent_2
                    bin_rule_as_agents[cols2swap[1]] = agent_1

                    agent_1_dec = rules_as_agents[cols2swap[0]]
                    agent_2_dec = rules_as_agents[cols2swap[1]]

                    rules_as_agents[cols2swap[0]] = agent_2_dec
                    rules_as_agents[cols2swap[1]] = agent_1_dec

            if slow_walking:
                history = CellularAutomata.swap_odd_columns(history)
                rules_as_agents = CellularAutomataEvolution.swap_odd_list_elements(rules_as_agents)

        return history, history_2, donation_sum, history_donations_for_recipients, rules_as_agents
