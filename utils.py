import sys

import numpy as np
from matplotlib import pyplot as plt
from pyprind import ProgBar

from celural_automata import CellularAutomata
from celural_automata_evolution import CellularAutomataEvolution


def figure3(rules: list[int], size: int = 50, steps: int = 50, seed: int = 142):
    init_type = "random"  # "single" for one black cell in the 1st row, "random" for a random pattern
    swap_columns = True  # Set to True to swap columns
    swap_pairs = 0  # Number of columns pairs to swap
    slow_walking = False  # Enable directed motion of agents

    # dict to store plots
    square_plots = {}

    cellular_automata = CellularAutomata(size=size, seed=seed)

    for rule_number in rules:
        # agents_history - returns reputation history for every agent
        # grid_history - returns history of agents occupying grid cells,
        # cells can be occupied by different agents in different iteration when swaping is enabled

        agents_history, grid_history, _, _ = cellular_automata.generate_automaton(rule_number, steps, init_type,
                                                                                  swap_columns,
                                                                                  swap_pairs, slow_walking)

        # store plot for given rule
        square_plots[rule_number] = grid_history

    # Plot patterns on a grid (3 row) x (4 cols)
    keys = list(square_plots.keys())  # make a list of rules to plot

    fig, axes = plt.subplots(3, 4, figsize=(10, 8), gridspec_kw={'hspace': 0.2, 'wspace': -0.05})
    for i, ax in enumerate(axes.flat):
        ax.imshow(square_plots[keys[i]], cmap="binary")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title('Rule ' + str(keys[i]))
    plt.show()


def results_for_rules(rules: list[int], size: int, steps: int, repeats: int) -> dict:
    # Parameters
    seed = 142

    init_type = "random"
    swap_columns = True  # Set to True to swap columns
    slow_walking = False

    results_for_rule = {}

    cellular_automata = CellularAutomata(size=size, seed=seed)
    max_number_of_pairs_to_swap = [i for i in range(1, size, 1)]

    bar = ProgBar(len(rules) * repeats, stream=sys.stdout,
                  title="Results for rules", bar_char='█')

    for rule_number in rules:
        median_hist = []

        for repeat in range(0, repeats):
            for number_of_pairs_to_swap in max_number_of_pairs_to_swap:
                history, _, _, _ = cellular_automata.generate_automaton(rule_number, steps, init_type,
                                                                        swap_columns, number_of_pairs_to_swap,
                                                                        slow_walking)

                median_hist.append(np.median(np.sum(history, axis=0)))

            if not rule_number in results_for_rule.keys():
                results_for_rule[rule_number] = [median_hist]
            else:
                results_for_rule[rule_number].append(median_hist)
            bar.update()

    return results_for_rule


def figure4(rules: list[int], steps=300, size=100, repeats=5):
    results_for_rule = results_for_rules(rules=rules, steps=steps, size=size, repeats=repeats)
    plt.close()
    # y and x coordinates of labels to easily identify curves on a figure
    # this data create text labels for 300 steps
    labels_cords = {219: [0, 302], 251: [10, 302], 243: [0, 250], 187: [8, 250], 90: [76, 155], 153: [83, 155],
                    195: [91, 155], 50: [93, 122], 18: [0, 88], 48: [2, 30], 34: [7, 24], 72: [-2, 5]}

    number_of_pairs_to_swap = [i for i in range(1, size, 1)]

    fig, ax = plt.subplots()

    for ii, RULE_NUMBER in enumerate(rules):
        tmp = []
        for j in range(0, len(number_of_pairs_to_swap)):
            suma = 0
            for i in range(0, repeats):
                suma += results_for_rule[RULE_NUMBER][i][j]
            tmp.append(suma / repeats)

        ax.set_ylabel('averaged median reputation')
        ax.set_xlabel('Number of swapped pairs')
        ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
        ax.plot(tmp, label=RULE_NUMBER)
        if steps == 300:
            ax.text(labels_cords[RULE_NUMBER][0], labels_cords[RULE_NUMBER][1], f'{RULE_NUMBER}', clip_on=True)

    plt.legend(title='Rule:', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def figure56(rules: list[int], slower_walking, size: int, steps: int, seed=142):
    # Parameters
    init_type = "random"  # "single" for one black cell, "random" for a random pattern
    swap_columns = True  # Set to True to swap columns
    swap_percentage_list = [0, 1, 5, 10, 20, 50, 100]

    cellular_automata = CellularAutomata(size=size, seed=seed)

    square_plots_history = {}
    square_plots_history_2 = {}

    for i, rule_number in enumerate(rules):
        for j, swap_percentage in enumerate(swap_percentage_list):
            history, history_2, _, _ = cellular_automata.generate_automaton(rule_number, steps, init_type,
                                                                            swap_columns,
                                                                            swap_percentage, slower_walking)

            if j == 0:
                square_plots_history[rule_number] = [history.copy()]
                square_plots_history_2[rule_number] = [history_2.copy()]
            else:
                square_plots_history[rule_number].append(history.copy())
                square_plots_history_2[rule_number].append(history_2.copy())

    cols = len(rules)

    swap_percentage_list_to_plot = [1, 5, 10, 20, 50, 100]

    total_rows = len(swap_percentage_list_to_plot)
    cell_size = 4

    fig, axes = plt.subplots(6, 4, figsize=(total_rows * cell_size, cols * cell_size),
                             gridspec_kw={'hspace': 0.15, 'wspace': -0.85})

    for j in range(len(swap_percentage_list_to_plot)):
        for i, rule_number in enumerate(rules):
            row = j
            col = i

            ax = axes[row, col]

            data = square_plots_history_2[rule_number][j][-100:, :]
            ax.imshow(data, cmap="binary", aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            ax.set_title(f'Rule {rule_number}, Swap = {swap_percentage_list_to_plot[j]}', fontsize=10, pad=7)

    fig.subplots_adjust(left=0.06, right=0.94, top=0.96, bottom=0.04)
    plt.show()
    plt.close(fig)


def plot_automaton(history: np.ndarray, rule_number: int):
    plt.figure(figsize=(10, 10))
    plt.title(f"Rule {rule_number}")
    plt.imshow(history, cmap="binary", interpolation="nearest")
    plt.axis("on")  # Enable axis to create a frame
    plt.show()


def figure7a(rule_number: int, size: int, steps: int, seed: int = 142):
    init_type = "random"  # "single" for one black cell, "random" for a random pattern
    swap_columns = False  # Set to True to swap columns
    swap_percentage = 0  # Percentage of columns to swap
    slow_walking = False
    perception_noise = True
    rand_threshold = 0.10

    # Generate and plot
    cellular_automata = CellularAutomata(size=size, seed=seed)
    history, history_2, _, _ = cellular_automata.generate_automaton(rule_number, steps, init_type, swap_columns,
                                                                    swap_percentage,
                                                                    slow_walking,
                                                                    perception_noise, rand_threshold)
    plot_automaton(history_2, rule_number=rule_number)


def figure7(rules: list[int], size: int, steps: int, seed: int = 142, repeats=10):
    # Parameters
    init_type = "random"  # "single" for one black cell, "random" for a random pattern
    swap_columns = True  # Set to True to swap columns
    swap_percentage = 1

    slow_walking = False
    perception_noise = True

    cellular_automata = CellularAutomata(size=size, seed=seed)

    results_for_rule = {}

    percent = [i / 100 for i in range(0, 100, 1)]

    bar = ProgBar(len(rules) * len(percent), stream=sys.stdout,
                  title="Results for rules", bar_char='█')

    for rule_number in rules:
        median_hist = []
        for perception_percentage in percent:
            for repeat in range(0, repeats):
                # Generate and plot
                history, _, _, _ = cellular_automata.generate_automaton(rule_number, steps, init_type, swap_columns,
                                                                        swap_percentage,
                                                                        slow_walking,
                                                                        perception_noise,
                                                                        perception_noise_rand_threshold=perception_percentage)

                median_hist.append(np.median(np.sum(history, axis=0)))

            if not rule_number in results_for_rule.keys():
                results_for_rule[rule_number] = {}

            if not perception_percentage in results_for_rule[rule_number].keys():
                results_for_rule[rule_number][perception_percentage] = []

            results_for_rule[rule_number][perception_percentage] = sum(median_hist) / len(median_hist)
            bar.update()

    data_4_plot = {}
    for rule in results_for_rule.keys():
        data_4_plot[rule] = []
        for percentage in results_for_rule[rule]:
            data_4_plot[rule].append(results_for_rule[rule][percentage])

    for rule in results_for_rule.keys():
        plt.plot(data_4_plot[rule][:], label=rule)

    plt.legend()
    plt.title('swap = 10 + walking')
    plt.legend(title='Rule:', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def figure8(rules: list[int], size: int, steps: int, seed: int = 142, repeats=10):
    init_type = "random"  # "single" for one black cell, "random" for a random pattern
    swap_columns = True  # Set to True to swap columns
    swap_percentage = 2

    slow_walking = True
    perception_noise = True

    perception_noise_rand_threshold = 0.05

    action_noise = True

    results_for_rule = {}

    cellular_automata = CellularAutomata(size=size, seed=seed)

    percent = [i / 100 for i in range(0, 100, 1)]

    bar = ProgBar(len(rules) * len(percent), stream=sys.stdout,
                  title="Results for rules", bar_char='█')

    for rule_number in rules:
        median_hist = []

        for action_noise_rand_threshold in percent:
            for _ in range(repeats):
                history, history_2, history_3, donation_sum = cellular_automata.generate_automaton(rule_number, steps,
                                                                                                   init_type,
                                                                                                   swap_columns,
                                                                                                   swap_percentage,
                                                                                                   slow_walking,
                                                                                                   perception_noise,
                                                                                                   perception_noise_rand_threshold=perception_noise_rand_threshold,
                                                                                                   action_noise=action_noise,
                                                                                                   action_noise_rand_threshold=action_noise_rand_threshold)

                median_hist.append(np.median(np.sum(history_3, axis=0)))

            if not rule_number in results_for_rule.keys():
                results_for_rule[rule_number] = {}

            if not action_noise_rand_threshold in results_for_rule[rule_number].keys():
                results_for_rule[rule_number][action_noise_rand_threshold] = []

            results_for_rule[rule_number][action_noise_rand_threshold] = sum(median_hist) / len(median_hist)
            bar.update()

    data_4_plot = {}
    for rule in results_for_rule.keys():
        data_4_plot[rule] = []
        for percentage in results_for_rule[rule]:
            data_4_plot[rule].append(results_for_rule[rule][percentage])

    for rule in results_for_rule.keys():
        plt.plot(data_4_plot[rule][:], label=rule)

    plt.legend()

    if swap_columns:
        plt.title(
            f'ACTION NOISE /  - Swap {swap_columns} = {swap_columns} + WALKING {int(slow_walking)}  + PERCEPTION NOISE {int(perception_noise)}',
            size=9)
    else:
        plt.title(
            f'ACTION NOISE /  - No Swap + WALKING {int(slow_walking)}  + PERCEPTION NOISE {int(perception_noise)}',
            size=9)

    plt.ylabel('Averaged  median donations')
    plt.xlabel('Action noise')
    plt.legend(title='Rule:', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def create_mutation_matrix(rules: list[int]):
    rules.sort()

    # Calculate bit differences
    def bit_difference(a, b):
        return bin(a ^ b).count("1")

    # Create a matrix of bit differences
    size = len(rules)
    diff_matrix = np.zeros((size, size), dtype=float)

    for i in range(size):
        for j in range(size):
            diff_matrix[i, j] = bit_difference(rules[i], rules[j])
    # Normalize to get probability-like values (0 to 1)
    probability_matrix = (8 - diff_matrix) / 8.0

    # Optional: convert to mutation probability (invert if needed)
    # e.g., smaller bit difference = higher probability
    # probability_matrix = diff_matrix #/ 8.0
    probability_matrix = probability_matrix * (probability_matrix < 1)

    mutation_matrix = probability_matrix / probability_matrix.sum(axis=1, keepdims=True)
    return mutation_matrix


def figure9a(rules: list[int], history_number_of_agents_vs_evolution_step: dict, ev_generations):
    max_points = ev_generations - 1
    plt.figure()

    for key in rules:
        plt.title('Number of agents following a rule')
        plt.ylabel('Agents counts')
        plt.xlabel('Generations')
        plt.plot(history_number_of_agents_vs_evolution_step[key][-max_points:], label=key)
    plt.legend()
    plt.show()


def figure9b(rules: list[int], history_number_of_agents_vs_evolution_step: dict):
    plt.figure()
    from_iter = 0
    to_iter = 50
    for key in rules:
        plt.title('Number of agents following a rule')
        plt.ylabel('Number of agents')
        plt.xlabel('Generation')
        plt.plot([i for i in range(from_iter, to_iter)],
                 history_number_of_agents_vs_evolution_step[key][from_iter:to_iter], label=key)
        plt.legend()
    plt.show()


def run_evolution(rules: list[int], size=100, steps=300, ev_generations=5000, seed=142):
    # Parameters
    # All rules used to simulate donation game, calculate mutation matrix
    mutation_matrix = create_mutation_matrix(rules)

    proportions = [0.08, 0.09, 0.08, 0.09, 0.08, 0.08, 0.09, 0.08, 0.08, 0.08, 0.08, 0.09]
    rules_agent = rules

    rules_as_agents_list = CellularAutomataEvolution.generate_integer_list(values=rules_agent, proportions=proportions,
                                                                           total_length=size)
    init_type = "random"  # "single" for one black cell, "random" for a random pattern
    swap_columns = True  # False # Set to True to swap columns
    swap_percentage = 2  # Percentage of columns to swap
    slow_walking = False
    perception_noise = False

    perception_noise_rand_threshold = 0.05

    action_noise = False
    action_noise_rand_threshold = 0.05

    # Fatigue parameter sets agent DOESN'T TO DONATE EVERY n ITERATIONS WHEN AGENT DONATED IN LAST
    # SET in var fatigue_iteration

    fatigue = False
    fatigue_iteration = 2
    evolution = True

    # after a selected number of STEPS in const STEPS
    # the every agent can mutate with chance below
    # mutation is governed by probability matrix
    evolution_chance = 0.001

    history_donations_vs_rule_vs_evolution_step = []
    history_number_of_agents_vs_evolution_step = {}

    for rule in rules:
        history_number_of_agents_vs_evolution_step[rule] = []

    cellular_automata_evolution = CellularAutomataEvolution(size=size, seed=seed)

    bar = ProgBar(ev_generations, stream=sys.stdout,
                  title="Evolution", bar_char='█')

    for evolution_step in range(ev_generations):

        #####print(rules_as_agents_list)
        # additional parameters are returned here, last one shows position of agents following a particular rule
        agent_history, grid_history, donation_sum, history_donations_for_recipients, rules_as_agents_after_generation = cellular_automata_evolution.generate_automaton(
            steps, init_type,
            swap_columns, swap_percentage,
            slow_walking,
            perception_noise,
            perception_noise_rand_threshold,
            action_noise,
            action_noise_rand_threshold,
            rules_as_agents_list,
            fatigue,
            fatigue_iteration
        )

        # Count agents following a particular rule
        for rule in rules:
            history_number_of_agents_vs_evolution_step[rule].append(rules_as_agents_after_generation.count(rule))

        donations_vs_rule_dict = {}
        for rule in set(rules_as_agents_after_generation):
            donations_vs_rule_dict[rule] = 0

        donations = np.sum(history_donations_for_recipients, axis=0)

        for i, rule in enumerate(rules_as_agents_after_generation):
            donations_vs_rule_dict[rule] += float(donations[i])

        donations_sum = sum([donations_vs_rule_dict[rule] for rule in donations_vs_rule_dict.keys()])
        normalized_donations_vs_rule_dict = {}

        for key in donations_vs_rule_dict.keys():
            normalized_donations_vs_rule_dict[key] = donations_vs_rule_dict[key] / donations_sum

        history_donations_vs_rule_vs_evolution_step.append(normalized_donations_vs_rule_dict)
        rules2 = list(donations_vs_rule_dict.keys())

        proportions = [normalized_donations_vs_rule_dict[key] for key in normalized_donations_vs_rule_dict.keys()]
        rules_as_agents_list = CellularAutomataEvolution.generate_integer_list(rules2, proportions, size)

        # EVOLUTION
        for i, agent in enumerate(rules_as_agents_list):
            chance = np.random.random()
            if evolution and chance < evolution_chance:
                rule_idx = rules.index(agent)
                probabilities = mutation_matrix[rule_idx]
                new_rule = np.random.choice(rules, p=probabilities)
                rules_as_agents_list[i] = int(new_rule)

        bar.update()

    return history_number_of_agents_vs_evolution_step
