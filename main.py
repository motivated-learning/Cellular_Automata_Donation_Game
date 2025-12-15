import random

from utils import figure56, figure3, figure4, figure7a, figure7, figure8, run_evolution, figure9a, figure9b
from donations_data import donations

if __name__ == "__main__":
    # rules to plot; rule is any integer value between 0-255

    # rules = [251, 219, 243, 187, 195, 153, 90, 50, 48, 34, 18, 72]
    # figure3(rules=rules, size=50, steps=50)
    #
    # rules = [243, 153, 219, 195, 251, 187, 50, 48, 34, 90, 72, 18]
    # figure4(rules=rules, steps=20, size=100, repeats=2)
    #
    # rules = [243, 90, 50, 18]
    # figure56(rules=rules, slower_walking=False, size=100, steps=300)
    #
    # rules = [243, 90, 50, 18]
    # figure56(rules=rules, slower_walking=True, size=100, steps=300)

    # rules = [153, 219, 195, 251, 243, 187, 50, 48, 34, 90, 72, 18]
    # figure7(rules=rules, size=100, steps=20, repeats=2)

    # rules = [153, 219, 195, 251, 243, 187, 50, 48, 34, 90, 72, 18]
    # figure8(rules=rules, size=100, steps=20, repeats=2)

    # probability_matrix_1()

    rules = [153, 219, 195, 251, 243, 187, 50, 48, 34, 90, 72, 18]
    ev_generations = 500
    history_number_of_agents_vs_evolution_step = run_evolution(rules=rules, size=100, steps=300, ev_generations=ev_generations)
    figure9a(rules, history_number_of_agents_vs_evolution_step, ev_generations)
    figure9b(rules, history_number_of_agents_vs_evolution_step)
