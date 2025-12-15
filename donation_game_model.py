# -*- coding: utf-8 -*-
import time
import os
import sys
import argparse
import random
import numpy as np
import csv
from numba import njit
import multiprocessing as mp
from tqdm import tqdm

# Constants for donations
B = 1.0  # Benefit for recipient
C = 0.1  # Cost for donor

# --- MODEL 1 (Global Noisy) Function ---

@njit(cache=False)
def tick_noisy(n, m, q, strategies, imageScores, rewards, ea, ep):
    """
    One generation tick for noisy global model (Model 1).
    Random interactions, includes action (ea) and perception (ep) noise.
    """
    for _ in range(m): # m interactions per generation
        donor = np.random.randint(0, n)
        recipient = np.random.randint(0, n - 1)
        if recipient >= donor:
            recipient += 1 # Avoid self-interaction

        intend_cooperate = (imageScores[donor, recipient] >= strategies[donor])
        action_cooperate = intend_cooperate
        # Apply action noise (ea)
        if intend_cooperate and ea > 1e-9 and np.random.random() < ea:
            action_cooperate = False
        # Optional: noise causing defection -> cooperation
        # elif not intend_cooperate and ea > 1e-9 and np.random.random() < ea:
        #     action_cooperate = True

        if action_cooperate:
            rewards[donor] -= C
            rewards[recipient] += B
        # Defection or failed cooperation

        # Update reputations
        for i in range(n): # Global observation loop
            if i == donor: continue
            if i == recipient: # Recipient observes perfectly
                imageScores[i, donor] = min(imageScores[i, donor] + 1.0, 5.0) if action_cooperate else max(imageScores[i, donor] - 1.0, -5.0)
            elif q == 1.0 or np.random.random() < q: # Others observe with probability q
                perceived_action_is_cooperate = action_cooperate
                # Apply perception noise (ep)
                if ep > 1e-9 and np.random.random() < ep:
                    perceived_action_is_cooperate = not action_cooperate
                # Update observer's score
                if perceived_action_is_cooperate:
                    imageScores[i, donor] = min(imageScores[i, donor] + 1.0, 5.0)
                else:
                    imageScores[i, donor] = max(imageScores[i, donor] - 1.0, -5.0)

    return strategies, imageScores, rewards

# --- Main simulation loop for Model 1 ---
@njit(cache=False)
def run_simulation_global_noisy(n, m, q, mr, ea, ep, generations):
    """Main loop for Global Noisy Model (Model 1)."""
    strategies = np.empty(n, dtype=np.int32)
    for i in range(n): strategies[i] = np.random.randint(-5, 7) # k in [-5, 6]
    imageScores = np.zeros((n, n), dtype=np.float64)
    rewards = np.zeros(n, dtype=np.float64)
    all_reward_averages = np.zeros(generations, dtype=np.float64)

    for g in range(generations):
        rewards[:] = 0.0

        # --- Interaction Phase ---
        strategies, imageScores, rewards = tick_noisy(n, m, q, strategies, imageScores, rewards, ea, ep)

        # --- Record Average Reward ---
        if n > 0:
            current_rewards = rewards[~(np.isnan(rewards) | np.isinf(rewards))]
            gen_avg_reward = np.mean(current_rewards) if current_rewards.size > 0 else np.nan
            all_reward_averages[g] = gen_avg_reward
        else:
            all_reward_averages[g] = 0.0 # Handle n=0

        # --- Selection Phase ---
        # Handle NaN/Inf rewards
        if np.any(np.isnan(rewards)) or np.any(np.isinf(rewards)):
              # Debug print
              # print(f"Warning: NaN or Inf detected in rewards at generation {g+1}. Resetting image scores.", file=sys.stderr)
              imageScores[:] = 0.0  # Reset scores on invalid reward
              # Skip selection this gen
              continue # Skip selection and mutation for this generation

        # Scale rewards >= 0 for selection
        scaled_rewards = rewards.copy()
        min_reward = scaled_rewards.min()
        if min_reward < 0: scaled_rewards -= min_reward # Shift all rewards so min is 0
        # Add constant if all rewards ~0
        if n > 0 and scaled_rewards.sum() <= 1e-9:
            scaled_rewards += 0.1

        total_scaled_reward = scaled_rewards.sum()
        new_strategies = np.empty_like(strategies)

        # Select parents by scaled rewards
        if n > 0 and total_scaled_reward > 1e-9:
            parent_indices = np.empty(n, dtype=np.int64)
            for i in range(n):
                parent_idx = _java_weighted_random_choice(scaled_rewards)
                # Fallback selection
                parent_indices[i] = parent_idx if parent_idx != -1 else np.random.randint(0, n)
            new_strategies = strategies[parent_indices]
        else:
            # Handle no rewards / n=0
            new_strategies = strategies.copy() # Keep current strategies if selection cannot proceed

        strategies = new_strategies

        # --- Mutation Phase ---
        for i in range(n):
            if np.random.random() < mr:
                current_strategy = strategies[i]
                # Ensure mutation changes strategy
                mutated_strategy = np.random.randint(-5, 7)
                while mutated_strategy == current_strategy:
                    mutated_strategy = np.random.randint(-5, 7)
                strategies[i] = mutated_strategy

        # --- Reset Image Scores for next generation ---
        # Reset scores post-selection/mutation
        imageScores[:] = 0.0

    # --- Calculate final average reward ---
    if generations > 0:
        valid_rewards_mask = ~np.isnan(all_reward_averages)
        if np.any(valid_rewards_mask):
           # Return avg of valid generation avgs
           return np.mean(all_reward_averages[valid_rewards_mask])
        else:
           # Debug print
           # print("Warning: All generation rewards were NaN.", file=sys.stderr)
           return np.nan # Return NaN on failure
    else:
        return 0.0 # No generations run


# --- MODEL 2 (Local Grid) Functions ---

@njit(cache=False)
def tick_grid_chunk(n, interactions_in_chunk, q, strategies, imageScores, ea, ep):
    """
    Interaction chunk for local grid model (Model 2).
    Local interactions (neighbors), global observation.
    """
    chunk_rewards = np.zeros(n, dtype=np.float64)
    for _ in range(interactions_in_chunk):
        # Select donor
        donor_idx = np.random.randint(0, n)

        # Select neighbor recipient (1D ring)
        direction = np.random.randint(0, 2) # 0 for left, 1 for right
        recipient_idx = (donor_idx - 1 + n) % n if direction == 0 else (donor_idx + 1) % n

        # Donor uses own score of recipient
        imageScore_donor_view = imageScores[donor_idx, recipient_idx]
        intend_cooperate = (imageScore_donor_view >= strategies[donor_idx]) # Donor uses their strategy k

        action_cooperate = intend_cooperate
        # Apply action noise (ea)
        if intend_cooperate and ea > 1e-9 and np.random.random() < ea:
            action_cooperate = False
        # Optional: noise causing defection -> cooperation
        # elif not intend_cooperate and ea > 1e-9 and np.random.random() < ea:
        #     action_cooperate = True

        # Apply payoffs
        if action_cooperate:
            chunk_rewards[donor_idx] -= C
            chunk_rewards[recipient_idx] += B

        # Global observation of donor
        for observer_idx in range(n):
            if observer_idx == donor_idx: continue # Exclude self-observation

            # Recipient observes perfectly
            if observer_idx == recipient_idx:
                if action_cooperate:
                    imageScores[recipient_idx, donor_idx] = min(imageScores[recipient_idx, donor_idx] + 1.0, 5.0)
                else:
                    imageScores[recipient_idx, donor_idx] = max(imageScores[recipient_idx, donor_idx] - 1.0, -5.0)
            # Others observe globally (prob q, noise ep)
            else:
                if q == 1.0 or np.random.random() < q:
                    perceived_action_is_cooperate = action_cooperate
                    # Apply perception noise (ep)
                    if ep > 1e-9 and np.random.random() < ep:
                        perceived_action_is_cooperate = not action_cooperate

                    # Update observer's score of donor
                    if perceived_action_is_cooperate:
                        imageScores[observer_idx, donor_idx] = min(imageScores[observer_idx, donor_idx] + 1.0, 5.0)
                    else:
                        imageScores[observer_idx, donor_idx] = max(imageScores[observer_idx, donor_idx] - 1.0, -5.0)

    # Return updated strategies, scores, chunk rewards
    return strategies, imageScores, chunk_rewards


@njit(cache=False)
def swap_agents_chunk(n, strategies, imageScores, total_rewards_gen, swap_prob):
    """
    Grid swap: Swaps agents (strategy, reward, image scores) with swap_prob.
    Simulates mobility (Model 2).
    """
    if swap_prob <= 1e-9: return strategies, imageScores, total_rewards_gen # No swapping needed
    if swap_prob >= 1.0: # Full shuffle if swap_prob >= 1
        p = np.random.permutation(n)
        return strategies[p], imageScores[p][:, p], total_rewards_gen[p]

    # Pairwise swap
    indices_to_swap = np.where(np.random.random(n) < swap_prob)[0]
    num_to_swap = len(indices_to_swap)

    if num_to_swap > 1:
        np.random.shuffle(indices_to_swap) # Shuffle for random pairs
        # Iterate through pairs
        for i in range(0, num_to_swap - 1, 2): # Step by 2 for pairs
            idx1, idx2 = indices_to_swap[i], indices_to_swap[i+1]

            # Swap strategies
            strategies[idx1], strategies[idx2] = strategies[idx2], strategies[idx1]
            # Swap rewards
            total_rewards_gen[idx1], total_rewards_gen[idx2] = total_rewards_gen[idx2], total_rewards_gen[idx1]

            # Swap image scores (rows/cols)
            # Swap rows (agent views)
            row1_copy = imageScores[idx1, :].copy()
            imageScores[idx1, :] = imageScores[idx2, :]
            imageScores[idx2, :] = row1_copy
            # Swap cols (views of agent)
            col1_copy = imageScores[:, idx1].copy()
            imageScores[:, idx1] = imageScores[:, idx2]
            imageScores[:, idx2] = col1_copy

    return strategies, imageScores, total_rewards_gen


@njit(cache=False)
def run_simulation_local_grid(n, m, q, mr, ea, ep, generations, swap_prob, chunk_size):
    """Main loop for Local Grid Model (Model 2)."""
    strategies = np.empty(n, dtype=np.int32)
    for i in range(n): strategies[i] = np.random.randint(-5, 7) # k in [-5, 6]
    imageScores = np.zeros((n, n), dtype=np.float64)
    all_reward_averages = np.zeros(generations, dtype=np.float64)

    if chunk_size <= 0: chunk_size = 1 # Ensure chunk size is at least 1
    # Handle chunk division
    # May adjust total m
    # Assume m large enough
    # Use integer division for chunks
    interactions_per_chunk = chunk_size
    num_chunks = max(1, m // chunk_size) # Number of chunks per generation

    for g in range(generations):
        total_rewards_gen = np.zeros(n, dtype=np.float64) # Accumulate rewards for the generation

        # --- Chunk Loop (Interact + Swap) ---
        for chunk_idx in range(num_chunks):
            # Interact within chunk
            strategies, imageScores, chunk_rewards = tick_grid_chunk(
                n, interactions_per_chunk, q, strategies, imageScores, ea, ep
            )
            total_rewards_gen += chunk_rewards # Accumulate rewards

            # Swap agents post-interaction
            strategies, imageScores, total_rewards_gen = swap_agents_chunk(
                n, strategies, imageScores, total_rewards_gen, swap_prob
            )
        # --- End Chunk Loop ---

        # Record gen avg reward
        if n > 0:
            current_rewards = total_rewards_gen[~(np.isnan(total_rewards_gen) | np.isinf(total_rewards_gen))]
            gen_avg_reward = np.mean(current_rewards) if current_rewards.size > 0 else np.nan
            all_reward_averages[g] = gen_avg_reward
        else:
            all_reward_averages[g] = 0.0 # Handle n=0

        # --- Selection (Global) ---
        if np.any(np.isnan(total_rewards_gen)) or np.any(np.isinf(total_rewards_gen)):
            # Debug print
            # print(f"Warning: NaN or Inf detected in rewards at generation {g+1} (Local Grid). Resetting image scores.", file=sys.stderr)
            imageScores[:] = 0.0
            continue # Skip selection/mutation

        scaled_rewards = total_rewards_gen.copy()
        min_reward = scaled_rewards.min()
        if min_reward < 0: scaled_rewards -= min_reward # Scale rewards >= 0 for selection
        if n > 0 and scaled_rewards.sum() <= 1e-9: scaled_rewards += 0.1 # Add constant if all rewards ~0

        total_scaled_reward = scaled_rewards.sum()
        new_strategies = np.empty_like(strategies)

        if n > 0 and total_scaled_reward > 1e-9: # Select parents by scaled rewards
            parent_indices = np.empty(n, dtype=np.int64)
            for i in range(n):
                parent_idx = _java_weighted_random_choice(scaled_rewards)
                parent_indices[i] = parent_idx if parent_idx != -1 else np.random.randint(0, n) # Fallback selection
            new_strategies = strategies[parent_indices]
        else:
             new_strategies = strategies.copy() # Handle no rewards / n=0

        strategies = new_strategies

        # --- Mutation Phase ---
        for i in range(n):
            if np.random.random() < mr:
                current_strategy = strategies[i]
                mutated_strategy = np.random.randint(-5, 7)
                while mutated_strategy == current_strategy: # Ensure mutation changes strategy
                    mutated_strategy = np.random.randint(-5, 7)
                strategies[i] = mutated_strategy

        # --- Reset Image Scores for next generation ---
        imageScores[:] = 0.0

    # --- Calculate Final Average Reward ---
    if generations > 0:
        valid_rewards_mask = ~np.isnan(all_reward_averages)
        if np.any(valid_rewards_mask):
           return np.mean(all_reward_averages[valid_rewards_mask]) # Return avg of valid generation avgs
        else:
           # Debug print
           # print("Warning: All generation rewards were NaN (Local Grid).", file=sys.stderr)
           return np.nan # Return NaN on failure
    else:
        return 0.0 # No generations run


# --- SHARED HELPER Functions ---

@njit(cache=False)
def _java_weighted_random_choice(rewards_scaled):
    """
    Weighted random choice (mimics Java logic).
    Assumes non-negative weights. Returns -1 on failure.
    """
    n_local = len(rewards_scaled)
    if n_local == 0: return -1

    # Find first agent with reward > 0
    start_idx = -1
    for i in range(n_local):
        if rewards_scaled[i] > 1e-9: # Use float tolerance
            start_idx = i
            break

    # Handle no positive rewards
    if start_idx == -1:
        # Fallback: uniform random choice
        return np.random.randint(0, n_local) if n_local > 0 else -1

    # Initialize selection process
    selected_idx = start_idx
    running_total = rewards_scaled[start_idx]

    # Iterate remaining agents
    for i in range(start_idx + 1, n_local):
        current_reward = rewards_scaled[i]
        if current_reward > 1e-9: # Only consider reward > 0
            running_total += current_reward
            # Prob. of switching to agent i
            # is current_reward / running_total_up_to_i
            if running_total > 1e-9: # Avoid division by zero
                # Core: proportional selection
                if np.random.random() <= (current_reward / running_total):
                    selected_idx = i
            # If running_total ~0, selection stays

    return selected_idx

# --- MULTIPROCESSING Worker ---

def simulation_worker(params):
    """Worker: calls simulation func for model 1 or 2."""
    # Unpack params (incl. seed)
    # Expected: (model, q, ea, ep, generations, n, m, mr, swap_prob, chunk_s, sim_seed)
    try:
        # Check param count
        if len(params) != 11:
            raise ValueError(f"Incorrect number of parameters received by worker: {len(params)}. Expected 11.")

        (model, q, ea, ep, generations, n, m, mr, swap_prob, chunk_s, sim_seed) = params

    except ValueError as e:
        print(f"Error unpacking parameters in worker: {e}. Params tuple: {params}", file=sys.stderr)
        # Return matching structure
        # Problematic to return NaN key
        # Return invalid struct/re-raise?
        # Return NaN key or error indicator
        # Key is (model, q, ea, ep, swap_prob, chunk_s)
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan) # Return structure indicating error


    try:
        # Use task-specific seed
        np.random.seed(sim_seed)
        # Optional: seed std random
        # random.seed(sim_seed)
    except ValueError as e:
        print(f"Error setting seed for process: seed={sim_seed}, error={e}", file=sys.stderr)
        # Return error structure
        return (model, q, ea, ep, swap_prob, chunk_s, np.nan) # Match aggregation key + NaN result

    avg_reward = np.nan # Default result in case of error
    try:
        if model == 1:
            # Call Global model sim
            avg_reward = run_simulation_global_noisy(n, m, q, mr, ea, ep, generations)
        elif model == 2: # Model 2 = Local Grid
            # Call Local Grid model sim
            avg_reward = run_simulation_local_grid(n, m, q, mr, ea, ep, generations, swap_prob, chunk_s)
        else:
            print(f"Error: Unsupported model number {model} in worker.", file=sys.stderr)
            # avg_reward remains np.nan

    except Exception as e:
        # Catch sim errors
        # Optional: full traceback: import traceback; traceback.print_exc()
        print(f"\nError during simulation run for model {model}, seed {sim_seed}. "
              f"Params: q={q}, ea={ea}, ep={ep}, swap={swap_prob}, chunk={chunk_s}. Error: {e}", file=sys.stderr)
        avg_reward = np.nan # Ensure NaN is returned on error

    # Return key + result
    # Key: (model, q, ea, ep, swap_prob, chunk_size)
    # Result: avg_reward
    return (model, q, ea, ep, swap_prob, chunk_s, avg_reward)

# --- MAIN SCRIPT ---

def main():
    parser = argparse.ArgumentParser(description="Donation Game Simulation: Global (1), Local Grid (2), or Comparison.")

    # --- Model Selection ---
    parser.add_argument("--model", type=str, required=True, choices=['1', '2', 'both'],
                        help="Simulation model: 1=Global, 2=Local Grid, both=Run comparison of 1 and 2")

    # --- General Parameters ---
    parser.add_argument("--size", type=int, default=100, help="Number of agents (n)")
    parser.add_argument("--interactions", type=int, default=300, help="Total interactions per generation (m)")
    parser.add_argument("--generations", type=int, default=100000, help="Number of generations")
    parser.add_argument("--runs", type=int, default=100, help="Number of independent runs for averaging")
    parser.add_argument("--mutation", type=float, default=0.001, help="Mutation rate (mr)")
    # Default output is results.csv
    parser.add_argument("--output", type=str, default="results.csv", help="Output CSV file name (defaults to results.csv)")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1), help="Number of worker processes")
    parser.add_argument("--seed", type=int, default=42, help="Seed for master random number generator for reproducibility")


    # --- Noise & Observation Parameters ---
    parser.add_argument("--q_values", nargs="+", type=float, default=[1.0], help="Observation probability (q)")
    parser.add_argument("--ea_values", nargs="+", type=float, required=True, help="List of action noise levels (ea)")
    parser.add_argument("--ep_values", nargs="+", type=float, required=True, help="List of perception noise levels (ep)")

    # --- Grid Params (Model 2/both) ---
    parser.add_argument("--swap_probs", nargs="+", type=float, default=[0.0],
                        help="Swap probabilities (Model 2/both)")
    parser.add_argument("--chunk_size", type=int, default=10,
                        help="Interactions per chunk (Model 2/both)")

    args = parser.parse_args()

    # --- Set master seed for reproducibility ---
    print(f"Setting global random seed to: {args.seed}")
    np.random.seed(args.seed)
    # Master RNG for task seeds
    master_rng = np.random.RandomState(args.seed)

    # --- Input Validation ---
    if len(args.ea_values) != len(args.ep_values):
        sys.exit("Error: Number of --ea_values and --ep_values must be the same.")

    # --- More Validation / Info ---
    if args.model == '1':
         # Check unnecessary grid params
         is_default_swap = (len(args.swap_probs) == 1 and args.swap_probs[0] == 0.0)
         is_default_chunk = (args.chunk_size == 10)
         if not is_default_swap or not is_default_chunk:
             print("Info: --swap_probs and --chunk_size arguments are ignored when --model=1 is selected.", file=sys.stderr)

    if args.model == '2' or args.model == 'both':
         if args.chunk_size <= 0:
             print("Warning: chunk_size must be positive for model 2/both. Setting to 1.", file=sys.stderr)
             args.chunk_size = 1
         # Warn if m not div by chunk_size
         if args.interactions % args.chunk_size != 0:
             print(f"Warning: interactions ({args.interactions}) not divisible by chunk_size ({args.chunk_size}). "
                   f"Total interactions per gen will be { (args.interactions // args.chunk_size) * args.chunk_size}.", file=sys.stderr)

    # --- Parameter Setup ---
    n = args.size
    m = args.interactions
    mr = args.mutation
    q_vals = args.q_values
    ea_vals = args.ea_values
    ep_vals = args.ep_values
    runs = args.runs
    generations = args.generations
    # Model 2/both params
    swap_probabilities_model2 = args.swap_probs
    chunk_s_model2 = args.chunk_size

    # --- Task Generation ---
    tasks = []
    print(f"\nSelected mode: --model {args.model}")

    model_1_tasks_generated = False
    model_2_tasks_generated = False

    # --- Generate tasks for Model 1 ---
    if args.model == '1' or args.model == 'both':
        print("--> Generating tasks for Model 1 (Global)...")
        model_1 = 1
        swap_prob_placeholder = 0.0  # Placeholder key for Model 1
        chunk_size_placeholder = 1   # Placeholder chunk size
        for q in q_vals:
            for i in range(len(ea_vals)):
                ea = ea_vals[i]
                ep = ep_vals[i]
                for run_idx in range(runs):
                    task_specific_seed = master_rng.randint(0, 2**32 - 1) # Ensure seed is within valid range
                    tasks.append((model_1, q, ea, ep, generations, n, m, mr,
                                  swap_prob_placeholder, chunk_size_placeholder,
                                  task_specific_seed))
        model_1_tasks_generated = True

    # --- Generate tasks for Model 2 ---
    if args.model == '2' or args.model == 'both':
        print(f"--> Generating tasks for Model 2 (Local Grid) with swap_probs={swap_probabilities_model2}...")
        model_2 = 2
        for q in q_vals:
            for i in range(len(ea_vals)):
                ea = ea_vals[i]
                ep = ep_vals[i]
                # Iterate Model 2 swap_probs
                for swap_prob in swap_probabilities_model2:
                     for run_idx in range(runs):
                         task_specific_seed = master_rng.randint(0, 2**32 - 1) # Ensure seed is within valid range
                         tasks.append((model_2, q, ea, ep, generations, n, m, mr,
                                       swap_prob, chunk_s_model2, # Use actual parameters
                                       task_specific_seed))
        model_2_tasks_generated = True

    if not model_1_tasks_generated and not model_2_tasks_generated:
        sys.exit("Error: No tasks were generated. Check model selection and parameters.")


    # --- Simulation Execution ---
    print(f"\nTotal simulations to run: {len(tasks)}")
    num_workers = min(args.workers, len(tasks))
    if num_workers <= 0: sys.exit("Error: No tasks to run or zero workers specified.")
    print(f"Using {num_workers} worker processes.")

    # Create the pool of workers
    # Optional: maxtasksperchild=1 for leaks
    pool = mp.Pool(processes=num_workers) #, maxtasksperchild=1)
    results = []
    start_time = time.time()

    # Process tasks using the pool
    with tqdm(total=len(tasks), desc="Simulations", unit="run") as pbar:
        # Use imap_unordered for performance
        for result in pool.imap_unordered(simulation_worker, tasks):
            if result is not None:
                # Expecting 7 elements: (model, q, ea, ep, swap_prob, chunk_s, avg_reward)
                if len(result) == 7:
                    # Check for valid (non-NaN) result
                    if not np.isnan(result[-1]):
                        results.append(result)
                    else:
                        # Log NaN results
                        m_err, q_err, ea_err, ep_err, swap_err, chunk_s_err, _ = result
                        print(f"\nWarning: Skipped NaN result for model {m_err}, "
                              f"params: q={q_err:.3f}, ea={ea_err:.3f}, ep={ep_err:.3f}, swap={swap_err:.3f}, chunk={chunk_s_err}", file=sys.stderr)
                else:
                    # Log malformed worker results
                    print(f"\nWarning: Received malformed result tuple from worker: {result}", file=sys.stderr)
            else:
                # Log None results (shouldn't happen)
                print(f"\nWarning: Received None result from worker.", file=sys.stderr)
            pbar.update(1) # Update progress bar

    pool.close() # Prevent sending more tasks
    pool.join()  # Wait for workers
    end_time = time.time()
    print(f"\nSimulations finished in {end_time - start_time:.2f} seconds.")

    # --- Results Aggregation ---
    # Aggregate results by key
    # Key: (model, q, ea, ep, swap_prob, chunk_size)
    data = {}
    for res_tuple in results:
       # Check tuple length
       if len(res_tuple) == 7:
            # Unpack the result tuple
            model_res, q_res, ea_res, ep_res, swap_res, chunk_res, avg = res_tuple
            # Create the key for aggregation
            key = (model_res, q_res, ea_res, ep_res, swap_res, chunk_res)
            # Append result to key's list
            data.setdefault(key, []).append(avg)
       else:
            # Safeguard check
            print(f"\nWarning: Skipping malformed result tuple during aggregation: {res_tuple}", file=sys.stderr)


    # --- Output CSV ---
    # Use --output filename (default results.csv)
    output_filename = args.output

    # Define CSV headers
    headers = ["model", "q", "ea", "ep", "swap_prob", "chunk_size", "avg_reward", "std_dev", "num_runs"]
    output_rows = [headers]
    print("\n--- Aggregated Results Summary ---")

    # Sort keys for consistent output
    # Sort by: model, q, noise, swap, chunk
    for k in sorted(data.keys(), key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5])):
        payoffs = data.get(k, []) # Get rewards list for key
        if payoffs: # Check if results exist
            mean_payoff = np.mean(payoffs)
            std_dev = np.std(payoffs)
            num_runs_completed = len(payoffs) # Num successful runs
            # Prepare row data
            row_data = list(k) + [mean_payoff, std_dev, num_runs_completed]
            output_rows.append(row_data)
            # Print summary to console
            print(f"Model={k[0]} q={k[1]:.3f} ea={k[2]:.3f} ep={k[3]:.3f} swap={k[4]:.3f} chunk={k[5]:<3} -> "
                  f"Avg={mean_payoff:.4f} Std={std_dev:.4f} Runs={num_runs_completed}/{args.runs}")
        else:
            # Shouldn't happen, but good practice
             print(f"Warning: No valid results found during aggregation for key: {k}", file=sys.stderr)

    # Write aggregated results to CSV
    try:
        with open(output_filename, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(output_rows)
        print(f"\nAggregated results saved to file: {output_filename}")
    except IOError as e:
        print(f"\nError writing results to file {output_filename}: {e}", file=sys.stderr)
    except csv.Error as e:
        print(f"\nError writing CSV data to file {output_filename}: {e}", file=sys.stderr)


if __name__ == "__main__":

    main()
