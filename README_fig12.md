# Figure 12

The project contains Python scripts to generate Figure10 (Model 1: Global, Model 2: Local).

* `donation_game_model.py`- Runs the simulation for the given parameters and saves the results to results.csv. 
* `plot_figure12.py`- Generates a graph (payoff_vs_noise_plot.png) based on results.csv, showing the dependence of the average payoff on noise (e_a = e_p). 
* `requirements_fig12.txt`- list of required Python libraries.

## Installation
1. Make sure you have Python installed (>= 3.7). 
2. Open a terminal in the project directory. 
3. (Recommended) Create and activate a virtual environment:

Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements_fig12.txt
``` 

## Running
### Run the simulation: 
The following command will run simulations for both models with the given noise and mobility parameters (swap_prob), using 100 runs and 100000 generations. 
The results will be saved in results.csv. 

***NOTE***: This command can take a very long time to execute!

Parameters used to generate the simulation for Figure10 from the scientific study:
```bash
python donation_game_model.py --model both --ea_values 0.0 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 --ep_values 0.0 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 --swap_probs 0.0 0.05 0.10 --runs 100 --generations 100000 --mutation 0.001 --chunk_size 10
```

For more information about the simulation file: 
```bash
python donation_game_model.py -h
```  

### Generate graph

After the simulation is complete and results.csv is created, run: 
```bash
python plot_figure12.py
```
The graph will be saved as payoff_vs_noise_plot.png.


## Output Files
* `results.csv` - simulation results.
* `payoff_vs_noise_plot.png` - graph of the dependence of the average payoff on noise, depending on the model and swap_probs.
