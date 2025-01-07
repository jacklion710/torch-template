import subprocess
import sys

def run_sweep(sweep_id, num_agents, total_runs):
    # Calculate runs per agent
    runs_per_agent = total_runs // num_agents
    remaining_runs = total_runs % num_agents
    
    print(f"Starting {num_agents} agents...")
    print(f"Total runs requested: {total_runs}")
    print(f"Each agent will do {runs_per_agent} runs", end="")
    if remaining_runs:
        print(f" (plus {remaining_runs} extra runs distributed to first {remaining_runs} agents)")
    else:
        print()
    
    # Start all agents
    processes = []
    for i in range(num_agents):
        # Add extra run to first few agents if there are remaining runs
        agent_runs = runs_per_agent + (1 if i < remaining_runs else 0)
        print(f"Starting agent {i+1} (will do {agent_runs} runs)...")
        p = subprocess.Popen(['wandb', 'agent', '--count', str(agent_runs), sweep_id])
        processes.append(p)
    
    # Wait for all to complete
    for p in processes:
        p.wait()
    
    print("All agents completed!")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run_sweep.py SWEEP_ID NUM_AGENTS TOTAL_RUNS")
        sys.exit(1)
        
    sweep_id = sys.argv[1]
    num_agents = int(sys.argv[2])
    total_runs = int(sys.argv[3])
    
    run_sweep(sweep_id, num_agents, total_runs) 