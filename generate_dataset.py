from matplotlib import pyplot as plt
import numpy as np
from sim import Simple2DCross

def generate_dataset(num_episodes=100, rs_start=0, rs_end=3):
    "Generate a dataset containing trajectories (obs) with random FSM parameters"
    env = Simple2DCross(num_pedestrians=5)
    obs_dict = {}
    rs_list = []

    for eid in range(num_episodes):
        obs_list = []
        repulsion_strength = np.random.randint(rs_start, rs_end+1)
        rs_list.append(repulsion_strength)
        obs, info = env.reset()
        obs_list.append(obs)

        terminated = False
        truncated = False

        print(f"Running episode {eid}...")
        while not (terminated or truncated):
            action = env.auto_action(repulsion_radius=1.0, repulsion_strength=repulsion_strength)
            obs, reward, terminated, truncated, info = env.step(action)
            obs_list.append(obs)
            env.render()
        obs_dict[str(eid)] = np.array(obs_list)

    env.close()
    print("All episodes finished.")
    np.savez(f"obs_data_rsRandInt{rs_start}-{rs_end}_numEpi{num_episodes}.npz", **obs_dict)
    np.save(f"rs_rsRandInt{rs_start}-{rs_end}_numEpi{num_episodes}.npy", rs_list)

if __name__ == "__main__":
    num_episodes=300
    rs_start=0
    rs_end=4
    generate_dataset(num_episodes, rs_start, rs_end)

    obs_dict = np.load(f"obs_data_rsRandInt{rs_start}-{rs_end}_numEpi{num_episodes}.npz") 
    plt.figure(figsize=(5,4)) 
    for obs in obs_dict.values(): 
        plt.plot(obs[:,0], obs[:,1]) 
        plt.xlim(-4.0, 4.0) 
        plt.ylim(-4.0, 4.0) 
    plt.gca().set_aspect("equal", adjustable="box") 

    rs = np.load(f"rs_rsRandInt{rs_start}-{rs_end}_numEpi{num_episodes}.npy") 
    counts = np.bincount(rs, minlength=rs_end+1)

    plt.figure(figsize=(5, 4))
    plt.bar(np.arange(rs_end+1), counts)
    plt.xticks(np.arange(rs_end+1))
    plt.xlabel("Repulsion Strength")
    plt.ylabel("Count")
    plt.tight_layout()

    plt.show()