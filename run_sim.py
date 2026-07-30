import imageio
import numpy as np
import matplotlib.pyplot as plt
from sim import Simple2DCross

def run_episodes(num_episodes=10, repulsion_strength=2.0):
    "Run a number of episodes without visualisation"
    env = Simple2DCross(num_pedestrians=5)
    obs_dict = {}

    for eid in range(num_episodes):
        obs_list = []
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
    np.savez(f"obs_data_rs{repulsion_strength}_.npz", **obs_dict)



def view_single_episode():
    "View a single episode in real-time"
    env = Simple2DCross(num_pedestrians=5, render_mode="human")
    obs, info = env.reset()

    terminated = False
    truncated = False

    print("Running native Pygame visualisation...")
    while not (terminated or truncated):
        action = env.auto_action(repulsion_radius=1.0, repulsion_strength=2.0)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

    env.close()
    print("Episode finished.")


def record_single_episode():
    "Record a single episode as mp4"
    env = Simple2DCross(num_pedestrians=5, render_mode="rgb_array")
    obs, info = env.reset()

    frames = []
    done = False

    print("Simulating episode...")

    while not done:
        frame = env.render()
        frames.append(frame)

        action = env.auto_action(repulsion_radius=1.0, repulsion_strength=2.0)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()
    print(f"Episode finished in {info['step']} steps. Saving video...")

    imageio.mimsave("episode_replay.mp4", frames, fps=10)
    print("Video saved as 'episode_replay.mp4'!")


if __name__ == "__main__":
    # for rs in np.linspace(0, 5, 6):
    #     run_episodes(num_episodes=10, repulsion_strength=rs)

    fig, axes = plt.subplots(1, 6, figsize=(30, 4), sharex=True, sharey=True)

    for ax, rs in zip(axes, np.linspace(0, 5, 6)):
        obs_dict = np.load(f"obs_data_rs{rs}_.npz")

        for obs in obs_dict.values():
            ax.plot(obs[:, 0], obs[:, 1], linewidth=0.8)

        ax.set_xlim(-4.0, 4.0)
        ax.set_ylim(-4.0, 4.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"rs = {rs:g}")

    fig.tight_layout()
    plt.show()