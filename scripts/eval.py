import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pong_env import PongEnv


def evaluate_model(model_path, num_episodes=10, render=True):
    """
    Loads a trained model and evaluates it in the Pong environment.

    Args:
        model_path (str): Path to the trained model.
        num_episodes (int): Number of evaluation episodes.
        render (bool): Whether to render the environment during evaluation.
    """
    model_dir = os.path.dirname(model_path)

    # Create the evaluation environment
    env = PongEnv(render_mode="human" if render else None)
    env = Monitor(env, filename=model_dir)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    # Load the VecNormalize statistics, if they exist
    vec_normalize_path = os.path.join(model_dir, "vecnormalize.pkl")
    if os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("VecNormalize statistics not found, running without normalization.")

    # Load the trained model
    model = PPO.load(model_path, env=env, device="cpu")

    total_rewards = 0
    episode_lengths = []

    for episode in range(num_episodes):
        obs = env.reset()  # Changed this line
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)  # Changed this line
            episode_reward += rewards[0]
            episode_length += 1
            done = dones[0]

            if render:
                env.render()
                # time.sleep(0.03)

        total_rewards += episode_reward
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")

    mean_reward = total_rewards / num_episodes
    mean_length = sum(episode_lengths) / num_episodes

    print(f"\nEvaluation Results:")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.2f}")

    env.close()


if __name__ == "__main__":
    model_path = "./logs/05/best_model.zip"
    evaluate_model(model_path, num_episodes=5, render=True)
