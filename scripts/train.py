import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor  # Import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pong_env import PongEnv


def main():
    # Create base log directory
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Create a unique run name based on timestamp
    timestamp = int(time.time())
    run_name = f"05"

    # Create run directory inside the base log directory
    run_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Create training environment without rendering
    env = PongEnv()
    env = Monitor(env, filename=run_dir)  # Wrap in Monitor
    env = DummyVecEnv([lambda: env])  # Wrap in DummyVecEnv
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)  # Use VecNormalize

    # Create evaluation environment with rendering
    eval_env = PongEnv(render_mode="human")
    eval_env = Monitor(eval_env, filename=run_dir)  # Wrap in Monitor
    eval_env = DummyVecEnv([lambda: eval_env])  # Wrap in DummyVecEnv
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)  # Use VecNormalize

    # Create the evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,  # Save best model to run directory
        log_path=run_dir,  # Save eval logs to eval directory
        eval_freq=50_000,
        n_eval_episodes=1,
        render=True,
        deterministic=True,
        verbose=1,
    )

    # Initialize the agent with custom parameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.9,
        verbose=0,
        tensorboard_log=run_dir,  # Save TensorBoard logs to run directory
        device="cpu",
        stats_window_size=10,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128, 64], vf=[128, 128, 64]),  # Deeper actor network  # Deeper critic network
        ),
    )

    # Train the agent
    model.learn(total_timesteps=10_000_000, progress_bar=True, callback=eval_callback, tb_log_name="run")

    # Save the final model in the same directory as logs
    model.save(os.path.join(run_dir, "final_model"))  # Save final model to run directory
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
