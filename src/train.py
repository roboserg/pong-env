from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from pong_env import PongEnv

def main():
    # Create log directory
    log_dir = "./logs/pong_ppo/"
    
    # Create training environment without rendering
    env = PongEnv()
    
    # Create evaluation environment with rendering
    eval_env = PongEnv(render_mode="human")

    # Create the evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}best_model",
        log_path=f"{log_dir}eval/",
        eval_freq=30000,
        n_eval_episodes=1,
        render=True,
        deterministic=True,
        verbose=1
    )

    # Initialize the agent with custom parameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.9,
        verbose=0,
        tensorboard_log=log_dir,
        device="cpu",
        policy_kwargs=dict(
            net_arch=dict(
                pi=[128, 128, 64],  # Deeper actor network
                vf=[128, 128, 64]   # Deeper critic network
            ),
        )
    )

    # Train the agent
    model.learn(
        total_timesteps=10_000_000,
        progress_bar=True,
        tb_log_name="PPO",
        callback=eval_callback
    )

    # Save the final model in the same directory as logs
    model.save(f"{log_dir}final_model")

    # Close environments
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
