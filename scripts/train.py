import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pong_env import PongEnv


def main():
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    run_name = "05"

    run_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    env = PongEnv()
    env = Monitor(env, filename=run_dir)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    eval_env = PongEnv(render_mode="human")
    eval_env = Monitor(eval_env, filename=run_dir)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=50_000,
        n_eval_episodes=1,
        render=True,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.9,
        verbose=0,
        tensorboard_log=run_dir,
        device="cpu",
        stats_window_size=10,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128, 64], vf=[128, 128, 64]),
        ),
    )

    model.learn(total_timesteps=10_000_000, progress_bar=True, callback=eval_callback, tb_log_name="run")

    model.save(os.path.join(run_dir, "final_model"))
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
