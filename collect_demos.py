"""
collect_demos.py

Generates expert demonstrations for the Robosuite Lift task.

We use:
- Robot: Franka Panda
- Task: Lift
- Input saved: RGB images + action vectors + language instruction
- Ground-truth state is used ONLY for expert control (allowed during data collection).

Outputs:
- demo_data.h5
- initial_attempt.mp4
- initial_success.mp4
"""

import robosuite as suite
from robosuite.controllers import load_controller_config
import numpy as np
import h5py
import imageio
import os
import random


# =========================
# Configuration
# =========================

DATA_PATH = "demo_data.h5"
FIRST_ATTEMPT_VIDEO = "initial_attempt.mp4"
FIRST_SUCCESS_VIDEO = "initial_success.mp4"

NUM_SUCCESSFUL_EPISODES = 50   # Increased dataset size (balanced for 9-hour timeline)
MAX_STEPS = 150

# Small language variation (improves conditioning robustness)
LANGUAGE_TEMPLATES = [
    "Lift the cube.",
    "Pick up the cube.",
    "Grab the cube.",
    "Please lift the cube."
]


# =========================
# Expert Policy
# =========================

def scripted_expert(obs):
    """
    State-based expert controller for Lift task.

    Uses ground-truth object and robot state ONLY during data collection.
    This is allowed because the trained model will NOT receive GT state.
    """

    cube_pos = obs["cube_pos"]
    eef_pos = obs["robot0_eef_pos"]
    gripper_qpos = obs["robot0_gripper_qpos"][0]

    action = np.zeros(7)

    # Phase 1: Move above cube
    target_above = cube_pos + np.array([0, 0, 0.10])
    delta = target_above - eef_pos
    action[:3] = np.clip(delta, -0.05, 0.05)

    # Phase 2: Close gripper when aligned
    if np.linalg.norm(delta) < 0.02:
        action[6] = 1.0

    # Phase 3: Lift after grasp
    if gripper_qpos < 0.02:
        action[2] = 0.05

    return action


# =========================
# Environment Creation
# =========================

def create_env():
    """
    Creates Lift environment with:
    - Offscreen rendering (for RGB capture)
    - No on-screen GUI
    """

    config = load_controller_config(default_controller="OSC_POSE")

    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=config,
        has_renderer=False,               # No live GUI window
        has_offscreen_renderer=True,      # Enables RGB image capture
        use_camera_obs=True,
        camera_names="agentview",
        reward_shaping=True,
    )

    return env


# =========================
# Data Collection Loop
# =========================

def collect_demos():

    # Remove old files if they exist
    for file in [DATA_PATH, FIRST_ATTEMPT_VIDEO, FIRST_SUCCESS_VIDEO]:
        if os.path.exists(file):
            os.remove(file)

    env = create_env()

    data_file = h5py.File(DATA_PATH, "w")
    grp = data_file.create_group("data")

    successful_episodes = 0
    episode_counter = 0

    first_attempt_recorded = False
    first_success_recorded = False

    print("Starting data collection...")

    while successful_episodes < NUM_SUCCESSFUL_EPISODES:

        obs = env.reset()

        images = []
        actions = []
        attempt_video_frames = []
        success_video_frames = []

        # Randomly choose instruction for this episode
        instruction = random.choice(LANGUAGE_TEMPLATES)

        for step in range(MAX_STEPS):

            action = scripted_expert(obs)
            obs, reward, done, info = env.step(action)

            img = obs["agentview_image"]

            images.append(img)
            actions.append(action)

            # Record first-ever episode (even if failure)
            if not first_attempt_recorded:
                attempt_video_frames.append(img)

            # Record first successful episode
            if not first_success_recorded:
                success_video_frames.append(img)

            if env._check_success():
                break

        # Save first attempt video
        if not first_attempt_recorded:
            print("Saving first-ever attempt video...")
            imageio.mimsave(FIRST_ATTEMPT_VIDEO, attempt_video_frames, fps=20)
            first_attempt_recorded = True

        # Save only successful episodes
        if env._check_success():

            ep_grp = grp.create_group(f"demo_{successful_episodes}")
            ep_grp.create_dataset("images", data=np.array(images))
            ep_grp.create_dataset("actions", data=np.array(actions))
            ep_grp.create_dataset("text", data=instruction)

            successful_episodes += 1
            print(f"Saved successful episode {successful_episodes}")

            # Save first successful episode video
            if not first_success_recorded:
                print("Saving first successful episode video...")
                imageio.mimsave(FIRST_SUCCESS_VIDEO, success_video_frames, fps=20)
                first_success_recorded = True

        episode_counter += 1

    data_file.close()

    print("\nData collection complete.")
    print(f"Dataset saved to {DATA_PATH}")
    print(f"First attempt video saved to {FIRST_ATTEMPT_VIDEO}")
    print(f"First success video saved to {FIRST_SUCCESS_VIDEO}")


# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    collect_demos()
