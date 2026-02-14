"""
collect_demos.py

Final version for Language-Guided Handler Challenge

Task: Robosuite PickPlace
Robot: Franka Panda
Inputs saved: RGB images + action vectors + language instruction

Ground-truth state is used ONLY for expert control.
No GT state is saved in dataset.

Outputs:
- demo_data.h5
- initial_attempt.mp4
- initial_success.mp4
"""

import robosuite as suite
from robosuite.controllers import load_part_controller_config
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

NUM_SUCCESSFUL_EPISODES = 50
MAX_STEPS = 200

# Balanced language diversity (good for 50 demos)
LANGUAGE_TEMPLATES = [
    "Pick the object and place it in the bin.",
    "Move the object into the bin.",
    "Put the object in the bin.",
    "Pick it up and place it in the bin.",
    "Transfer the object to the bin.",
    "Place the object inside the bin."
]


# =========================
# Environment Creation
# =========================

def create_env():

    config = load_part_controller_config(default_controller="OSC_POSE")

    env = suite.make(
        env_name="PickPlace",
        robots="Panda",
        controller_configs=config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
        reward_shaping=True,
    )

    return env


# =========================
# Scripted Expert Policy
# =========================

def scripted_expert(obs, env):
    """
    Multi-phase expert:
    1. Move above object
    2. Close gripper
    3. Move above bin
    4. Release
    """

    # Object position (first 3 dims of object-state)
    object_pos = obs["object-state"][:3]

    eef_pos = obs["robot0_eef_pos"]
    gripper_qpos = obs["robot0_gripper_qpos"][0]

    # Get bin world position from Mujoco
    bin_body_id = env.sim.model.body_name2id("bin1")
    bin_pos = env.sim.data.body_xpos[bin_body_id]

    action = np.zeros(7)

    # ------------------------
    # Phase 1: Move above object
    # ------------------------
    target_obj = object_pos + np.array([0, 0, 0.10])
    delta_obj = target_obj - eef_pos
    action[:3] = np.clip(delta_obj, -0.05, 0.05)

    # ------------------------
    # Phase 2: Close gripper
    # ------------------------
    if np.linalg.norm(delta_obj) < 0.02:
        action[6] = 1.0

    # ------------------------
    # Phase 3: If grasped, move above bin
    # ------------------------
    if gripper_qpos < 0.02:
        target_bin = bin_pos + np.array([0, 0, 0.15])
        delta_bin = target_bin - eef_pos
        action[:3] = np.clip(delta_bin, -0.05, 0.05)

        # ------------------------
        # Phase 4: Release at bin
        # ------------------------
        if np.linalg.norm(delta_bin) < 0.03:
            action[6] = -1.0

    return action


# =========================
# Data Collection
# =========================

def collect_demos():

    # Clean previous files
    for file in [DATA_PATH, FIRST_ATTEMPT_VIDEO, FIRST_SUCCESS_VIDEO]:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted old file: {file}")

    env = create_env()

    data_file = h5py.File(DATA_PATH, "w")
    grp = data_file.create_group("data")

    successful_episodes = 0
    first_attempt_recorded = False
    first_success_recorded = False

    print("Starting PickPlace data collection...")

    while successful_episodes < NUM_SUCCESSFUL_EPISODES:

        obs = env.reset()

        instruction = random.choice(LANGUAGE_TEMPLATES)

        images = []
        actions = []
        attempt_video_frames = []
        success_video_frames = []

        for step in range(MAX_STEPS):

            action = scripted_expert(obs, env)
            obs, reward, done, info = env.step(action)

            img = obs["agentview_image"]

            images.append(img)
            actions.append(action)

            if not first_attempt_recorded:
                attempt_video_frames.append(img)

            if not first_success_recorded:
                success_video_frames.append(img)

            if env._check_success():
                break

        # Save first-ever attempt
        if not first_attempt_recorded:
            imageio.mimsave(FIRST_ATTEMPT_VIDEO, attempt_video_frames, fps=20)
            first_attempt_recorded = True
            print("Saved first attempt video.")

        # Save successful episodes only
        if env._check_success():

            ep_grp = grp.create_group(f"demo_{successful_episodes}")
            ep_grp.create_dataset("images", data=np.array(images))
            ep_grp.create_dataset("actions", data=np.array(actions))
            ep_grp.create_dataset("text", data=instruction)

            successful_episodes += 1
            print(f"Saved successful episode {successful_episodes}")

            if not first_success_recorded:
                imageio.mimsave(FIRST_SUCCESS_VIDEO, success_video_frames, fps=20)
                first_success_recorded = True
                print("Saved first success video.")

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
