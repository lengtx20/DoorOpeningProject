# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Robot G1
#
# Shows how to set up a simulation of a G1 robot articulation
# from a USD stage using newton.ModelBuilder.add_usd().
#
# Command:  uv run -m newton.examples robot_g1 --viewer usd --output-path pathname.usd --num-frames 950
#
###########################################################################

import warp as wp
import pandas as pd
import numpy as np
import torch

import newton
import newton.examples
import newton.utils


class Example:
    def __init__(self, viewer, num_worlds=1, args=None):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 6 # does not matter if not using physics
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_worlds = num_worlds
        self.csv_path = newton.examples.get_asset("merged_50hz_log20.csv")
        self.csv_data = None
        self.current_frame = 0
        # Use args.num_frames if it exists (from create_parser)
        self.max_frames = args.num_frames if args and hasattr(args, 'num_frames') and args.num_frames else 900
        self.playback_mode = True  # Always enable playback mode

        self.viewer = viewer

        g1 = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(g1)
        g1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
        g1.default_shape_cfg.ke = 2.0e3
        g1.default_shape_cfg.kd = 1.0e2
        g1.default_shape_cfg.kf = 1.0e3
        g1.default_shape_cfg.mu = 0.75

        mjcf_filename = newton.examples.get_asset("g1_29dof_lock_waist_rev_1_0.xml")
        g1.add_mjcf(
            mjcf_filename,
            ignore_names=["floor", "ground"],
            xform=wp.transform(wp.vec3(0, 0, 1.3)),
        )
        print("using mjcf")

        for i in range(6, g1.joint_dof_count):
            g1.joint_target_ke[i] = 1000.0
            g1.joint_target_kd[i] = 5.0

        # approximate meshes for faster collision detection
        g1.approximate_meshes("bounding_box")

        builder = newton.ModelBuilder()
        builder.replicate(g1, self.num_worlds)

        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.add_ground_plane()

        self.model = builder.finalize()
        print("joint_key", len(self.model.joint_key))
        print("joint_key", self.model.joint_key) 

        # for xml file joint order to match csv file
        # 0-12 -> 7-19
        # 13 -> 19
        # 15-29 -> 20-34
        joint_start =  [ 0,  7 , 8 , 9 , 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
        joints = ['floating_base_joint', 
         'left_hip_pitch_joint', #7
         'left_hip_roll_joint', #8
         'left_hip_yaw_joint', #9
         'left_knee_joint', #10
         'left_ankle_pitch_joint', #11
         'left_ankle_roll_joint', #12
         'right_hip_pitch_joint', #13
         'right_hip_roll_joint', #14
         'right_hip_yaw_joint', #15
         'right_knee_joint', #16
                'right_ankle_pitch_joint', #17
                'right_ankle_roll_joint', #18
         'waist_yaw_joint', #19
         'waist_roll_link_joint', #20
         'torso_link_joint', #20
         'left_shoulder_pitch_joint', #20
           'left_shoulder_roll_joint', 
           'left_shoulder_yaw_joint',
             'left_elbow_joint', 
             'left_wrist_roll_joint', 
             'left_wrist_pitch_joint', 
             'left_wrist_yaw_joint', 
             'right_shoulder_pitch_joint', 
             'right_shoulder_roll_joint', 
             'right_shoulder_yaw_joint',
               'right_elbow_joint', 
               'right_wrist_roll_joint',
                 'right_wrist_pitch_joint',
                   'right_wrist_yaw_joint']
        
        for i in range(len(joints)):
            print(f"{i} '{joints[i]} {joint_start[i+1]-joint_start[i]}'")
            print("start:", joint_start[i], "end:", joint_start[i+1])

        print("body_count", self.model.body_count)
        print("joint_q", len(self.model.joint_q))
        print("joint_dof_count", self.model.joint_dof_count)
        print("joint_start", self.model.joint_q_start) 

        print("lengthjoint_start", len(self.model.joint_q_start)) # 40

        #515
        # only need this if you wnat to render a frame
        self.joint_demo  = [-0.267530,
            0.001227,
            0.010486,
            0.477375,
            -0.249103,
            0.000399,
            -0.248023,
            0.009877,
            -0.002424,
            0.458916,
            -0.237479,
            -0.035524,
            -1.203289,
            0.000000,
            0.000000,
            -0.322400,
            0.581810,
            0.948252,
            0.361133,
            0.104730,
            0.221430,
            -0.120709,
            0.075189,
            0.039596,
            0.041573,
            1.461908,
            -0.191712,
            0.079130,
            -0.161183]

        joint_init = [0.0] *34
        joint_init[0:7] = [4.615876, 1.292796, 0.720714, -0.002288,	0.012643,	0.215251, 0.976474]  # floating base
        joint_init[0:7] = [4.465375,	1.134937,	0.721901,-0.003387,	0.018121,	0.248279, 0.968513] 
        joint_init[7:19] = self.joint_demo[0:12]  # left leg
        joint_init[19] = self.joint_demo[13]  # waist
        joint_init[20:] = self.joint_demo[15:]  # left arm
        print("joint_init:", joint_init)

        # not using physics
        # self.solver = newton.solvers.SolverMuJoCo(
        #     self.model,
        #     use_mujoco_cpu=False,
        #     solver="newton",
        #     integrator="implicitfast",
        #     njmax=300,
        #     nconmax=150,
        #     cone="elliptic",
        #     impratio=100,
        #     iterations=100,
        #     ls_iterations=50,
        #     use_mujoco_contacts=args.use_mujoco_contacts if args else False,
        # )

        self.state_0 = self.model.state()
        self.state_0.joint_q = wp.array(joint_init , dtype=float)
    
        self.state_1 = self.model.state()
        self.state_1.joint_q = wp.array(joint_init , dtype=float)
        self.control = self.model.control()

        # Load CSV data if provided
        if self.playback_mode:
            # Try reading with different separators
            try:
                self.csv_data = pd.read_csv(self.csv_path, sep='\t')
                if len(self.csv_data.columns) == 1:
                    # Tab didn't work, try comma
                    self.csv_data = pd.read_csv(self.csv_path, sep=',')
            except:
                self.csv_data = pd.read_csv(self.csv_path)
            
            # Limit to max_frames
            if len(self.csv_data) > self.max_frames:
                self.csv_data = self.csv_data.iloc[:self.max_frames]
            
            print(f"Loaded {len(self.csv_data)} frames from CSV (max: {self.max_frames})")
            print(f"CSV columns: {list(self.csv_data.columns[:10])}")  # Print first 10 columns
            self.load_frame(0)

        # Evaluate forward kinematics
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

        print("bodykey", self.model.body_key) 
        # ['pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link', 'waist_yaw_link', 'waist_roll_link', 'torso_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link']
        body_q = wp.to_torch(self.state_0.body_q)
        print("left_wrist_yaw_link", body_q[22])
        print("right_wrist_yaw_link", body_q[29])


        # Create collision pipeline from command-line args (default: CollisionPipelineUnified with EXPLICIT)
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)

        self.capture()

    def load_frame(self, frame_idx):
        """Load a frame from CSV data into the state."""
        if self.csv_data is None or frame_idx >= len(self.csv_data):
            return
        
        row = self.csv_data.iloc[frame_idx]
        
        # Build joint_q as a list, then convert to wp.array
        joint_q = [0.0] * 34
        
        # Set floating base position and quaternion
        # CSV has: pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z
        joint_q[0] = row['pos_x']
        joint_q[1] = row['pos_y']
        joint_q[2] = row['pos_z']
        joint_q[3] = row['quat_x']  # Newton uses (x, y, z, w)
        joint_q[4] = row['quat_y']
        joint_q[5] = row['quat_z']
        joint_q[6] = row['quat_w']
        
        # Left+right legs: CSV q_0 to q_11 -> joint_q[7:19]
        for i in range(12):
            joint_q[7 + i] = row[f'q_{i}']
        
        # Waist: CSV q_13 -> joint_q[19]
        joint_q[19] = row['q_13']
        
        # Arms: CSV q_15 to q_28 -> joint_q[20:34] (14 values)
        for i in range(14):
            joint_q[20 + i] = row[f'q_{15 + i}']
        
        # Assign to state
        self.state_0.joint_q = wp.array(joint_q, dtype=float)
        
        # Build joint_qd as a list
        joint_qd = [0.0] * 33
        
        # Left+right legs velocities
        for i in range(12):
            joint_qd[6 + i] = row[f'dq_{i}']
        
        # Waist velocity
        joint_qd[18] = row['dq_13']
        
        # Arms velocities
        for i in range(14):
            joint_qd[19 + i] = row[f'dq_{15 + i}']
        
        # Assign to state
        self.state_0.joint_qd = wp.array(joint_qd, dtype=float)
        
        # Update forward kinematics
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            # self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.playback_mode:
            # Playback mode: load next frame from CSV
            self.current_frame += 1
            if self.current_frame >= len(self.csv_data):
                self.current_frame = 0  # Loop back to start
            self.load_frame(self.current_frame)
        else:
            # Simulation mode: run physics
            if self.graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.0,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all body velocities are small",
            lambda q, qd: max(abs(qd))
            < 0.015,  # Relaxed from 0.005 - G1 has higher residual velocities with unified pipeline
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=1, help="Total number of simulated worlds.")

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args.num_worlds, args)

    newton.examples.run(example, args)
