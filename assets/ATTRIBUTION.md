# Human pose model provenance

The multi-view 2D detection and 33-point keypoint frontend uses the official OpenMMLab `YOLOX-M`, `RTMPose-M` (COCO17 body), and `RTMW-L` (133-point whole-body hand/foot details) ONNX exports distributed through RTMLib's pinned model catalog. Model SHA256 values are recorded in `model_checksums.json`. Weights remain in the external RTMLib cache.
Research reference: [RTMPose](https://arxiv.org/abs/2303.07399).

# Unitree G1 robot descriptions

`unitree/g1_23dof.urdf` and `unitree/g1_29dof.urdf` are unmodified files from [Unitree Robotics unitree_ros](https://github.com/unitreerobotics/unitree_ros/tree/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description).
They are distributed under the BSD 3-Clause license retained in `unitree/LICENSE`.
Pinned revisions and SHA256 hashes are recorded in `unitree/manifest.json`.
Meshes are not included in Git. Inclusion does not imply hardware verification or endorsement.

# Human full-body imitation controllers (TWIST2 / GEAR-SONIC / BeyondMimic)

Pinned ONNX trackers, G1 29-DoF dynamics meshes/XML, and deployment references are recorded in `humanoid/manifest.json`:

- **TWIST2**: [amazon-far/TWIST2](https://github.com/amazon-far/TWIST2/tree/b06178f19a22f2138cbd31f60c6d494bc263f67d), MIT License, copyright Yanjie Ze (`humanoid/TWIST2-LICENSE.txt`).
- **GEAR-SONIC**: NVIDIA GEAR-SONIC release checkpoints (`v1.0` and `v1.1` encoder + decoder ONNX policies), Apache 2.0 License (`humanoid/GEAR-SONIC-LICENSE.txt`).
- **BeyondMimic**: Hybrid Robotics BeyondMimic G1 checkpoint (`humanoid/BeyondMimic-LICENSE.txt`).

Large model weights and meshes remain in the external runtime cache (`%LOCALAPPDATA%/multiview-pose-sim/robot-assets`). We port the policy observation/PD loops to an offline NumPy/ONNX Runtime adapter with explicit reference validity and command boundaries. No upstream endorsement or physical robot validation is implied.
