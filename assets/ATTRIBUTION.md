# Example human asset

Cesium Man, copyright 2017 Cesium, CC BY 4.0.

Source: https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/CesiumMan

Asset: https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/CesiumMan/glTF-Binary/CesiumMan.glb

License: https://creativecommons.org/licenses/by/4.0/

The embedded Cesium logo is separately identified as a Cesium trademark in the
upstream metadata. No endorsement is implied. The original asset is preserved;
the demo scales and positions the imported character and renders its animation.

This stylized sample is for testing the pipeline, not a representative benchmark
of real human image detection accuracy. Replace it with suitably licensed human
assets and diverse motions for dataset production.

# Human pose model provenance

Experimental human body-head comparison uses the official OpenMMLab RTMPose-M
256×192 and RTMPose-X 384×288 ONNX exports distributed through RTMLib's pinned
model catalog. URLs, input sizes and inference settings are recorded in
`outputs/vision_joint_round/body_head_development/freeze.json`; model SHA256
values are retained in `model_checksums.json` under `experimental_body_heads`.
Weights remain in the external RTMLib cache. The X model is an experimental
comparison, not an automatically selected runtime model.
Research reference: [RTMPose](https://arxiv.org/abs/2303.07399).

# Unitree G1 robot descriptions

`unitree/g1_23dof.urdf` and `unitree/g1_29dof.urdf` are unmodified files from
[Unitree Robotics unitree_ros](https://github.com/unitreerobotics/unitree_ros/tree/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description).
They are distributed under the BSD 3-Clause license retained in `unitree/LICENSE`.
Pinned revisions and SHA256 hashes are recorded in `unitree/manifest.json`.
Meshes are not included. Inclusion does not imply hardware verification or endorsement.

# Markerless Panda / G1 dynamics research baseline

Large upstream assets stay outside Git in the isolated runtime's `robot-assets` cache.
Every file, URL/revision and SHA256 is recorded in `robot_baseline/manifest.json`.

- [NVIDIA DREAM](https://github.com/NVlabs/DREAM/tree/3360f2aa45f66a58eaa70d0d40f2c46c2682c0bb):
  Panda VGG-Q checkpoint, architecture, decoder reference and Panda-3Cam data links.
  Retained [NVIDIA Source Code License-NC](robot_baseline/licenses/DREAM-LICENSE.md).
  We convert verified weights to ONNX and reject ambiguous peaks. Image inference does not use known-joint PnP;
  dataset labels are evaluation/calibration only.
- [MuJoCo Menagerie Panda](https://github.com/google-deepmind/mujoco_menagerie/tree/62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35/franka_emika_panda):
  visual/collision meshes and articulated XML, retained [Apache 2.0 license](robot_baseline/licenses/Panda-Menagerie-LICENSE.txt).
  We add cameras, lighting and a floor; robot appearance is not modified with markers.
- [OpenHomie](https://github.com/InternRobotics/OpenHomie/tree/33071f79f1884bb122bd45631c8c60d61dac1eb8):
  released G1 locomotion policy, 27-active-joint physics model and deployment observation/gain reference.
  Retained [CC BY-NC 4.0 license](robot_baseline/licenses/OpenHomie-LICENSE.txt).
  The local adapter performs independent simulation with visual upper-body tasks and explicit watchdogs.

DREAM and OpenHomie include non-commercial restrictions. This research baseline does not grant commercial deployment rights.
No upstream endorsement or physical hardware validation is implied.

# Human full-body imitation / TWIST2

[TWIST2](https://github.com/amazon-far/TWIST2/tree/b06178f19a22f2138cbd31f60c6d494bc263f67d), MIT, copyright Yanjie Ze.
Pinned ONNX tracker, G1 dynamics meshes/XML and deployment reference are recorded in `humanoid/manifest.json`; license in `humanoid/TWIST2-LICENSE.txt`. Large files remain in the external runtime cache. We port the policy observation/PD loop to an offline NumPy/ORT adapter with explicit reference validity and command boundaries. No upstream endorsement or physical robot validation is implied.

The Panda / OpenHomie section above is historical attribution for retained cached assets; those algorithms are no longer the project demo.

# Experimental PMPose observation head

[BBoxMaskPose / PMPose](https://github.com/MiraPurkrabek/BBoxMaskPose/tree/49a070a6f8396147323b1c4959474077dbe2ce8e), GPL-3.0, by Miroslav Purkrabek and contributors. The unmodified official source, its license and the verified PMPose-S checkpoint remain in the external runtime cache. The optional adapter calls the official network and crop pipeline; its no-mask ablation is described in [the experiment](../docs/pmpose_experiment.md). This is not a reproduction of the complete segmentation/iteration pipeline, an upstream endorsement, or permission for deployment under a different license.
