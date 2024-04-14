# Stereo MSCKF Code

## Course: [WPI's](https://www.wpi.edu/) [RBE/CS549](https://nitinjsanket.github.io/teaching/rbe549/spring2024.html)
## Instructor: [Prof. Nitin J. Sanket](https://nitinjsanket.github.io/)

See [the course website](https://nitinjsanket.github.io/teaching/rbe549/fall2022.html) for more details. The code was modified by [Dr. Lening Li.](https://lening.li/)

MSCKF (Multi-State Constraint Kalman Filter) is an EKF based **tightly-coupled** visual-inertial odometry algorithm. [S-MSCKF](https://arxiv.org/abs/1712.00036) is MSCKF's stereo version. This project is a Python reimplemention of S-MSCKF, the code is directly translated from official C++ implementation [KumarRobotics/msckf_vio](https://github.com/KumarRobotics/msckf_vio).  

Currently this code also consists of the [RPG Toolbox](https://github.com/uzh-rpg/rpg_trajectory_evaluation)


For algorithm details, please refer to:
* Robust Stereo Visual Inertial Odometry for Fast Autonomous Flight, Ke Sun et al. (2017)
* A Multi-State Constraint Kalman Filterfor Vision-aided Inertial Navigation, Anastasios I. Mourikis et al. (2006)  
* Zichao Zhang, Davide Scaramuzza: A Tutorial on Quantitative Trajectory Evaluation for Visual(-Inertial) Odometry, IEEE/RSJ Int. Conf. Intell. Robot. Syst. (IROS), 2018.

## Requirements
* Python 3.6+
* numpy
* scipy
* cv2
* [pangolin](https://github.com/uoip/pangolin) (optional, for trajectory/poses visualization)
* [RPG Toolbox](https://github.com/uzh-rpg/rpg_trajectory_evaluation) - Required functions also exists in this toolbox

## Dataset
* [EuRoC MAV](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets): visual-inertial datasets collected on-board a MAV. The datasets contain stereo images, synchronized IMU measurements, and ground-truth.  
This project implements data loader and data publisher for EuRoC MAV dataset.

## Run  
`python vio.py --view --path path/to/your/EuRoC_MAV_dataset/MH_01_easy`  
or    
`python vio.py --path path/to/your/EuRoC_MAV_dataset/MH_01_easy` (no visualization)  

## RUN - Updated with Results and Video Saver
    python3 vio.py --path <dataset_path> --view --save_pred --save_video

#### Args explanation
    --path - Dataset path usually EuRoC Dataset works with this code
    --view - To view the video or not
    --save_pred - To save the (Timestamp, tx,ty,tz,qx,qy,qz,qw) of the results into txt for further evaluation using RPG Toolbox
    --save_video - To save the Video output - This is a frame buffer grabbed from OpenGL Output Viewer

## Ground Truth Conversion from ASL format to .txt based format for RPG Toolbox
    python3 tools/asl_groundtruth_to_pose.py --gt <Ground Truth CSV Path> --output <Output Path- Usually results>

## To generate plots using RPG Toolbox
    cd tools
    cd rpg_toolbox 
    python3 scripts/analyze_trajectory_single.py <Results_path>

Results_path should contain the files
* stamped_groundtruth.txt - Ground truth of ASL generated after conversion
* stamped_traj_estimate.txt - Trajectory estimate generated from the VIO script
* ALso createa file eval_cfg.yaml with the below configuration for se3 alignment
    * align_type: se3
    * align_num_frames*: -1


## Results
results/ 
* Contains Predicted Poses
* Contains Pose Plots generated from RPG toolbox


## License and References
Follow [license of msckf_vio](https://github.com/KumarRobotics/msckf_vio/blob/master/LICENSE.txt). Code is adapted from [this implementation](https://github.com/uoip/stereo_msckf).