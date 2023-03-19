# Stacking Env
A gym-like environment for panda stack.
## Installation
```
# clone this repo
cd stacking_env
conda env create -f=environment.yaml
conda activate stacking
```
## Run random policy
```
 python random_policy.py --env BulletStack-v1 --n_object 6 --n_to_stack 1 2 3 4 5 6
```

## Try a simple motion planner
Checkout this repo to planning branch. Checkout onpolicy_algorithm to distill-visual. Then run this command from onpolicy_algorithm folder.
```
python play_robot_trajectory.py --config config.bullet_pixel_stack_plan --use_rl --load_path logs/ppo_BulletPixelStack-v1/slot_attn_rl_newdata_newenc_round123_xy41/model_0.pt --task_path test_tasks_raw.pkl
```