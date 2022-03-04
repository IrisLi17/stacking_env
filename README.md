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