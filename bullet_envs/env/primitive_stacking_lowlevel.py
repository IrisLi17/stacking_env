from bullet_envs.env.primitive_stacking import ArmStack

class LowLevelEnv(ArmStack):
    def __init__(self, *args, actionRepeat=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.actionRepeat = actionRepeat

    def step_simulation(self):
        for _ in range(self.actionRepeat):
            self.p.stepSimulation()