class LearningRateScheduler:
    """Base class for learning rate schedulers."""
    
    def __init__(self, initial_lr: float):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
    
    def step(self, epoch: int) -> float:
        """Update learning rate based on epoch."""
        raise NotImplementedError
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr


class StepLR(LearningRateScheduler):
    """Step learning rate scheduler."""
    
    def __init__(self, initial_lr: float, step_size: int, gamma: float = 0.1):
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma
    
    def step(self, epoch: int) -> float:
        """Reduce learning rate by gamma every step_size epochs."""
        if epoch > 0 and epoch % self.step_size == 0:
            self.current_lr *= self.gamma
        return self.current_lr


class ExponentialLR(LearningRateScheduler):
    """Exponential learning rate decay."""
    
    def __init__(self, initial_lr: float, gamma: float = 0.95):
        super().__init__(initial_lr)
        self.gamma = gamma
    
    def step(self, epoch: int) -> float:
        """Exponential decay: lr = initial_lr * gamma^epoch."""
        self.current_lr = self.initial_lr * (self.gamma ** epoch)
        return self.current_lr
