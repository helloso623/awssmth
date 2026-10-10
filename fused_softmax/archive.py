"""Unmodified authors' PUCT behavior; only the softmax state/environment changes."""
import hashlib
from upstream import PUCTSampler, State

def kernel_state(source, reward, feedback, step=0):
    return State(timestep=step, construction=[hashlib.sha256(source.encode()).hexdigest()],
                 code=source, value=reward, observation=feedback)

class KernelEnvironment:
    state_type = State
    seed = None
    @classmethod
    def create_initial_state(cls, problem_type):
        return State.from_dict(cls.seed.to_dict())

def open_archive(path, seed, resume_step=None):
    KernelEnvironment.seed = seed
    return PUCTSampler(str(path), KernelEnvironment, max_buffer_size=1000,
        batch_size=1, resume_step=resume_step, puct_c=1., topk_children=2)
