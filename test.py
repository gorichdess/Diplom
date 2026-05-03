import torch
import time
from robot_env import RobotEnv
from model import ActorCritic


def test():
    env = RobotEnv(render=True)
    model = ActorCritic(env.obs_dim, 2)
    model.load_state_dict(torch.load("checkpoint_250.pth", map_location="cpu"))
    model.eval()

    state, _ = env.reset()
    done = False

    while not done:
        with torch.no_grad():
            action, _, _ = model.get_action(state, deterministic=True)
        state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        time.sleep(1/60)

    time.sleep(3)
    env.close()


if __name__ == "__main__":
    test()