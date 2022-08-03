import gym
import numpy as np
import enum

env = gym.make("LunarLander-v2")
observation, info = env.reset(seed=42, return_info=True)


class Action(enum.IntEnum):
    DO_NOTHING = 0
    RIGHT_FIRE = 1
    MAIN_FIRE = 2
    LEFT_FIRE = 3

reward_sum = 0
done = False
while not done:
    env.render()

    action = np.random.choice([Action.DO_NOTHING, Action.RIGHT_FIRE, Action.MAIN_FIRE, Action.LEFT_FIRE])
    observation, reward, done, info = env.step(action)

    reward_sum += reward
    coordinates = observation[0:1]
    velocities = observation[2:3]
    angle = observation[3]
    angular_velocity = observation[4]
    contact_ground_left = observation[5]
    contact_ground_right = observation[6]

print(reward_sum)
print(info)

env.close()