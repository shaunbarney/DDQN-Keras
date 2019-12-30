import gym
import numpy as np
from ddqn import Agent
from envs.cartpole.setup import CartPoleSetup

def train_setup(agent: Agent, env: gym.Env):
    num_games = 150
    n_step = 0
    best_score = 0
    scores = []

    for i in range(num_games):
        done = False
        state = env.reset()
        score = 0

        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, _ = env.step(action)
            n_step += 1
            score += reward

            agent.remember(state, action, reward, new_state, int(done))
            agent.learn()
            state = new_state

        scores.append(score)

        print(f"Game: {i}\tScore: {score}\tEpsilon: {agent.epsilon}'\t")
        avg_score = np.mean(scores[-100:])
        if avg_score > best_score:
            print("New best average.")
            agent.save_models()
            best_score = avg_score
        
def test(agent: Agent, env: gym.Env):
    pass


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    setup = CartPoleSetup(env)
    agent = Agent(setup)
    train_setup(agent, env)