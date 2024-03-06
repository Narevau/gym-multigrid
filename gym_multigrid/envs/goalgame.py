from gym_multigrid.multigrid import *
from ray.rllib.env.env_context import EnvContext
from typing import List

class GoalGameEnv(MultiGridEnv):
    def __init__(
        self,
        env_config: EnvContext,
    ):
        self.world = World
        self.width = env_config["width"]
        self.height = env_config["height"]
        self.agents_index = env_config["agents_index"]
        self.agents: List[Agent] = []
        self.view_size = env_config["view_size"]
        self.partial_obs = env_config["partial_obs"]
        self.goal_coord = env_config["goal_coord"]
        self.max_steps = env_config["max_steps"]
        self.agents_coords = env_config["agents_coords"]
        self.see_through_walls = env_config.get('see_through_walls', False)
        self.decaying_reward = env_config.get('decaying_reward', False)

        for i in range(len(self.agents_index)):
            agent = Agent(self.world, self.agents_index[i], view_size=self.view_size)
            agent.dir = 0
            agent.pos = self.agents_coords[i]
            agent.init_pos = self.agents_coords[i]
            self.agents.append(agent)  
         
        super().__init__(
        max_steps=self.max_steps,
        width=self.width,
        height=self.height,
        agents=self.agents,
        agent_view_size=self.view_size,
        partial_obs=self.partial_obs,
        see_through_walls=self.see_through_walls,
    )   
        
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        # Goal
        self.grid.set(self.goal_coord[0], self.goal_coord[1], Goal(self.world,0,reward=1,color=1))

        # Place the agents
        for i in range(len(self.agents)):
            self.grid.set(self.agents_coords[i][0], self.agents_coords[i][1], self.agents[i])
            self.agents[i].pos = self.agents_coords[i]
            self.agents[i].dir = 0

    def step(self, actions):
        obs, rewards, done, truncated, info = MultiGridEnv.step(self, actions)
       
        reward = rewards[0]
        return obs, reward, done, truncated, info
    
    def reset(self, *, seed=None, options=None):
        obs, info = MultiGridEnv.reset(self)
        
        return obs, info
    
    def _reward(self, i, rewards,reward=1):
        if self.decaying_reward:
            rewards[i] = 1 - 0.9 * (self.step_count / self.max_steps)
        else:
            rewards[i] = rewards[i] + reward