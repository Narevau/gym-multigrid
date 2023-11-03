from gym_multigrid.multigrid import *

class DoorGameEnv(MultiGridEnv):
    def __init__(
        self,
        width=None,
        height=None,
        agents_index = [],
        view_size=None,
    ):
        self.world = World

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))

        super().__init__(
            max_steps= 1000,
            width=width,
            height=height,
            agents=agents,
            agent_view_size=view_size,  
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        # Door walls
        self.grid.vert_wall(self.world, width-5, 0, 3)
        self.grid.vert_wall(self.world, width-5, height-3, 3)

        self.grid.set(6, 3, Door(self.world, 'red', is_open=False, is_locked=True))
       

        # Randomize the player start position and orientation
        for a in self.agents:
            self.place_agent(a)

    def step(self, actions):
        obs, rewards, done, info = MultiGridEnv.step(self, actions)
        return obs, rewards, done, info

class DoorGame1(DoorGameEnv):
    def __init__(self):
        super().__init__(
        width=11,
        height=7,
        agents_index = [1,1],
        view_size=7,)

# d = DoorGame1()