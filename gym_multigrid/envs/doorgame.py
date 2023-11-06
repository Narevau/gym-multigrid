from gym_multigrid.multigrid import *
from typing import List

class DoorGameEnv(MultiGridEnv):
    def __init__(
        self,
        width=None,
        height=None,
        agents_index = [],
        view_size=None,
    ):
        self.world = World
        self.door = Door(self.world, 'red', is_open=False, is_locked=True)

        #self.switch = Switch(self.world)
        self.preassurePlateCoords = np.array([5, 4])

        agents: List[Agent] = []
        

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

        # Door
        # self.grid.set(6, 3, self.door)
        self.place_obj(self.door, top=(6, 3), size=(1, 1))
        
        # Switch
        # self.grid.set(5, 4, self.switch)
        #self.place_obj(self.switch, top=(5, 4), size=(1, 1))

        # Goal, reward defined here
        #self.place_obj(ObjectGoal(self.world, 0, 'ball'),top=(7,1), size=(3,5)) 

        # Ball
        # self.place_obj(Ball(self.world,0))
       
        # Randomize the player start position and orientation
        for a in self.agents:
            self.place_agent(a)


    # If the is on the switch, open the door
    def _handle_special_moves(self, i, rewards, fwd_pos, fwd_cell):
        agent = self.agents[i]
        print("agent pos:", agent.pos)
        if np.array_equal(self.preassurePlateCoords, agent.pos):
            self.door.is_open = True
        else: 
            self.door.is_open = False


    def step(self, actions):
        obs, rewards, done, info = MultiGridEnv.step(self, actions)
        return obs, rewards, done, info

class DoorGame1(DoorGameEnv):
    def __init__(self):
        super().__init__(
        width=11,
        height=7,
        agents_index = [1],
        view_size=7)

# d = DoorGame1()