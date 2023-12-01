from gym_multigrid.multigrid import *
from typing import List
from ray.rllib.env.env_context import EnvContext

class DoorGameEnv(MultiGridEnv):
    def __init__(
        self,
        env_config: EnvContext,
    ):
        self.world = World
        self.door = Door(self.world, 'red', is_open=False, is_locked=True)
        self.agents: List[Agent] = []
        self.pressure_plates: List[PreassurePlate] = []
        self.reward_tile_coords = env_config["reward_tile_coords"]
        self.agents_index = env_config["agents_index"]
        self.view_size = env_config["view_size"]
        self.width = env_config["width"]
        self.height = env_config["height"]
        self.partial_obs = env_config["partial_obs"]
        self.num_agents = len(self.agents_index)

        for i in self.agents_index:
            self.agents.append(Agent(self.world, i, view_size=self.view_size))

        for pos in env_config["pressure_plates_coords"]:
            self.pressure_plates.append(PreassurePlate(self.world, pos[0], pos[1]))

        super().__init__(
            max_steps= 1000,
            width=self.width,
            height=self.height,
            agents=self.agents,
            agent_view_size=self.view_size,
            partial_obs=self.partial_obs, 
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
        self.grid.set(6, 3, self.door)
        
        # Preassure plates
        for pressure_plate in self.pressure_plates:
            self.grid.set(pressure_plate.pos[0], pressure_plate.pos[1], pressure_plate.switch)

        # Goals
        #self.place_obj(ObjectGoal(self.world, 0, 'ball'),top=(4,4), size=(1,1)) 
        for reward_tile_coord in self.reward_tile_coords:
            self.grid.set(reward_tile_coord[0], reward_tile_coord[1], ObjectGoal(self.world, 0, 'ball', reward=1))

        # Ball
        self.grid.set(5,3,Ball(self.world,0))

        # Randomize the player start position and orientation
        for a in self.agents:
            self.place_agent(a)


    # If the agent is on the switch, open the door
    def _handle_special_moves(self, i, rewards, fwd_pos, fwd_cell):
        agent = self.agents[i]
        for pressure_plate in self.pressure_plates:
            if np.array_equal(pressure_plate.pos, agent.pos):
                self.door.is_open = True
            else: 
                self.door.is_open = False


    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                if self.agents[i].carrying is None:
                    self.agents[i].carrying = fwd_cell
                    self.agents[i].carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
    

    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        if self.agents[i].carrying:
            if fwd_cell:
                # Drop ball on goal
                if fwd_cell.type == 'objgoal' and fwd_cell.target_type == self.agents[i].carrying.type:
                    if self.agents[i].carrying.index in [0, fwd_cell.index]:
                        self._reward(fwd_cell.index, rewards, fwd_cell.reward)
                        self.agents[i].carrying = None
                # Give ball to other agent
                elif fwd_cell.type=='agent':
                    if fwd_cell.carrying is None:
                        fwd_cell.carrying = self.agents[i].carrying
                        self.agents[i].carrying = None
            else:
                self.grid.set(*fwd_pos, self.agents[i].carrying)
                self.agents[i].carrying.cur_pos = fwd_pos
                self.agents[i].carrying = None


    def _reward(self, i, rewards,reward=1):
        # Shared reward for all agents
        for j,a in enumerate(self.agents):
                rewards[j]+=reward
                print(f"Agent {j} got reward {reward}")


    def step(self, actions):
        obs, rewards, done, truncated, info = MultiGridEnv.step(self, actions)
        return obs, rewards, done, truncated, info


class PreassurePlate():
    def __init__(self, world, x, y):
        self.world = world
        self.pos = np.array([x, y])
        # Render as switch
        self.switch = Switch(self.world)