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
        self.agents_coords = env_config["agents_coords"]
        self.ball_coord = env_config["ball_coord"]
        self.view_size = env_config["view_size"]
        self.width = env_config["width"]
        self.height = env_config["height"]
        self.partial_obs = env_config["partial_obs"]
        self.num_agents = len(self.agents_index)
        self.max_steps = env_config["max_steps"]
        self.easy_reward = env_config.get('easy_reward', False)
        self.pressure_plates_pressed_reward = False
        self.ball_picked_up_reward = False
        self.version = env_config["version"]
        self.see_through_walls = env_config.get('see_through_walls', False)
        self.decaying_reward = env_config.get('decaying_reward', False)
        self.door_coords = env_config.get('door_coords', (6, 3))
        # Metrics, number of completed tasks (preassure plate, ball pickup) by each agent per episode
        self.completed_task_by_agent = [[0,0],[0,0]]

        if self.easy_reward:
            self.pressure_plates_pressed_reward = True
            self.ball_picked_up_reward = True

        for i in range(len(self.agents_index)):
            agent = Agent(self.world, self.agents_index[i], view_size=self.view_size)
            agent.dir = 0
            agent.pos = self.agents_coords[i]
            agent.init_pos = self.agents_coords[i]
            self.agents.append(agent)

        for pos in env_config["pressure_plates_coords"]:
            self.pressure_plates.append(PreassurePlate(self.world, pos[0], pos[1]))

        super().__init__(
            max_steps=self.max_steps,
            width=self.width,
            height=self.height,
            agents=self.agents,
            agent_view_size=self.view_size,
            partial_obs=self.partial_obs, 
            door=self.door,
            door_pos=self.door_coords,
            see_through_walls=self.see_through_walls,
            preassureplates=self.pressure_plates,
        )


    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        # Door walls
        self.grid.vert_wall(self.world, self.door_coords[0], 0, 3)
        self.grid.vert_wall(self.world, self.door_coords[0], height-3, 3)

        # Door
        self.door.is_open = False
        self.grid.set(self.door_coords[0], self.door_coords[1], self.door)

        # Fill with walls around the goal for version >1_0
        if float(self.version.replace("_", ".")) >= 1.0:
            self.grid.vert_wall(self.world, self.door_coords[0]+1, 0, 3)
            self.grid.vert_wall(self.world, self.door_coords[0]+1, height-3, 3)
        
        # Preassure plates
        for pressure_plate in self.pressure_plates:
            pressure_plate.occupied = None
            self.grid.set(pressure_plate.pos[0], pressure_plate.pos[1], pressure_plate.switch)

        # Goals
        #self.place_obj(ObjectGoal(self.world, 0, 'ball'),top=(4,4), size=(1,1)) 
        for reward_tile_coord in self.reward_tile_coords:
            self.grid.set(reward_tile_coord[0], reward_tile_coord[1], ObjectGoal(self.world, 0, 'ball', reward=1))

        # Ball
        self.grid.set(self.ball_coord[0], self.ball_coord[1], Ball(self.world,0))

        # Agents
        for i in range(len(self.agents)):
            self.grid.set(self.agents_coords[i][0], self.agents_coords[i][1], self.agents[i])
            self.agents[i].pos = self.agents_coords[i]
            self.agents[i].dir = 0

        # Reset metrics
        self.completed_task_by_agent = [[0,0],[0,0]]


    # If the agent is on the switch, open the door
    def _handle_special_moves(self, i, rewards, fwd_pos, fwd_cell):
        agent = self.agents[i]
        for pressure_plate in self.pressure_plates:
            if np.array_equal(pressure_plate.pos, agent.pos):
                self.door.is_open = True
                pressure_plate.occupied = i
                # Keep track of completed tasks
                self.completed_task_by_agent[i][0] += 1
                # In easy mode, give a reward for reaching the pressure plate
                if self.pressure_plates_pressed_reward == True:
                    self._reward(i, rewards, reward=0.5)
                    self.pressure_plates_pressed_reward = False
            else:
                if pressure_plate.occupied == i:
                    self.door.is_open = False
                    pressure_plate.occupied = None
        

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                if self.agents[i].carrying is None:
                    self.agents[i].carrying = fwd_cell
                    self.agents[i].carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
                    # Keep track of completed tasks
                    self.completed_task_by_agent[i][1] += 1
                    # In easy mode, give a reward for picking up the ball
                    if self.ball_picked_up_reward == True:
                        self._reward(i, rewards, reward=0.5)
                        self.ball_picked_up_reward = False
    

    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        if self.agents[i].carrying:
            if fwd_cell:
                # Drop ball on goal
                if fwd_cell.type == 'objgoal' and fwd_cell.target_type == self.agents[i].carrying.type:
                    if self.agents[i].carrying.index in [0, fwd_cell.index]:
                        self._reward(fwd_cell.index, rewards, reward=1)
                        self.agents[i].carrying = None
                        return True
                # Give ball to other agent
                elif fwd_cell.type=='agent':
                    if fwd_cell.carrying is None:
                        fwd_cell.carrying = self.agents[i].carrying
                        self.agents[i].carrying = None
            else:
                self.grid.set(*fwd_pos, self.agents[i].carrying)
                self.agents[i].carrying.cur_pos = fwd_pos
                self.agents[i].carrying = None
        return False


    def _reward(self, i, rewards,reward=1):
        # Shared reward for all agents, divide by number of agents to keep reward in range [0, 1]
        for j,a in enumerate(self.agents):
                if self.decaying_reward:
                    rewards[j] = (1 - 0.9 * (self.step_count / self.max_steps))/self.num_agents
                else:
                    rewards[j]+=reward/2
                #print(f"Agent {j} got reward {reward}")


    def step(self, actions):
        obs, rewards, done, truncated, info = MultiGridEnv.step(self, actions)
        # Custom metrics
        info['completed_task_by_agent'] = self.completed_task_by_agent
        return obs, rewards, done, truncated, info


class PreassurePlate():
    def __init__(self, world, x, y):
        self.world = world
        self.pos = np.array([x, y])
        # Render as switch
        self.switch = Switch(self.world)
        self.occupied = None