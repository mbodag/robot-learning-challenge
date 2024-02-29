##########################
# YOU CAN EDIT THIS FILE #
##########################


# Imports from external libraries
import numpy as np

# Imports from this project
import constants
import configuration
from graphics import PathToDraw
import torch


PATHS_BEFORE_TRAINING = 100
LEARNING_RATE = 0.01
ALPHA = 1


class Network(torch.nn.Module):

    # The class initialisation function.
    def __init__(self):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 10 units.
        self.layer_1 = torch.nn.Linear(in_features=4, out_features=20, dtype=torch.float32)
        self.layer_2 = torch.nn.Linear(in_features=20, out_features=20, dtype=torch.float32)
        self.output_layer = torch.nn.Linear(in_features=20, out_features=2, dtype=torch.float32)

    # Function which sends some input data through the network and returns the network's output.
    def forward(self, input):
        layer_1_output = torch.nn.functional.sigmoid(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


class Robot:

    def __init__(self, goal_state):
        #The goal state
        self.goal_state = goal_state
        #Paths to draw in the visualisation
        self.paths_to_draw = []
        #Num demos asked
        self.demo_requests = 0
        #Num steps taken in current path
        self.step = 0
        
        #Reset flag
        self.reset = False
        #Number of past resets
        self.num_resets = 0
        #Flag for if a path was found during training
        self.found_path_during_training = False
        #Best path found during training
        self.best_path = None
        #Buffer size
        self.start_buffer_size = 0
        self.middle_buffer_size = 0
        self.goal_buffer_size = 0
        #Buffer with state, action, next_state
        self.replay_buffer_start = np.zeros([1, 3, 2])
        self.replay_buffer_middle = np.zeros([1, 3, 2])
        self.replay_buffer_goal = np.zeros([1, 3, 2])
        self.is_first_state = True
        self.first_state = [0,0]
        self.START_BUFFER_DISTANCE_THRESHOLD = 25
        #Dynamics neural network
        self.dynamics_model_network_start = Network()
        self.dynamics_model_network_middle = Network()
        self.dynamics_model_network_goal = Network()
        #Training action type
        self.train_action = 'semi-random'
        #Current path
        self.current_path = []
        #Randomness
        self.randomness = 70
        #Whether to explore near the goal if reached in training
        self.goal_currently_found = False
        self.explore_near_goal = True
        self.near_goal_explorations = 0
        self.MAX_NEAR_GOAL_EXPLORATIONS = 50
        #Cross entropy method parameters
        self.CEM_NUM_ITERATIONS = 10
        self.CEM_NUM_PATHS = 50
        self.CEM_PATH_LENGTH = 100
        self.CEM_NUM_ELITES = 5


    def get_next_action_type(self, state, money_remaining):
        # TODO: This informs robot-learning.py what type of operation to perform
        # It should return either 'demo', 'reset', or 'step'
        if self.is_first_state:
            self.first_state = state
            self.is_first_state = False
        if self.demo_requests < 0:
            self.demo_requests += 1
            print('Requesting a demonstration')
            return 'demo'
        if self.reset == True:
            self.step = 0
            self.reset = False
            self.current_path = []
            self.goal_currently_found = False
            self.near_goal_explorations = 0
            print('Resetting the environment')
            return 'reset'
        if True:
            return 'step'

    def get_next_action_training(self, state, money_remaining):
        # TODO: This returns an action to robot-learning.py, when get_next_action_type() returns 'step'
        # Currently just a random action is returned
        if self.train_action == 'trick':
            action = constants.ROBOT_MAX_ACTION*(state - self.goal_state) / max(np.abs(state-self.goal_state))
        elif self.train_action == 'perp': 
            #Find the closest visited state
            pass
            distances = np.linalg.norm(self.replay_buffer_middle[:-200, 0] - state, axis=1)
            arg_closest_state = np.argmin(distances)
            print(arg_closest_state, distances.shape[0])
            closest_state = self.replay_buffer_middle[arg_closest_state, 0]
            angle = np.arctan((closest_state[0] - state[0]) / (closest_state[1] - state[1] + 0.0001))
            angle += np.random.uniform(0.7, 1.3) * np.pi
            direction = np.array([np.cos(angle), np.sin(angle)])
            perp_action = constants.ROBOT_MAX_ACTION*direction / np.max(direction)
            #Pick an action that does not go towards that state
            trick_action = constants.ROBOT_MAX_ACTION*(state - self.goal_state) / max(np.abs(state-self.goal_state))
            action = 0.7 * trick_action + 0.3 * perp_action
        elif self.train_action == 'random':
            action = np.random.uniform(-constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION, 2)
        elif self.train_action == 'semi-random':
            randint = np.random.randint(0, 100)
            if randint > self.randomness:
                action = constants.ROBOT_MAX_ACTION*(state - self.goal_state) / max(np.abs(state-self.goal_state))
            else:
                action = np.random.uniform(-constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION, 2)
            
        elif self.train_action == 'cross_entropy':
            pass
        elif self.train_action == 'explore near goal':
            action = np.random.uniform(-constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION, 2)
        return action

    def get_next_action_testing(self, state):
        # TODO: This returns an action to robot-learning.py, when get_next_action_type() returns 'step'
        # Currently just a random action is returned
        self.cross_entropy_method_planning(state)
        action = self.planned_actions[0]
        return action

    # Function that processes a transition
    def process_transition(self, state, action, next_state, money_remaining):
        #Add transition to replay buffer

        if np.linalg.norm(state - self.first_state) < self.START_BUFFER_DISTANCE_THRESHOLD:
            self.replay_buffer_start=np.vstack((self.replay_buffer_start, np.array([[state, action, next_state]])))
            if self.start_buffer_size == 0:
                print('Removing initial start buffer row')
                self.replay_buffer_start = self.replay_buffer_start[1:]
            self.start_buffer_size += 1
        elif np.linalg.norm(state - self.goal_state) < 2 * constants.TEST_DISTANCE_THRESHOLD:
            self.replay_buffer_goal=np.vstack((self.replay_buffer_goal, np.array([[state, action, next_state]])))
            if self.goal_buffer_size == 0:
                print('Removing initial goal buffer row')
                self.replay_buffer_goal = self.replay_buffer_goal[1:]
            self.goal_buffer_size += 1
        else:
            self.replay_buffer_middle=np.vstack((self.replay_buffer_middle, np.array([[state, action, next_state]])))
            if self.middle_buffer_size == 0:
                print('Removing initial middle buffer row')
                self.replay_buffer_middle = self.replay_buffer_middle[1:]
            self.middle_buffer_size += 1
        
        #Calculate the 'reward' for the transition
        reward = np.sqrt(np.sum((next_state - state)**2)) / np.sqrt(np.sum((action)**2))
        
        #Show the transition in the visualisation
        self.paths_to_draw.append(PathToDraw(np.array([state, next_state]), (255*reward, 255*reward, 255*reward), 2))
        
        self.step += 1
        
        #Add to current_path
        self.current_path.append(np.array([next_state, action]))
        if np.linalg.norm(state - self.goal_state) < constants.TEST_DISTANCE_THRESHOLD:
            self.goal_currently_found = True
        
        #Reset if the goal is reached or the step limit is reached
        if self.goal_currently_found and self.explore_near_goal and self.near_goal_explorations < self.MAX_NEAR_GOAL_EXPLORATIONS:
            self.train_action = 'explore near goal'
            self.near_goal_explorations += 1
        elif self.step > 500 or self.goal_currently_found:
            print('MONEY REMAINING: ', money_remaining)
            self.randomness = self.randomness * 0.9
            self.reset = True 
            self.train_action = 'perp'
            print("REWARD:", self.compute_reward(self.current_path))
            # print('currentpath', self.current_path[:, 0])
        #Increment step counter

        #Train model if enough transitions have been collected
        if self.start_buffer_size == 50:
            print('Training model')
            self.train_model(20, 'start')
        elif self.start_buffer_size > 50:
            self.train_model(5, 'start')
        
        if self.middle_buffer_size == 50:
            print('Training model')
            self.train_model(20, 'middle')
        elif self.middle_buffer_size > 50:
            self.train_model(5, 'middle')
        
        if self.goal_buffer_size == 50:
            print('Training model')
            self.train_model(20, 'goal')
        elif self.goal_buffer_size > 50:
            self.train_model(5,'goal')
            
            
        
        

    # Function that takes in the list of states and actions for a demonstration
    def process_demonstration(self, demonstration_states, demonstration_actions, money_remaining):
        self.paths_to_draw.append(PathToDraw(demonstration_states, (255, 0, 0), 2))


    def dynamics_model(self, state, action):
        # TODO: This iates the learned dynamics model, which is currently called by graphics.py when visualising the model
        if np.linalg.norm(state - self.first_state) < self.START_BUFFER_DISTANCE_THRESHOLD:
            next_state = state + self.dynamics_model_network_start.forward(torch.tensor(np.concatenate((state, action), axis=0), dtype=torch.float32)).detach().numpy()
        elif np.linalg.norm(state - self.goal_state) < 2* constants.TEST_DISTANCE_THRESHOLD:
            next_state = state + self.dynamics_model_network_goal.forward(torch.tensor(np.concatenate((state, action), axis=0), dtype=torch.float32)).detach().numpy()
        else:
            next_state = state + self.dynamics_model_network_middle.forward(torch.tensor(np.concatenate((state, action), axis=0), dtype=torch.float32)).detach().numpy()
        
        return next_state

    def train_model(self, num_iterations, which_buffer):
        if which_buffer == 'start':
            buffer = self.replay_buffer_start
            buffer_size = self.start_buffer_size
            model = self.dynamics_model_network_start
        elif which_buffer == 'middle':
            buffer = self.replay_buffer_middle
            buffer_size = self.middle_buffer_size
            model = self.dynamics_model_network_middle
        elif which_buffer == 'goal':
            buffer = self.replay_buffer_goal
            buffer_size = self.goal_buffer_size
            model = self.dynamics_model_network_goal
        optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        losses = []
        iterations = []
        for training_iteration in range(num_iterations):
            minibatch_indices = np.random.choice(range(buffer_size), 5)
            minibatch_inputs = buffer[minibatch_indices, 0:2]
            minibatch_labels = buffer[minibatch_indices, 2] - buffer[minibatch_indices, 0]
            minibatch_inputs = np.reshape(minibatch_inputs, [5, 4])
            minibatch_labels = np.squeeze(minibatch_labels)

            # Convert the NumPy array into a Torch tensor
            minibatch_input_tensor = torch.tensor(minibatch_inputs, dtype=torch.float32)
            minibatch_labels_tensor = torch.tensor(minibatch_labels, dtype=torch.float32)
            # Do a forward pass of the network using the inputs batch
            network_prediction = model.forward(minibatch_input_tensor)
            # Compute the loss based on the label's batch
            loss = torch.nn.MSELoss()(network_prediction, minibatch_labels_tensor)
            # Compute the gradients based on this loss,
            # i.e. the gradients of the loss with respect to the network parameters.
            optimiser.zero_grad()
            loss.backward()
            # Take one gradient step to update the network
            optimiser.step()
            # Get the loss as a scalar value
            loss_value = loss.item()
            # Print out this loss
            #print('Iteration ' + str(training_iteration) + ', Loss = ' + str(loss_value))
        # Store this loss in the list
        losses.append(loss_value)
        # Update the list of iterations
        iterations.append(training_iteration)
        # Plot and save the loss vs iterations graph
        # ax.plot(iterations, losses, color='blue')
        # plt.yscale('log')
        # plt.show()
        # plt.pause(0.1)
        # # Optionally, save the figure
        # if 0:
        #     fig.savefig("loss_curve.png")

    def compute_reward(self, path):
        path = np.array(path)
        path_length = path.shape[0]
        speed_reward = np.linalg.norm(path[:-1,0] - path[1:, 0], axis=1) / np.linalg.norm(path[:-1, 1], axis = 1)
        distance_reward = -np.linalg.norm(path[-1, 0] - self.goal_state)
        goal_reached = int(np.linalg.norm(path[-1, 0] - self.goal_state) < constants.TEST_DISTANCE_THRESHOLD)
        
        total_reward = 20*(speed_reward / path_length) + ALPHA * distance_reward + 100*goal_reached
        return total_reward

    def compute_reward_one_step(self, state, action, next_state):
        speed_reward = np.linalg.norm(next_state - state) / np.linalg.norm(action, axis = 1)
        distance_reward = -np.linalg.norm(next_state - self.goal_state)
        goal_reached = int(np.linalg.norm(next_state[-1, 0] - self.goal_state) < constants.TEST_DISTANCE_THRESHOLD)
        
        total_reward = 10*(speed_reward) + 0.8 * distance_reward + 100*goal_reached
        return total_reward

    def cross_entropy_method_planning(self, robot_current_state):
        # planning_actions is the full set of actions that are sampled
        self.planning_actions = np.zeros([self.CEM_NUM_ITERATIONS, self.CEM_NUM_PATHS, self.CEM_PATH_LENGTH, 2], dtype=np.float32)
        # planning_paths is the full set of paths (one path is a sequence of states) that are evaluated
        self.planning_paths = np.zeros([self.CEM_NUM_ITERATIONS, self.CEM_NUM_PATHS, self.CEM_PATH_LENGTH, 2], dtype=np.float32)
        # planning_path_rewards is the full set of path rewards that are calculated
        self.planning_path_rewards = np.zeros([self.CEM_NUM_ITERATIONS, self.CEM_NUM_PATHS])
        # planning_mean_actions is the full set of mean action sequences that are calculated at the end of each iteration (one sequence per iteration)
        self.planning_mean_actions = np.zeros([self.CEM_NUM_ITERATIONS, self.CEM_PATH_LENGTH, 2], dtype=np.float32)
        self.planning_std_actions = np.zeros([self.CEM_NUM_ITERATIONS, self.CEM_PATH_LENGTH, 2], dtype=np.float32)
        # Loop over the iterations
        for iteration_num in range(self.CEM_NUM_ITERATIONS):
            # In each iteration, a new set of paths will be sampled
            for path_num in range(self.CEM_NUM_PATHS):
                # For each sampled path, compute a sequence of states by sampling actions from the current action distribution
                # To begin, set the state for the first step in the path to be the robot's actual current state
                planning_state = np.copy(robot_current_state)

                if iteration_num == 0:
                    for step_num in range(self.CEM_PATH_LENGTH):
                        action = np.random.uniform(-constants.ROBOT_MAX_ACTION, constants.ROBOT_MAX_ACTION, 2)
                        self.planning_actions[iteration_num, path_num, step_num] = action
                        # Compute the next state given this action
                        next_state = self.dynamics_model(planning_state, action)
                        self.planning_paths[iteration_num, path_num, step_num] = next_state
                        #Update the current state in this path
                        planning_state = next_state
                else:
                    for step_num in range(self.CEM_PATH_LENGTH):
                        action = np.random.normal(self.planning_mean_actions[iteration_num-1, step_num], self.planning_std_actions[iteration_num-1, step_num], 2)
                        self.planning_actions[iteration_num, path_num, step_num] = np.maximum(-5,np.minimum(5,action))
                        # Compute the next state given this action
                        next_state = self.dynamics_model(planning_state, action)
                        self.planning_paths[iteration_num, path_num, step_num] = next_state
                        #Update the current state in this path
                        planning_state = next_state
                # Calculate the reward for this path
                path = np.transpose(np.array([self.planning_paths[iteration_num, path_num], self.planning_actions[iteration_num, path_num]]),[1,0,2])
                path_reward = self.compute_reward(path)
                self.planning_path_rewards[iteration_num, path_num] = path_reward
            elites = np.argsort(self.planning_path_rewards)[iteration_num, -self.CEM_NUM_ELITES:]
            mean_elites = np.mean(self.planning_actions[iteration_num, elites], axis=0)
            std_elites = np.std(self.planning_actions[iteration_num, elites], axis=0)
            self.planning_mean_actions[iteration_num] = mean_elites
            self.planning_std_actions[iteration_num] = std_elites
        
        self.planned_path = np.zeros([self.CEM_PATH_LENGTH, 2])
        self.planned_actions = self.planning_mean_actions[-1]
        for step_num in range(self.CEM_PATH_LENGTH):
            action = self.planned_actions[step_num]
            next_state = self.dynamics_model(planning_state, action)
            self.planned_path[step_num] = next_state
            planning_state = next_state
        print(self.planned_path[0], robot_current_state )
        self.paths_to_draw.append(PathToDraw(self.planned_path, (255, 255, 255), 2))
        # return self.compute_reward(self.planned_path)
    