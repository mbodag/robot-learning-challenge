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
import copy


LEARNING_RATE = 0.01

class ResidualActorNetwork(torch.nn.Module):

    # The class initialisation function.
    def __init__(self):
        # Call the initialisation function of the parent class.
        super(ResidualActorNetwork, self).__init__()
        
        self.layer_1 = torch.nn.Linear(in_features=2, out_features=20, dtype=torch.float32)
        self.layer_2 = torch.nn.Linear(in_features=20, out_features=20, dtype=torch.float32)
        self.output_layer = torch.nn.Linear(in_features=20, out_features=2, dtype=torch.float32)


    # Function which sends some input data through the network and returns the network's output.
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output

class ResidualCriticNetwork(torch.nn.Module):

    # The class initialisation function.
    def __init__(self):
        # Call the initialisation function of the parent class.
        super(ResidualCriticNetwork, self).__init__()
        
        self.layer_1 = torch.nn.Linear(in_features=4, out_features=20, dtype=torch.float32)
        self.layer_2 = torch.nn.Linear(in_features=20, out_features=20, dtype=torch.float32)
        self.output_layer = torch.nn.Linear(in_features=20, out_features=1, dtype=torch.float32)


    # Function which sends some input data through the network and returns the network's output.
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output
    
class BehaviouralCloningNetwork(torch.nn.Module):

    # The class initialisation function.
    def __init__(self):
        # Call the initialisation function of the parent class.
        super(BehaviouralCloningNetwork, self).__init__()
        
        self.layer_1 = torch.nn.Linear(in_features=2, out_features=20, dtype=torch.float32)
        self.layer_2 = torch.nn.Linear(in_features=20, out_features=20, dtype=torch.float32)
        self.output_layer = torch.nn.Linear(in_features=20, out_features=2, dtype=torch.float32)

    # Function which sends some input data through the network and returns the network's output.
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
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
        #Buffer with state, action, next_state
        self.behavioural_cloning_buffer = np.zeros([1, 2, 2])
        self.behavioural_cloning_buffer_size = 0
        self.residual_buffer = np.zeros([1, 4, 2])
        self.residual_buffer_size = 0
        self.behavioural_buffer_size_at_start_of_episode = 0
        
        self.is_first_state = True
        self.first_state = [0,0]
        #Dynamics neural network
        self.behavioural_cloning_network = BehaviouralCloningNetwork()
        self.main_actor_network = ResidualActorNetwork()
        self.main_critic_network = ResidualCriticNetwork()
        self.target_actor_network = copy.deepcopy(self.main_actor_network)
        self.target_critic_network = copy.deepcopy(self.main_critic_network)
        self.total_residual_training_iterations = 0
        #Training action type
        self.train_action = 'trick'
        #Current path
        self.current_path = []
        #Randomness
        self.randomness = 70
        #Whether to explore near the goal if reached in training
        self.goal_currently_found = False
        self.explore_near_goal = True
        self.near_goal_explorations = 0
        self.MAX_NEAR_GOAL_EXPLORATIONS = 50
        self.first_demonstration_process = False
        self.demonstration_process = False
        self.residual_trick = False
        self.test_step = 0
        

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
        if self.step == 0:
            self.behavioural_buffer_size_at_start_of_episode = self.behavioural_cloning_buffer_size
        if self.train_action == 'trick':
            action = constants.ROBOT_MAX_ACTION*(state - self.goal_state) / max(np.abs(state-self.goal_state))
        elif self.train_action == 'residual':
            baseline_action = self.behavioural_cloning_network.forward(torch.tensor(state, dtype=torch.float32)).detach().numpy()
            residual_action = self.main_actor_network.forward(torch.tensor(state, dtype=torch.float32))[0].detach().numpy()
            residual_action = np.clip(residual_action, -5.0, 5.0)  
            #std = 5*(np.linalg.norm(state - self.goal_state) /  np.linalg.norm(self.first_state - self.goal_state))
            residual_action_with_noise = residual_action + np.random.normal(0, 5, 2)
            residual_action_with_noise = np.clip(residual_action_with_noise, -5.0, 5.0)  
            action = 0.8 * baseline_action + 0.2 * residual_action_with_noise
            action = constants.ROBOT_MAX_ACTION* action / np.linalg.norm(action)
            self.residual_action = residual_action_with_noise
        return action

    def get_next_action_testing(self, state):
        # TODO: This returns an action to robot-learning.py, when get_next_action_type() returns 'step'
        # Currently just a random action is returned

        baseline_action = self.behavioural_cloning_network.forward(torch.tensor(state, dtype=torch.float32)).detach().numpy()
        residual_action = self.main_actor_network.forward(torch.tensor(state, dtype=torch.float32))[0].detach().numpy()
        action = 0.8 * baseline_action + 0.2 * residual_action
        action = constants.ROBOT_MAX_ACTION* action / np.linalg.norm(action)
        self.test_step += 1
        if np.linalg.norm(state - self.goal_state) < 10:
            action = action = constants.ROBOT_MAX_ACTION*(state - self.goal_state) / max(np.abs(state-self.goal_state))
        return action

    # Function that processes a transition
    def process_transition(self, state, action, next_state, money_remaining):
        reward = self.compute_reward_one_step(state, action, next_state)
        double_reward = np.array([reward, reward])
        
        #Add to replay buffer
        if self.train_action == 'trick':
            self.behavioural_cloning_buffer = np.vstack((self.behavioural_cloning_buffer, np.array([[state, action]])))
            if self.behavioural_cloning_buffer_size == 0:
                print('Removing initial behavioural cloning buffer row')
                self.behavioural_cloning_buffer = self.behavioural_cloning_buffer[1:]
            if self.behavioural_buffer_size_at_start_of_episode > 0: 
                distances = np.linalg.norm(self.behavioural_cloning_buffer[:self.behavioural_buffer_size_at_start_of_episode, 0] - state, axis=1)
                closest_bc = np.min(distances)
                if closest_bc < 1:
                    self.train_action = 'residual'
            self.behavioural_cloning_buffer_size += 1

            
        elif self.train_action == 'residual':
            distances = np.linalg.norm(self.behavioural_cloning_buffer[:self.behavioural_buffer_size_at_start_of_episode, 0] - state, axis=1)
            closest_bc = np.min(distances)
            if closest_bc > 15 and self.step > 50:
                self.train_action = 'trick'
            self.residual_buffer = np.vstack((self.residual_buffer, np.array([[state, self.residual_action, next_state, double_reward]])))
            if self.residual_buffer_size == 0:
                print('Removing initial residual buffer row')
                self.residual_buffer = self.residual_buffer[1:]
            self.residual_buffer_size += 1
            if self.residual_buffer_size > 50 and self.step % 100 == 0:
                self.train_residual_learning(200)
        

        #Show the transition in the visualisation
        speed_reward = np.linalg.norm(next_state - state) / np.linalg.norm(action)
        self.paths_to_draw.append(PathToDraw(np.array([state, next_state]), (255*speed_reward, 255*speed_reward, 255*speed_reward), 2))
        
        self.step += 1
        
        #Add to current_path
        self.current_path.append(np.array([next_state, action]))
        if np.linalg.norm(state - self.goal_state) < constants.TEST_DISTANCE_THRESHOLD:
            self.goal_currently_found = True
        
        if self.train_action == 'trick':
            if self.step > 500 or self.goal_currently_found:
                if self.goal_currently_found:
                    if not self.found_path_during_training:
                        self.best_path = self.current_path
                    else:
                        if len(self.current_path) < len(self.best_path):
                            self.best_path = self.current_path
                    self.found_path_during_training = True
                self.reset = True 
                self.train_action = 'residual'
                self.demonstration_process = True
                self.first_demonstration_process = True
                if self.first_demonstration_process:
                    self.train_behavioural_cloning(700)
                    self.first_demonstration_process = False

        elif self.train_action == 'residual':
            if self.step > 200 or self.goal_currently_found:
                print('MONEY REMAINING: ', money_remaining)
                self.reset = True 
                if self.demonstration_process:
                    self.train_behavioural_cloning(50)

            
                

    # Function that takes in the list of states and actions for a demonstration
    def process_demonstration(self, demonstration_states, demonstration_actions, money_remaining):
        self.paths_to_draw.append(PathToDraw(demonstration_states, (255, 0, 0), 2))
        for i in range(len(demonstration_states)):
            self.behavioural_cloning_buffer = np.vstack((self.behavioural_cloning_buffer, np.array([[demonstration_states[i], demonstration_actions[i]]])))
            if self.replay_buffer_start.shape[0] == 1:
                print('Removing initial replay buffer row')
                self.replay_buffer_start = self.replay_buffer_start[1:]
            self.start_buffer_size += 1
        


    def dynamics_model(self, state, action):
        # TODO: This iates the learned dynamics model, which is currently called by graphics.py when visualising the model
        next_state = state + action
        return next_state

    def compute_reward_one_step(self, state, action, next_state):
        speed_reward = np.linalg.norm(next_state - state) / np.linalg.norm(action)
        distance_reward = -np.linalg.norm(next_state - self.goal_state)
        goal_reached = int(np.linalg.norm(next_state - self.goal_state) < constants.TEST_DISTANCE_THRESHOLD)
        
        total_reward = (speed_reward) + 0.2 * distance_reward + 10*goal_reached
        return total_reward
    
    def train_behavioural_cloning(self, num_iterations):
        print('Training behavioural cloning')
        buffer = self.behavioural_cloning_buffer
        buffer_size = self.behavioural_cloning_buffer_size
        model = self.behavioural_cloning_network
        optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        losses = []
        iterations = []
        for training_iteration in range(num_iterations):
            minibatch_indices = np.random.choice(range(self.behavioural_cloning_buffer_size), 5)
            minibatch_inputs = buffer[minibatch_indices, 0]
            minibatch_labels = buffer[minibatch_indices, 1]
            minibatch_inputs = np.squeeze(minibatch_inputs)
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

    
    def train_residual_learning(self, num_iterations):
        BATCH_SIZE = 20
        buffer = self.residual_buffer
        buffer_size = self.residual_buffer_size
        main_actor_network = self.main_actor_network
        main_critic_network = self.main_critic_network
        critic_optimiser = torch.optim.Adam(main_critic_network.parameters(), lr=0.00001, weight_decay=1e-2)
        actor_optimiser = torch.optim.Adam(main_actor_network.parameters(), lr=0.00001, weight_decay=1e-2)
        for training_iteration in range(num_iterations):
            self.main_critic_network.zero_grad()
            self.main_actor_network.zero_grad()
            minibatch_indices = np.random.choice(range(buffer_size), BATCH_SIZE)
            minibatch_states = torch.tensor(buffer[minibatch_indices, 0], dtype=torch.float32)
            minibatch_actions = torch.tensor(buffer[minibatch_indices, 1], dtype=torch.float32)
            minibatch_next_states = torch.tensor(buffer[minibatch_indices, 2], dtype=torch.float32)
            minibatch_rewards = torch.tensor(buffer[minibatch_indices, 3, 0], dtype=torch.float32).reshape((BATCH_SIZE, 1))
            
            # Do a forward pass of the network using the inputs batch
            next_actions = self.target_actor_network.forward(minibatch_next_states)
            next_Q_values = self.target_critic_network.forward(torch.cat((minibatch_next_states, next_actions), dim=1)).detach()
            expected_Q_values = minibatch_rewards + 0.8 * next_Q_values
            
            current_Q_values = self.main_critic_network.forward(torch.cat((minibatch_states, minibatch_actions), dim=1))
            #print('current Q values: ', current_Q_values, 'expected Q values: ', expected_Q_values, 'next Q values', next_Q_values)
            self.main_critic_network.zero_grad()
            critic_loss = torch.nn.MSELoss()(current_Q_values, expected_Q_values)
            critic_loss.backward()
            critic_optimiser.step()
            
            self.main_actor_network.zero_grad()
            
            predicted_actions = self.main_actor_network(minibatch_states)
            actor_Q_values = self.main_critic_network.forward(torch.cat((minibatch_states, predicted_actions), dim=1))
            actor_loss = -torch.mean(actor_Q_values)
            actor_loss.backward()
            actor_optimiser.step()
            if critic_loss > 1000 or actor_loss > 1000:
                self.copy_networks(backwards = True)
            self.total_residual_training_iterations += 1
            if self.total_residual_training_iterations % 400 == 0:
                self.copy_networks()
    
    def copy_networks(self, backwards = False):
        with torch.no_grad():
            for target_param, main_param in zip(self.target_actor_network.parameters(), self.main_actor_network.parameters()):
                if backwards: #Tries to prevent divergence
                    main_param.data.copy_(target_param.data)
                else:
                    target_param.data.copy_(main_param.data)
        with torch.no_grad():   
            for target_param, main_param in zip(self.target_critic_network.parameters(), self.main_critic_network.parameters()):
                if backwards: #Tries to prevent divergence
                    main_param.data.copy_(target_param.data)
                else:
                    target_param.data.copy_(main_param.data)
                    
