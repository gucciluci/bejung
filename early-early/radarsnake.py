import torch
import numpy as np
from collections import deque
from copy import deepcopy

'''
priority = [0,1]
x_velocity = [1,x_vel_max]
angular_velocity = [-angular_vel_max,angular_vel_max]
'''




gameparams = {'r_angle': 1.10,
			  'r_radius': 1,
			  'frequency': 1,
			  'tstep': 1/60,
			  'xval_max': 5,
			  'angvel_max': 10}



#work in progress... ignore this class for now
class PriorityNet(torch.nn.Module):

	def __init__(self, xvel_max, angvel_max, actions_convolved=3):
		self.actions_convolved = actions_convolved
		self.fc1 = torch.nn.Linear(2,16)
		self.fc2 = torch.nn.Linear(16,8)
		self.fc3 = torch.nn.Linear(8,3)
		self.relu = torch.nn.ReLU()

		self.xvel_sigmoid = lambda x: torch.sigmoid(x) * (xvel_max-1) + 1
		self.angvel_sigmoid = lambda x: torch.tanh(x) * angvel_max

		self.dqn1 = torch.nn.Linear(actions_convolved*3,6)
		self.dqn2 = torch.nn.Linear(6,2)

		#x, y are states, adding velocity later, too much complicated math right now

	def forward(self, x):
		#returns 3 action values: priority, angular velocity, x velocity
		x = self.fc1(x)
		x = self.relu(self.fc2(x))
		x = self.fc3(x)

		priority = torch.sigmoid(x[:,0])
		xvel = self.xvel_sigmoid(x[:,1])
		angvel = self.angvel_sigmoid(x[:,2])

		priority_list = torch.hstack(priority, angvel, xvel)
		#sort by priority

		actions = self.relu(self.dqn1(priority_list[:self.actions_convolved].view(self.actions_convolved*3)))
		return actions



class DQN:

	#experience tuple must be of the form (object_pos_0, a_vector, object_pos_1, r, done)
	#since this net crashes at closest dot, only consider closest dot in valid r

	#Should i consider priority list, or closest object, or what? Keep priority list in the net that it thinks
	#or what...... thoughts to consider here.

	#if it fails, don't feed top priority object, feed the failure object so that it is punished
	#properly

	#if it failed on the nth priority, retroactively change rewards to top n priorities to -100
	#to puniish the network for mishandling priorities correctly

	def __init__(self, xvel_max, angvel_max, mem_len=1000000, alpha=None, epsilon=None):

		self.memory = mem_len
		self.alpha = 0.9 if not alpha else alpha
		self.epsilon = 1.0 if not epsilon else epsilon
		self.mem_idx = 0
		self.net = PriorityNet(xvel_max, angvel_max)
		self.t_net = deepcopy(self.net)
		self.mem_len = deque(maxlen=mem_len)

		#look into torch deque?


	def q_update(self, batchsize=100):

		#pump multiple experiences through in separate batches
		batch_indices = np.random.randint(0,min(self.mem_len,self.mem_idx),batch_size)

		with torch.no_grad():

            target = r_stack + ~done_stack * torch.tensor(self.gamma) * torch.max(target_net(s1_stack),1)[0]

        optimizer.zero_grad()
        chosen_action = net(state_stack).gather(1,action_stack.view(-1,1)).squeeze()
        self.loss_val = loss(chosen_action, target)
        if trans_net:
            self.loss_val += loss(chosen_action,trans_target)
        self.loss_val.backward()



        optimizer.step()

	def choose_action(self, state):
		#######how to implement e greedy in continuous action space? #########


		#######sort items by first column of tensor

		# should agent return one action or average of actions
		return priority_list[0,1:]









#takes form (x,y)
def convert_to_polar(coord):
	return (np.sqrt(coord[0]**2 + coord[1]**2), np.arctan(coord[1]/coord[0]))

#takes form (r,angle)
def convert_to_xy(coord):
	return (coord[0]*np.sin(coord[1]),coord[0]*np.cos(coord[1]))


class ObjectDodgerGame():


	#######FIX GAMEPARAMS TO TAKE DICTIOINARY AS INPUT########
	def __init__(self, gameparams):
		#frequence is defined as avg number of objects within 1 meter
		#objects stored as (r,angle)
		# r_ stands for receptive

		#r_angle is 60 degrees
		self.r_angle = 1.10

		#not used yet, supposed to be how far ahead you can see things
		self.r_radius = 1

		#frequency objects appear, standard is 1/60 chance to see object times velocity
		self.frequency = 1
		self.tstep = 1/60
		self.objects = []
		self.objradius = self.tstep
		self.turn_radius = self.r_angle*self.tstep

		#5 times the standard t_step for x
		#10 times the turnradius for angle (~10 degrees for 1/60th tstep, and r_angle of 1.10)
		self.xvel_max = 5
		self.angvel_max = 10


	def step(self,action):

		#action is of form (ang_vel, x_vel)
		#angular velocity must be from 1 to 10, x vel from 1 to 5

		#this is what moves each object to the left or right of your vision
		#by the angular velocity
		for i, obstacle in enumerate(self.objects):
			new_polar =  convert_to_polar((convert_to_xy(obstacle)[0]-action[1]*self.tstep,
													  convert_to_xy(obstacle)[1]))
			new_polar = ((new_polar[0],new_polar[1]-action[0]*self.turn_radius))

			#returns done if you crashed
			if new_polar[0] < self.tstep:
				r = -100
				return True, r
			else:
				self.objects[i] = new_polar

		# the faster you go, you are more likely to encounter random objects in your future
		#consider this a warning
		for chance_to_spawn in np.round(action[1]).astype(np.int):
			if np.random.random() < self.tstep*self.frequency:
				self.objects.append((np.random.uniform(self.r_radius-action[1]*self.tstep,self.r_radius),
										            np.random.uniform(-self.r_angle/2,self.r_angle/2)))
		r = action[1] - self.xvel_max - 1

		return False, r


def play():
	done = False
	game = ObjectDodgerGame()
	learner = DQN(xvel_max=5, angvel_max=10, alpha=0.9, gamma=0.7)
	while not done:
		s0 = game.objects
		learner.choose_action(s0)
		done, r = game.step(action)
		if not obstacles:
			continue
		s1 = game.objects
		learner.memory[learner.mem_idx % learner.mem_len] = (s0, action, r, s1, done)




