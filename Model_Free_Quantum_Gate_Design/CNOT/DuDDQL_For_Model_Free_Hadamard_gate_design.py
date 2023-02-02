##===========================================================
"""
@author: Omar Shindi
Paper: "Model-free Quantum Gate Design and Calibration using Deep Reinforcement Learning"

Dueling Double DQL for model free quantum gate design 
CNOT - gate

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
This file mainly contains from three parts:
Part 1: DQL agent.
Part 2: Quantum system.
Part 3: Main parameters and Learning Procedure

##>>>>>>>>>>>>>    MIT License    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
Copyright (c) [2023] [Omar Shindi]
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
##=========================================================
# Import all the required packages
#==========================================================
import numpy as np
np.random.seed(0)
import tensorflow._api.v2.compat.v1 as tf
import matplotlib.pyplot as plt
import itertools
import math 
from scipy.linalg import expm

##==========================================================
##>>>>>>>>> Part 1 : DQL Agent <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
##==========================================================
tf.config.set_visible_devices([], 'GPU')
tf.disable_v2_behavior()
tf.set_random_seed(0)
class DuDDQL:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate,
            reward_decay,
            e_greedy,
            replace_target_iter,
            memory_size,
            batch_size,
            e_greedy_increment,
            size_hidden_layer,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.size_hidden_layer=size_hidden_layer
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features*2+2))
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            # Dueling DQN
            with tf.variable_scope('Value'):
                w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
                self.V = tf.matmul(l1, w2) + b2

            with tf.variable_scope('Advantage'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.A = tf.matmul(l1, w2) + b2

            with tf.variable_scope('Q'):
                out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)

            return out

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], self.size_hidden_layer, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)


    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1


    def choose_action(self, observation,x):
        observation = observation[np.newaxis, :]
        self.epsilon_max=x
        if np.random.uniform() < self.epsilon:  # choosing action
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)

        self.epsilon_increment=0.0001
        return action, self.epsilon
    
    
    def Epsilon_e(self):
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon <= self.epsilon_max else self.epsilon_max
        return self.epsilon
    

    def learn(self):

        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
        
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],    
                       self.s: batch_memory[:, -self.n_features:]})    
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        max_act4next = np.argmax(q_eval4next, axis=1)        # the action that brings the highest value is evaluated by q_eval
        selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})

        self.learn_step_counter += 1


##============================================================================
##>>>>>>>>>>>>>>> Part 2 : Quantum system  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#=============================================================================
# The output from this quantum systme will be used by RL after constructing the control protocol only to get final gate and to compute reward
class State():
    def __init__(self, N, T, Action_value):
        super(State, self)
        SPIN_NUM = 2
        actions_num=4
        self.D_Hilbert=2**SPIN_NUM # Dimension of Hilbert Space 
        State_num=self.D_Hilbert**2
        self.N= N
        self.DT=T/N
        self.action_space = np.array(list(itertools.product([-Action_value, Action_value], repeat = actions_num))) # repeate --> number of control field.
        
        self.n_actions = len(self.action_space)
        self.n_features = 2*State_num 
        
        
        self.sx=np.mat(([0,1],[1,0]))
        self.sy=np.mat(([0,-1j],[1j, 0]))
        self.sz=np.mat(([1, 0],[0,-1]))     
        
        
        self.SX1=np.kron(self.sx,np.identity(2))   
        self.SX2=np.kron(np.identity(2),self.sx)
        
        self.SY1=np.kron(self.sy,np.identity(2))   
        self.SY2=np.kron(np.identity(2),self.sy)

        self.SZ1=np.kron(self.sz,np.identity(2))   
        self.SZ2=np.kron(np.identity(2),self.sz)        
        
        self.H_T=[[1., 0., 0., 0.],
                  [0., 1., 0., 0.],
                  [0., 0., 0., 1.],
                  [0., 0., 1., 0.],]

             
    def reset(self):
        H0=np.identity(self.D_Hilbert)
        self.state =  np.array(H0)     
        self.counter=0
    
    def step(self, actionnum):
        actions = self.action_space[actionnum]
        H0=self.SZ1*self.SZ2        
        H1=actions[0]*self.SX1+actions[1]*self.SX2
        H2=actions[2]*self.SY1+actions[3]*self.SY2
        H=np.array(H0+H1+H2) 

        R=expm(-1j*H*self.DT)

        next_state =np.dot(R,self.state)
        fidelity =(np.abs((np.trace(np.dot(np.conjugate(np.transpose(self.H_T)),next_state))/self.D_Hilbert)))**2
        self.state = next_state # Store current Quantum Gate for the next iteration
        self.counter+=1
        return next_state, fidelity 

##===============================================================================================
##>>>>>>>>>>>>>>> Part 3 : Main parameters and Learning Procedure  <<<<<<<<<<<<<<<<<<<<<<<<<<
#================================================================================================
B_FP=[]#Best Fidelity + Number of Pulses 
T=1.1 # Evolution Time
N = 38 # Length of control protocol
DT=T/N # Time duration for each control pulse
C_max_value = 4  # Positive value of control pulse
env = State(N , T, C_max_value) # Create RL environment
actions_num=4 # number of control fields
action_space = np.array(list(itertools.product([-4, 4], repeat = actions_num))) # repeate --> number of control fields.

# Create RL agnet
RL= DuDDQL(env.n_actions, 5,
            learning_rate=0.0005,
            reward_decay=0.95, #gamma
            e_greedy=0.95,#1.0,#max_epsilon
            replace_target_iter=30,
            memory_size=25000,
            batch_size=64,
            e_greedy_increment=0.0001,
            size_hidden_layer=2*256
            )

# Extra parameters to save the resutls 
step = 0
fid_max = 0
FideList = []
STEPS=[]
Best_Exp=[]
 

plt.ion()
fig = plt.figure()

F_fid_max=0.99
e_greedy=0.99

Best_Exp_F=[]
Best_AAAA=0


#--------------------------------------------------------------
##>>>>>>>>>>>>>>> Learning Process  <<<<<<<<<<<<<<<<<<<<<<<<<<
if __name__ == '__main__':    
    for episode in range(500001): # Learning loop for 5000001 episodes
    
        M = []
        M_F=[]
        FF=[] # to store fidelity 

        #-----------------------------------------------------
        # 1st Stage : Construct control protocol
        #-----------------------------------------------------
        Constructed_Control_Protocol = []
        if  np.random.uniform()  < RL.Epsilon_e() and episode>2:
            AAAA=Best_AAAA
        else: 
            AAAA = np.random.randint(0, env.n_actions) 

        action =AAAA
        A_previous=action
        Obser_action=np.append(action_space[action]/(C_max_value*10),0)
        Constructed_Control_Protocol.append(action)
        for i in range(1,N):   
            action,e = RL.choose_action(Obser_action,e_greedy)
            A_previous=action # use the previous action not the following one.
            Obser_action_=np.append(action_space[action]/(C_max_value*10),i/N)
            M_F.append((Obser_action, action, 0, Obser_action_)) #  Next_observation, action, reward, Previous_observation
            Obser_action=Obser_action_
            Constructed_Control_Protocol.append(action)

            if(step%20==0 and episode>20 ):  
                    RL.learn()

            step += 1


        #--------------------------------------------------------------------------
        # 2nd Stage : Compute final quantum gate, then compute fidelity and reward
        #--------------------------------------------------------------------------
        # Apply constructed control protocol to build the quantum gate
        env.reset()
        for i in range(0,len(Constructed_Control_Protocol)):
                observation_L_gate, fidelity= env.step(Constructed_Control_Protocol[i])
                reward = -math.log(1-fidelity)#fidelity
                FF.append(fidelity)
                F_fid_max=fidelity
    
     
        #--------------------------------------------------------------------------
        # 3ed Stage : N-step reward process
        #--------------------------------------------------------------------------
        # N-step reward process --> Save state transition of an episode with the same reward.
        for ii in range(len(M_F)):           
            M_f=[]
            M_f=M_F[ii]           
            observation, action, rewardf, observation_=[M_f[jj] for jj in range(4)]
            RL.store_transition(observation, action, reward, observation_)
        

        #--------------------------------------------------------------------------
        # 4th Stage : Save best discovered results
        #--------------------------------------------------------------------------
        if fidelity >= fid_max:
            Best_AAAA=[]
            Best_Exp=[]
            Best_Exp_F=[]
            Best_AAAA=AAAA
            M_FF=[]
            M_FF=FF
            fid_max = fidelity
            Best_Exp=M   
            Best_Exp_F=M_F
            B_FP.append([fid_max,episode,step,i+1])
            print('Final_fidelity=', fid_max, 'Steps' , step,'episode', episode,'pulse num',i+1,"   First action:", AAAA )
          
        STEPS.append(episode)
        FideList.append(fidelity)
     
        # Update the value of epsilon "Exploration - Exploitation percentage"
        if (fid_max>=0.99):
            e_greedy=0.9999
        else:
            e_greedy=0.95


        #--------------------------------------------------------------------------------------------------------
        # 5th Stage : Modified experience memory -- Keep adding the best discovered restuls to experience buffer
        #--------------------------------------------------------------------------------------------------------
        if(episode % 3 == 0 and step>100):
           for ii in range(len(Best_Exp_F)):
               M_f=[]
               M_f=Best_Exp_F[ii]
               observation, action, rewardf, observation_=[M_f[jj] for jj in range(4)]
               RL.store_transition(observation, action, -math.log(1-fid_max), observation_) 
       

        #Plot the training progress
        if(episode % 10000 == 0):  
            plt.clf()
            plt.plot(FideList, '.', markersize=0.3)
            plt.title("Quantum gate design - CNOT -  ",fontsize=10)
            plt.xlabel("Episode",fontsize=10)
            plt.ylabel("Fidelity",fontsize=10)
            plt.show()   
            plt.clf()


        if(episode % 1 == 0 ):  
                RL.learn()


    

    
    
    
    
    
    
    
    
