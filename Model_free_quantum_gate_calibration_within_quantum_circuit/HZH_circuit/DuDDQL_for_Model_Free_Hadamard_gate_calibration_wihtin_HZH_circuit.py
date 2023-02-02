##===========================================================
"""
@author: Omar Shindi


Paper: "Model-free Quantum Gate Design and Calibration using Deep Reinforcement Learning"
Note: To understand this code please see Figure 3, page 4 of the same Paper.

Dueling Double DQL for model free quantum gate calibration within quantum circuit

--> Model free calibration of Hadamard gates within BitFilp quantum circuit

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
This file mainly contains from three parts:
Part 1: Dueling Double DQL Agent
Part 2: Quantum system that we want to calibrate to Hadamard gate
Part 3: Test calibrated quantum gate within real quantum circuit
Part 4 : Desierd quantum operator
Part 5 : Main parameters and Learning Procedure
Part 6 : Generate training states (Input State, Target State)
Part 7 : Testing procedure for the best calibrated Hadamard gates within real quantum circuit

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
from S0_Generator import S0_G
from S0_Generator_Testing import S0_G_t
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math 
np.random.seed(0)
import tensorflow._api.v2.compat.v1 as tf
from scipy.linalg import expm

##==========================================================
##>>>>>>>>> Part 1 : Dueling Double DQL Agent <<<<<<<<<<<<<<
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

##=============================================================================================
##>>>>>>  Part 2 : Quantum system that we want to calibrate to Hadamard gate  <<<<<<<<
# Again, this system is unknown to RL agent, this part only to the calibrated operator
#=============================================================================================
# In following paper: "Model-free Quantum Gate Design and Calibration using Deep Reinforcement Learning"
# This quantum system described by Hamiltonian equation (16) page number 6.
# The output quantum operator from this part will be used as Hadamard gate ...
#..into the quantum circuit described in Figure 12, Page 9. 

class State(object):
    def __init__(self):
        
        super(State, self)
        self.SPIN_NUM=1
        self.D_Hilbert=2**self.SPIN_NUM # Dimension of Hilbert Space
        self.State_num=self.D_Hilbert**2
        self.N = 38 
        self.DT=1.1/self.N	
        self.actions_num=2
        self.action_space = list(itertools.product([-4, 4], repeat =self.actions_num)) # repeate --> number of control field.
        self.n_actions = len(self.action_space)
        self.n_features = 2*self.State_num
        
        
        self.sx=np.mat(([0,1],[1,0]))
        self.sy=np.mat(([0,-1j],[1j, 0]))
        self.sz=np.mat(([1, 0],[0,-1]))     
        
    def reset(self):        
        H0=np.identity(self.D_Hilbert)
        self.state =  np.array(H0)     
        self.counter=0


    def step(self, actionnum):       
        actions = self.action_space[actionnum]
        H0=actions[0]*self.sz        
        H1=actions[1]*self.sx
        H=np.array(H0+H1) 
        R=expm(-1j*H*self.DT)        
        next_state =np.dot(R,self.state)    
        self.state = next_state # Store current Quantum Gate for the next iteration
        self.counter+=1
        return next_state

##=================================================================================
##>>>>>>>> Part 3 : Testing circuit for the calibrated gate to Hadamard gate  <<<<<
##  in other words, Testing calibrated quantum gate within real quantum circuit
#==================================================================================
# In the following paper: "Model-free Quantum Gate Design and Calibration using Deep Reinforcement Learning"
# Real quantum circuit as described in Figure 3 page 4.
# This part is to test the calibrated gate in Part 2 of this code within real quantum circuit.
class HC1_circuit(object):
    def __init__(self):
        super(HC1_circuit, self)               
        self.sx=np.mat(([0,1],[1,0]))
        self.sy=np.mat(([0,-1j],[1j, 0]))
        self.sz=np.mat(([1, 0],[0,-1]))     
                         
    def reset(self, H, S0, ST):
        self.H=H      
        self.state0 = S0
        self.stateT = ST

    def step(self):
        H=np.dot(np.dot(self.H,self.sz),self.H)
        Final_state=np.matrix( np.dot(self.state0, H))  
        g=np.matrix(self.stateT)
        fidelity1 = np.abs((np.dot(Final_state,g.getH()))**2)

        fidelity=fidelity1.item()

        return fidelity

##===========================================================================
##>>>>>>>> Part 4 : Desierd quantum circuit  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#============================================================================
# In the following paper: "Model-free Quantum Gate Design and Calibration using Deep Reinforcement Learning"
# Desired quantum operator as described in Figure 3 page 4.
# This Part is using to generate the Target quantum states that will be used for training ...
# .. and testing calibrated quantum gate within quantum circuit in Part 3 of this code.

class HC1_circuit1 (object):
    def __init__(self):
        super(HC1_circuit1, self)               
        self.sx=np.mat(([0,1],[1,0]))
        self.sy=np.mat(([0,-1j],[1j, 0]))
        self.sz=np.mat(([1, 0],[0,-1])) 
        self.H_T=[[(1/np.sqrt(2)), (1/np.sqrt(2))],
                [(1/np.sqrt(2)), -(1/np.sqrt(2))],]  
                       
    def reset(self, S0):
        self.state0 = S0

    def step(self):
           
        H=np.dot(np.dot(self.H_T,self.sz),self.H_T)
        Target_state= np.dot(self.state0, H)        

        return Target_state

##===============================================================================================
##>>>>>>>>>>>>>>> Part 5 : Main parameters and Learning Procedure  <<<<<<<<<<<<<<<<<<<<<<<<<<
#================================================================================================

B_FP=[]#Best Fidelity + Number of Pulses 
env = State() # quautm system --> This quantum system is unknown to RL agnet
env_test = HC1_circuit() # To test the calibrated quantum gate within real quantum circuit
env_Desires=HC1_circuit1() # To get the desired Target state.

# Create RL agnet
RL= DuDDQL(env.n_actions, 3,
            learning_rate=0.0005,
            reward_decay=0.95, #gamma
            e_greedy=0.95,#1.0,#max_epsilon
            replace_target_iter=30,
            memory_size=25000,
            batch_size=128,
            e_greedy_increment=0.0001,
            size_hidden_layer=2*256
            )

 

step = 0
fid_max = 0
FideList = []
STEPS=[]
Best_Exp=[]

plt.ion()
fig = plt.figure()


f_fidelity=0
F_fid_max=0.99
e_greedy=0.99
N=38

SPIN_NUM = 2
actions_num= 2 # number of control fields
D_Hilbert=2**SPIN_NUM # Dimension of Hilbert Space 
State_num=D_Hilbert**2

DT=1.1/N


action_space = np.array(list(itertools.product([-4, 4], repeat = actions_num))) # repeate --> number of control field.




Best_Exp_F=[]
Best_AAAA=0

##===============================================================================================
##>>>>>>>>>>>>>>> Part 6 : Generate training states (Input State, Target State)<<<<<<<<<<<<<<<<<<
#================================================================================================

S0_Gen=S0_G() # To generate the initial state
S0_Gen_test=S0_G_t() # to generate the initial states for testing

 
S0=np.array((1,0))
ST=np.array((0,1))

S0_train=[]
ST_train=[]
S_train = []
S0=np.array((1,0))

for tt in range(100):                    
    S0_Gen_test.reset((S0)) #Generate next initial state 
    S0,cc=S0_Gen_test.step()
    env_Desires.reset(S0) # Initial State 
    ST=env_Desires.step() # Desiered Target State
    S0_train.append(S0)
    ST_train.append(ST)
    S_train.append((S0, ST))

S_Train = np.array(S_train)
S0_Train = np.array(S0_train)
ST_Train = np.array(ST_train)




#--------------------------------------------------------------
##>>>>>>>>>>>>>>> Learning Process  <<<<<<<<<<<<<<<<<<<<<<<<<<

if __name__ == '__main__':    
    for episode in range(250000):
    #-------------------------------------------------------------------------------
    # 1st Stage : Construct control protocol to calibrate "Real quantum operator"
    #--------------------------------------------------------------------------------
        Constructed_Control_Protocol = []
        M_F=[]                            
        if  np.random.uniform()  < RL.Epsilon_e() and episode>2:
            AAAA=Best_AAAA
        else: 
            AAAA = np.random.randint(0, env.n_actions) 

        action = AAAA
        A_previous=action
        Obser_action=np.append(action_space[action]/40,0)
        Constructed_Control_Protocol.append(action)

        for i in range(1,N):   
            action,e = RL.choose_action(Obser_action,e_greedy)
            Obser_action_=np.append(action_space[action]/40,i/38)
            M_F.append((Obser_action, action, 0, Obser_action_))
            Obser_action=Obser_action_
            Constructed_Control_Protocol.append(action)

            if(step%20==0 and episode>20 ):  
                    RL.learn()
    
            step += 1

        #--------------------------------------------------------------------------------------------
        # 2nd Stage : Calibrate quantum gates, then place them into real quantum circuit for testing
        #--------------------------------------------------------------------------------------------
        # Apply constructed control protocol from 1st Stage to calibrate quantum gate 
        env.reset()
        for action_i in Constructed_Control_Protocol:
                observation_L_gate = env.step(action_i)


        #--------------------------------------------------------------------------------------------
        # 3ed Stage : Test the calibrated quantum gate from 2nd stage within real quantum circuit
        #--------------------------------------------------------------------------------------------
        # As explained in Figure 3, page 4, this is the feedback to RL agent to be used for computing the reward
        FFid=[]
        for tt in range(100):                    
            env_test.reset(observation_L_gate,S0_train[tt],ST_train[tt])  # Reset the approximate gate
            fide = env_test.step()     # Find the fidelity for the approximate gate 
            FFid.append(fide)
                
        fidelity=min(FFid)
        reward=-math.log(1-fidelity)
 

        #--------------------------------------------------------------------------
        # 4th Stage : N-step reward process
        #--------------------------------------------------------------------------
        # N-step reward process --> Save state transition of an episode with the same reward.
        for ii in range(len(M_F)):           
            M_f=[]
            M_f=M_F[ii]           
            observation, action, rewardf, observation_=[M_f[jj] for jj in range(4)]
            RL.store_transition(observation, action, reward, observation_)


        #--------------------------------------------------------------------------
        # 5th Stage : Save best discovered results
        #--------------------------------------------------------------------------   
        if fidelity >= fid_max:
            Best_AAAA=[]
            Best_Exp=[]
            Best_Exp_F=[]
            Best_AAAA=AAAA
            fid_max = fidelity
            Best_Exp_F=M_F
            Best_Gate=observation_L_gate
            print(observation_L_gate)

            B_FP.append([fid_max,episode,step,i+1])
            print('Final_fidelity=', fid_max, 'Steps' , step,'episode', episode,'pulse num',i+1,"   First action:", AAAA )
    
    
        STEPS.append(episode)
        FideList.append(fidelity)

        # Update the value of epsilon "Exploration - Exploitation percentage" 
        if (fid_max>=0.99 ):
            e_greedy=0.9999
        elif (fid_max>=0.999):
            e_greedy=0.99999
        else:
            e_greedy=0.95

    

        #--------------------------------------------------------------------------------------------------------
        # 6th Stage : Modified experience memory -- Keep adding the best discovered restuls to experience buffer
        #--------------------------------------------------------------------------------------------------------
        if(episode % 3 == 0 and step>100):
           for ii in range(len(Best_Exp_F)):
               M_f=[]
               M_f=Best_Exp_F[ii]
               observation, action, rewardf, observation_=[M_f[jj] for jj in range(4)]
               RL.store_transition(observation, action, -math.log(1-fid_max), observation_) 

        # Train RL agent
        if(episode % 1 == 0 ):  
                RL.learn()
 


    #Plot the training progress
    plt.clf()
    plt.plot(FideList, '.', markersize=0.3)
    plt.title("Calibration progress for Hadamard gates within HZH quantum circuit",fontsize=10)
    plt.xlabel("Episode",fontsize=10)
    plt.ylabel("Fidelity",fontsize=10)
    plt.show()   
    plt.clf()

    
##===============================================================================================
##>>>>>>>>>>>>>>> Part 7 : Testing procedure for the best calibrated Hadamard gates within real ...
#                              ... quantum circuit.
# The results from this part described in figure 14, page 9.
#================================================================================================

SS0_0=[]
SS0_1=[]
SST_0=[]
SST_1=[]
SSF_0=[]
SSF_1=[]

FFid=[]
CC=[]
for tt in range(50000):                    

    S0_Gen_test.reset((S0)) #Generate next initial state 
    S0,cc=S0_Gen_test.step()
    env_Desires.reset(S0) # Initial State 
    ST=env_Desires.step() # Desiered Target State
    env_test.reset(Best_Gate,S0,ST)  # Reset the approximate gate
    reward,fide,Sf = env_test.step()     # Find the fidelity for the approximate gate 
    FFid.append(fide)
    
    
    CC.append(cc)


fidelity=min(FFid)
reward=-math.log(1-fidelity)


    
    
    
    
