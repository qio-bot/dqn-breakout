# Mid-term Assignments Report

| **Member** |ID| **Ideas (%)** | **Coding (%)** | **Writing (%)** |
| ---------- | --|------------- | -------------- | --------------- |
| 青悦       | 19335169 |20%           | 0%             | 80%             |
| 欧阳蓓     | 19335161 |80%           | 100%           | 20%             |

## Requirements

1. Read through the implementation and explain in detail what each component is responsible for and how the components are connected together.

2. Pick one of the questions below to research into. 

   • Can you boost the training speed using Prioritized Experience Replay?

   • Can you improve the performance using Dueling DQN?

   • Can you stabilize the movement of the paddle (avoid high-freq paddle shaking effects) so that the agent plays more like a human player?

   Explain your work and exhibit the performance gain (or explain why things won’t work) .

## Our work in brief

We use Prioritized Experience Replay to boost the training speed. We've also done some research into Dueling DQN and we'll show both of them.

We've completed this report in English and open-source our project on [GitHub](https://github.com/qio-bot/dqn-breakout) as well.



## Original Implementation

During training we generate episodes using $\epsilon$-greedy policy with the current approximation of the action-value function $Q .$ The transition tuples $\left(s_{t}, a_{t}, r_{t}, s_{t+1}\right)$ encountered during training are stored in the replay buffer. The generation of new episodes is interleaved with neural network training. The network is trained using mini-batch gradient descent on the loss $\mathcal{L}$ which encourages the approximated Q-function to satisfy the Bellman equation: $\mathcal{L}=\mathbb{E}\left(Q\left(s_{t}, a_{t}\right)-y_{t}\right)^{2}$, where $y_{t}=r_{t}+\gamma \max _{a^{\prime} \in \mathcal{A}} Q\left(s_{t+1}, a^{\prime}\right)$ and the tuples $\left(s_{t}, a_{t}, r_{t}, s_{t+1}\right)$ are sampled from the replay buffer . 

### main function:

- While training, the agent interacts with the environment constantly, choosing action by $\epsilon-greedy$ according to observations.
- The last four frames will be taken as the observation, thus the agent learn once every four steps.
- The agent calls the learn function, samples from the memory, calculates the Q estimate and Q target, and updates the behavior network parameters.
- Update the target value network parameters at regular intervals. 

### `utils_drl.py`

This component implements the DQN `Agent` class.

#### Field Summary

+ *self*.__action_dim: dimensions of the action space.

+ *self*.__device. 

+ *self*.__gamma: the discount factor in the process of updating sample priorities using TD-error.

+ *self*.__eps_start: the initial value of $\epsilon$ in $\epsilon-greedy$.

+ self.__eps_final: the minimal value of $\epsilon$ in $\epsilon-greedy$.

+ self.__eps_decay: the decay factor. $\epsilon$ weighs exploitation and exploration. At the beginning of training, it should be dominated by exploration. In the later stage of learning, $\epsilon$ gradually decreases allowing it to exploit its knowledge. 

+ *self*.__policy: a Q network. Its input is the state and output is the value corresponding to each action.

+ *self*.__target: the target network. Its input is the state and output is the value corresponding to each action. Copy the parameters from the Q network above at regular intervals 

  **target network**
  Using the target network, DQN regards  predicting the q value as a regression problem. Regression problems require supervision signals. If the same network is used for both prediction and supervision, it will cause moving target. 
  In RL, when a non-linear function is used to approximate the Q value function, the update of the Q value is prone to fluctuation, showing unstable learning behavior. Therefore, the target network is introduced. 

#### Agent Functions

+ `def run(self, state: TensorStack4, training: bool = False)`
  The agent chooses an action for the given state.

  + If it is during training,  perform the $\epsilon$ decay process to gradually reduce the probabilities of exploration.

  + Then, we follow the strategy $\epsilon-greedy$.
    Randomly generate a 0-1number.

    + If it is less than $\epsilon$, choose an action randomly.
    + else using greedy policy, Input the state into the policy network, output the value of each action, and select the action with the largest value. 

    

+ `def learn(self, memory: ReplayMemory, batch_size: int) -> float:`

  + First sample the batch_size transition from memory. 

    ```python
    state_batch, action_batch, reward_batch, next_batch, done_batch = memory.sample(batch_size)
    ```

  + Calculate Q estimate using policy network

    ```python
    values = self.__policy(state_batch.float()).gather(1, action_batch)
    ```

  + Calculate Q target using the formula below.
    $Y_t^Q = R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \theta_t).$ 
    Calculate values_next using target network.

    ```python
    values_next = self.__target(next_batch.float()).max(1).values.detach()
    expected = (self.__gamma * values_next.unsqueeze(1)) * (1. - done_batch) + reward_batch        
    ```

  + Get trained neural network loss.

    ```python
    loss = F.smooth_l1_loss(values, expected)
    self.__optimizer.zero_grad()
    loss.backward()
    for param in self.__policy.parameters():
    param.grad.data.clamp_(-1, 1)
    self.__optimizer.step()
    ```

+ `def sync(self) -> None:`

  Update the parameters of the target network, copy directly from the policy network.

+ `def save( self , path: str)`

  Save the model parameters. 

###  `utils_model.py`

This component implements DQN network architecture.

Note: This implementation combines 4 observations into one observation input, and the output dimension is action_dim.

Network composition: Three convolutional layers and two fully connected layers.

The input to the network is a 84x84x4 tensor containing  the last four frames. 

The first convolution layer convolves the input with 32 filters of size 8 (stride 4), the second layer has 64 layers of size 4 (stride 2), the final convolution layer has 64 filters of size 3 (stride 1). This is followed by a fully-connected hidden layer of 512 units. All these layers are separated by Rectifier Linear Units (ReLu). Finally, a fully-connected linear layer projects to the output of the network,  the Q-values.

### `utils.memory`

This component implements the `ReplayMemory` class. It stores the data obtained from the system's exploration environment, and then randomly samples to update the parameters of the deep neural network. 

Since the training samples obtained by the interaction between the agent and the environment are not independent and identically distributed, DQN introduces an experience replay mechanism to solve this problem. Using a buffer that replays past experience information, the past experience and the current experience are mixed to reduce data relevance. Moreover, the experience replay also makes the samples reusable, thereby improving the efficiency of learning. 

#### Field Summary

+ self.__device.
+ self.__capacity: maximum storage capacity.
+ self.__size: current storage number.
+ self.__pos: last insert position.
+ self.__m_states: store states, states is 5X84X84, Where states[:4] is obs and states[1:] is next_obs.
+ self.__m_actions: store actions.
+ self.__m_rewards: store rewards.
+ self.__m_dones: store the state of game (whether it's done).

#### ReplayMemory Functions

+ `def push(self,folded_state: TensorStack5,action: int,reward: int, done: bool,  ) -> None:`

  Insert a transition collected by the agent into the position of self.__pos.

+ ` def sample(self, batch_size: int) -> Tuple[BatchState,  BatchAction, BatchReward, BatchNext,  BatchDone]`

  Randomly sample batch_size transitions from memory to reduce data relevance.

+ ` def __len__(self) -> int:`
  Return the size of the memory.



## Dueling DQN

### Dueling Network



Let's see the difference between the normal DQN and the Dueling DQN. We can alter the structure of the original DQN by separating the representation of state value and action advantages to improve the performance significantly.

The DQN neural network directly outputs the Q value of each action, and the Q value of each action of Dueling DQN is determined by the following formula. This is particularly useful when there're states that don't affect the environment whatever actions they take.

$𝐴^𝜋(𝑠,𝑎)=𝑄^𝜋(𝑠,𝑎)−𝑉^𝜋(𝑠).$

The value function $V$ measures the how good it is to be in a particular state $s$. The $Q$ function, however, measures the the value of choosing a particular  action when in this state. Now, using the definition of advantage, we  might be tempted to construct the aggregating module as follows:

$Q(s,a;\theta,\alpha,\beta)=V(s;\theta,\beta)+(A(s,a;\theta,\alpha)-\frac{1}{\mid\mathcal{A}\mid}\sum\limits_{a'}A(s,a’;\theta,\alpha))$,

where $𝜃$ denotes the parameters of the convolutional layers, while $𝛼$ and $𝛽$ are the parameters of the two streams of fully-connected layers.

​	<img src="https://pig-1307013046.cos.ap-nanjing.myqcloud.com/PigCo/image-20211115133103281.png" alt="image-20211115133103281" style="zoom: 50%;" />

## Prioritized Experience Replay

Using Prioritized replay, you will pay attention to such a small amount of samples that are worth learning 

The transition tuples are uniformly sampled from the experience pool, ignoring the importance of experience while the prioritized experience replay framework increases the probabilities of being sampled by the importance of experience that the more important the more often the experience will be used, so as to improve the learning efficiency.

We use TD-error, denoted as $\delta$,  to determine the importance of the sample. It refers to the difference between the current Q value and its target Q value in the timing difference. 

$\delta=r+\max _{a^{\prime}} \gamma Q_{\text {target }}\left(s^{\prime}, a^{\prime}\right)-Q(s, a)$.

In DQN, the goal of our training is to make $\delta$ expectations as small as possible. From this point of view, it makes sense for us to choose the training order based on $\mid\delta\mid$. For each sample in the Buffer, we calculate a probability, and sample according to this probability. 

We use SumTree for sampling to avoid sorting all samples for each sampling.

### SumTree

memory: size = capacity. Store capacity transitions( [s, a, r, s', d] ).
	![image-20211113161637087](https://pig-1307013046.cos.ap-nanjing.myqcloud.com/PigCo/image-20211113161637087.png)

SumTree structure: a binary tree with size = capacity * 2. Used to store priority only and the priority of each transition is stored in the leaf nodes. The value of the parent node is equal to the sum of the values of the two child nodes.  The tree is implemented with an array.

<img src="https://pig-1307013046.cos.ap-nanjing.myqcloud.com/PigCo/image-20211113160330325.png" alt="image-20211113160330325" style="zoom: 67%;" />



| parameter    | definition                                |
| ------------ | ----------------------------------------- |
| leaf_num     | number of leaf nodes                      |
| total_num    | number of total nodes                     |
| p_idx        | the index of parent node                  |
| L_idx, R_idx | the indices of left and right child nodes |

$$
Table1:parameters~of~SumTree
$$

Here are some theorems we could infer from the definition.

1. total_num = leaf_num*2-1. 

2. If a transition has a data_idx position in the data, its priority is calculated like this in the index in the tree: tree_idx = dataIdx +self.capacity -1.
3. p_idx = (L_idx-1)/2 = (R_idx-1)/2  (round down)
   and L_idx = p_idx*2+1, R_idx = L_idx+1. This means we can find both nodes if we know any one of the parent node or the child nodes.

### Implementation

+ init
  Initialize capacity, count, pos (point to the location of the data).

  ```python
  def __init__(self, capacity):
          self.capacity = capacity
          self.tree = np.zeros(2 * capacity - 1)
          # self.data = np.zeros(capacity, dtype=object)
          self.count = 0
          self.pos = 0
  ```

+ update

  Update the priority at tree_idx to p, and recursively update the parent node.

  ```python
  def update(self, tree_idx, p):
          change = p - self.tree[tree_idx]
          self.tree[tree_idx] = p
          self.update_father(tree_idx, change)
  ```

+ update_father

  Recursively update the priority of the parent node. When the priority of a child node changes, the parent node, as the sum of its children, also needs to be updated. Because there is only one child node change, only the change value of this child node needs to be propagated, without dealing with its sibling.

  ```python
  def update_father(self, tree_idx, change):
          parent = (tree_idx - 1) // 2
          self.tree[parent] += change
          if parent != 0:
              self.update_father(parent, change)
  ```

+ add

  Insert priority in the tree_idx position corresponding to data_idx, and update the priority of the parent node. 

  ```python
  def add(self, p):
          idx = self.pos + self.capacity - 1
          self.update(idx, p)
          self.pos += 1
          if self.pos >= self.capacity:
              self.pos = 0
          if self.count < self.capacity:
              self.count += 1
  ```

+ Sample a copy of data 

  + Take node 0 as the parent node and traverse its child nodes 

  + If the left child node is greater than s, take the left node as the parent node and go through its child nodes 

  + else subtract the value of the left child node from s, select the right child node as the parent node, and traverse its child nodes 

  + Until the leaf node is traversed, the value of the leaf node is the priority, and the subscript corresponds to the value subscript. The corresponding value can be found from the transition 

  ```python
  def retrieve(self, tree_idx, s):
          left = 2 * tree_idx + 1
          right = 2 * tree_idx + 2
          if left >= len(self.tree):
              return tree_idx
          if s <= self.tree[left]:
              return self.retrieve(left, s)
          else:
              return self.retrieve(right, s - self.tree[left])
  ```

  ```python
  def get(self, s):
          tree_idx = self._retrieve(0, s)
          data_Idx = tree_idx - self.capacity + 1
          return  tree_idx,  data_Idx
  ```

- sample batch data 

  We divide the total of p by the batch size and divide it into several intervals. Then randomly select a number in each interval to get tree_idx and data_idx 

```python
def sample(self, batch_size: int) :
        tree_idxs = []
        indices= []
        segment = self .total() / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, data_idx = self.tree.get(s)
            tree_idxs.append(idx)
            indices.append(data_idx)
        return tree_idxs,indices
```



###  Replay Memory

On the basis of the original, add SumTree member variables and record priority.

+ init
  a is in [0,1]. It converts the importance of TD error to priority.

  e is a small amount to avoid zero priority.

  ```python
  self.prio_max = 0.1
  self.a = 0.6    
  self.e = 0.01
  self.tree = SumTree(capacity)
  ```

+ push

  + Add priority to tree.
  + Calculate initial p. 
  + Call function add in SumTree.

  ```python
  p = (np.abs(self.prio_max) + self.e) ** self.a #  proportional priority
  self.tree.add(p)
  ```

+ sample 

  + Call function sample in SumTree to get the tree_idx and data_idx of the batch_size size. 

  + Convert data_idx to tensor, take out transitions. 

  + Return tree_idx and transitions

    ```python
    def sample(self, batch_size: int)  :
            tree_idx ,indices = self.tree.sample(batch_size )
            indices =torch.tensor(indices).to(self.__device)
            #sample
            b_state = self.__m_states[indices, :4].to(self.__device).float()
            b_next = self.__m_states[indices, 1:].to(self.__device).float()
            b_action = self.__m_actions[indices].to(self.__device)
            b_reward = self.__m_rewards[indices].to(self.__device).float()
            b_done = self.__m_dones[indices].to(self.__device).float()
            return tree_idx, b_state, b_action, b_reward, b_next, b_done
    ```

+ update

  + Update the priority in the tree.

  + For each tree_idx, update its stored priority with TD-error. Using the formula:

    $ error\rightarrow priority:~priority= (\mid error\mid+~e)^{a}$

    ```python
    def update(self, idxs, errors):
            self.prio_max = max(self.prio_max, max(np.abs(errors)))
            for i, idx in enumerate(idxs):
                p = (np.abs(errors[i]) + self.e) ** self.a
                self.tree.update(idx, p) 
    ```

+ agent learn

  + Add tree_idx to the return value of the sampling function 

    ```python
    tree_idx, state_batch, action_batch, reward_batch, next_batch, done_batch = \
    	memory.sample(batch_size)
    ```

  + Calculate errors, and then update the priority of tree_idx in SumTree 

    ```python
    # upadte priority in Sumtree
    errors =( values-expected) .detach().cpu().squeeze().tolist()
    memory.update(tree_idx, errors)
    ```



## Additional Work

Use DDQN, since DQN is consistently and sometimes vastly over optimistic about the value of the current greedy policy and the over estimations are harming the quality of the resulting. Double DQN improves over DQN both in terms of value accuracy and in terms of policy quality. And it's simple to modify the algorithm.

DQN: $Y_{t}^{\mathrm{DQN}} \equiv R_{t+1}+\gamma \max _{a} Q\left(S_{t+1}, a ; \boldsymbol{\theta}_{t}^{-}\right)$

DDQN: $ Y_{t}^{\text {Double } Q} \equiv R_{t+1}+\gamma Q\left(S_{t+1}, \underset{a}{\operatorname{argmax}} Q\left(S_{t+1}, a ; \boldsymbol{\theta}_{t}\right) ; \boldsymbol{\theta}_{t}^{\prime}\right)$

The idea of Double Q-learning is to reduce overestimations by decomposing the max operation in the target into action selection and action evaluation.

```python
#double　DQN
next_actions =  self.__policy(next_batch.float()).argmax(dim=1, keepdim=True)
values_next = self.__target(next_batch.float()).gather(1, next_actions)
```



## Result & Analysis

### Hyperparameters

In all experiments, the discount was set to γ = 0.99, and the learning rate to α = 0.0000625. The number of steps between target network updates was τ = 10, 000. The memory gets sampled to update the network every 4 steps with minibatches of size 32. The simple exploration policy used is an ϵ-greedy policy with the ϵ decreasing linearly from 1 to 0.1 over 1M steps. The agent is evaluated every 10,000 steps. 

```
STACK_SIZE = 4
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 1000000
BATCH_SIZE = 32
POLICY_UPDATE = 4
TARGET_UPDATE = 10_000
WARM_STEPS = 50_000
MAX_STEPS = 10_000_000
EVALUATE_FREQ = 100_00
```



### Result

We run a total of 10 million steps and take data points every 10,000 steps. We have plotted the average rewards of every ten data of DDQN, Dueling DQN, PER comparing to DQN respectively. And output the maximum value and maximum average value of reward.

# ###这里放图

|         | max reward | max average reward |
| ------- | ---------- | ------------------ |
| DQN     | 302.7      | 171.44             |
| DUL     | 370.7      | 193.1              |
| PER     | 695.3      | 283.09             |
| PER+DUL | 636.7      | 387.07             |
| DDQN    | 290.3      | 147.77             |

$$
Table2:summary~of~scores
$$

#### plot function:

```python
def plot_average_reward (filename_1, filename_2, color1, color2  ):
    plt.figure(figsize=(30,15))
    step = []
    reward = []
    with open(filename_1,"r") as f:
        line = f.readlines()
    for i in line:
        l = i.split()
        if l==[]:
            break
        if(int(l[1]) % 100000==0   ):
            step.append(int(l[1])  // 100000  )
        reward.append(float(l[2]))
    label = filename_1 .split(".")[0]
    print(label + " max:")
    print(max(reward))
    #每十个计算均值
    avg = [sum(reward[i*10:(i+1)*10])/10 for i in range(len(reward)//10)] 
    plt.plot(step, avg, color1, label =label  )
    print(label+ " max_avg")
    print(max(avg))
    step = []
    reward = []
    with open(filename_2,"r") as f:
        line = f.readlines()
    for i in line:
        l = i.split()
        if l==[]:
            break
        if(int(l[1]) % 100000==0   ):
            step.append(int(l[1])  // 100000  )
        reward.append(float(l[2]))
    label = filename_2 .split(".")[0]
    print(label + " max:")
    print(max(reward))
    avg = [sum(reward[i*10:(i+1)*10])/10 for i in range(len(reward)//10)]
    print(label+ " max_avg")
    print(max(avg))
    plt.plot(step, avg, color2, label =  label  )
    plt.ylabel("AverageRewards", fontsize=20)
    plt.xlabel("Steps(10^5)", fontsize=20)
    plt.ylim(0,300)
    plt.legend(fontsize="large")
    plt.savefig(label + '.png')
```



### Analysis

+ PER and PER+DUL get 600 points by chance, and the other highest points are 403.0 and 494.7 respectively. Watch the generated video and find that the game has gone through multiple rounds.

+ Causes of instability late in the training:
  epsilon min =0.1which is a bit large.

  The reasons for the poor promotion effect in some places is that the output interval between per and DDQN is too large for some time.

  The experiment is contingent since only ran once.

+ Poor Performance of DDQN 
  The original paper uses 6 different random seeds with the same hyper-parameters.
  Because computing power and time are prioritized, only one experiment is performed, which is more contingent.



## Problems & Solutions

+ torch.cat and torch.stack
  When converting a states list into a tensor of [32,4,84,84], `torch.cat(states,dim=0)` is used at the beginning, and the result is [128,1,84,84]. And then `torch.stack(states, dim =0) ` is used.

+ Data type conversion 
  numpy.array & torch.tensor
  Tensor can be stored on the GPU (or 'cuda' device) or on the CPU (to be precise, on the memory of the CPU); while the Array can only be stored on the memory of the CPU. Unified conversion in the calling interface, so as to avoid confusion.

  .item() converts the value of the tensor to a standard Python value. It can only be used when the tensor contains only one element. When using it, it is unnecessary to consider whether the tensor is on the CPU or GPU, nor whether the tensor is with gradient. Generally use this command to convert loss into a value. It can also be done with .detach().cpu().numpy(). 

+ dimension problem because of broadcasting

  ```python
  >>> x.size()
  torch.Size([32])
  >>> y = torch.arange(32).view(32,1)
  >>> y.size()
  torch.Size([32, 1])
  >>> z = x*y
  >>> z.size()
  torch.Size([32, 32])
  ```

  Then, [32]*[32,1] will be [32,32].
  solution:

  ```python
  >>> x = x.view(32,1)
  >>> z = x*y
  >>> z.size()
  torch.Size([32, 1])
  ```

  or

  ```python
  >>> x = torch.arange(32)
  >>> x = x.unsqueeze(1)
  >>> z=x*y
  >>> z.size()
  torch.Size([32, 1])
  ```

  

## Reference

https://yulizi123.github.io/tutorials/machine-learning/reinforcement-learning/4-6-prioritized-replay/

https://zhuanlan.zhihu.com/p/47578210

https://shmuma.medium.com/speeding-up-dqn-on-pytorch-solving-pong-in-30-minutes-81a1bd2dff55

https://arxiv.org/pdf/1511.05952.pdf

https://arxiv.org/pdf/1511.06581.pdf

https://arxiv.org/pdf/1509.06461.pdf

