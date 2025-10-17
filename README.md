

# **Project: Deep Q-Learning Lunar Lander**

## **Project Overview**

This project applies **Reinforcement Learning (RL)** to teach an artificial agent how to control a lunar lander in the OpenAI Gym *LunarLander-v2* environment. The agent observes the state of the lander, things like its coordinates, velocity, angle, and whether its legs are touching the ground and decides which thruster to fire next. Over time, by trial and error, it learns how to land softly between the flags without crashing or running out of fuel.

The approach is based on **Deep Q-Learning (DQN)**, which combines **Q-Learning**, a core reinforcement learning algorithm, with **deep neural networks**. Instead of storing a giant table of possible states and actions (which is infeasible for complex problems), the neural network estimates the *Q-values* ( expected future rewards) directly from the lander’s continuous state. Two networks are used:

* a **Q-Network** that learns from new experience, and
* a **Target Q-Network** that stabilizes learning by updating more slowly.

Through repeated simulation, the agent gradually figures out the optimal policy, how to act in each situation to maximize its long-term reward.

---

## **Why Reinforcement Learning Is So Fascinating**

Reinforcement learning becomes especially powerful when combined with deep learning, because it allows agents to operate in continuous, high-dimensional spaces where the number of possible states is practically infinite. Instead of relying on predefined rules, the agent learns directly from raw numerical input, using neural networks to approximate complex value functions and policies. Over time, it gradually reduces its epsilon, the factor controlling exploration vs. exploitation, meaning it starts by acting randomly and then becomes more deliberate as it gains experience. This process mirrors how infants learn about the world: lots of random trial-and-error at first, then increasingly precise actions guided by learned patterns. It’s an elegant, biologically inspired approach that opens the door to systems capable of mastering intricate real-world tasks autonomously.

Another reason I find this project so cool is that it reminded me of that viral TikTok clip showing a **SpaceX rocket booster landing itself**. Replicating a simplified version of that through code (teaching a virtual spacecraft to self-stabilize and land) felt both futuristic and cool.

---

## **What This Code Is Doing**

### Train the Agent**

* **1**: We initialize the `memory_buffer` with a capacity of $N =$ `MEMORY_SIZE`. Notice that we are using a `deque` as the data structure for our `memory_buffer`.

* **2**: We skip this line since we already initialized the `q_network`.

* **3**: We initialize the `target_q_network` by setting its weights to be equal to those of the `q_network`.

* **4**: We start the outer loop. Notice that we have set $M =$ `num_episodes = 2000`. This number is reasonable because the agent should be able to solve the Lunar Lander environment in less than `2000` episodes using this notebook's default parameters.

* **5**: We use the `.reset()` method to reset the environment to the initial state and get the initial state.

* **6**: We start the inner loop. Notice that we have set $T =$ `max_num_timesteps = 1000`. This means that the episode will automatically terminate if the episode hasn't terminated after `1000` time steps.

* **7**: The agent observes the current `state` and chooses an `action` using an $\epsilon$-greedy policy. Our agent starts out using a value of $\epsilon =$ `epsilon = 1` which yields an $\epsilon$-greedy policy that is equivalent to the equiprobable random policy. This means that at the beginning of our training, the agent is just going to take random actions regardless of the observed `state`. As training progresses we will decrease the value of $\epsilon$ slowly towards a minimum value using a given $\epsilon$-decay rate. We want this minimum value to be close to zero because a value of $\epsilon = 0$ will yield an $\epsilon$-greedy policy that is equivalent to the greedy policy. This means that towards the end of training, the agent will lean towards selecting the `action` that it believes (based on its past experiences) will maximize $Q(s,a)$. We will set the minimum $\epsilon$ value to be `0.01` and not exactly 0 because we always want to keep a little bit of exploration during training. If you want to know how this is implemented in code we encourage you to take a look at the `utils.get_action` function in the `utils` module.

* **8**: We use the `.step()` method to take the given `action` in the environment and get the `reward` and the `next_state`.

* **9**: We store the `experience(state, action, reward, next_state, done)` tuple in our `memory_buffer`. Notice that we also store the `done` variable so that we can keep track of when an episode terminates.

* **10**: We check if the conditions are met to perform a learning update. We do this by using our custom `utils.check_update_conditions` function. This function checks if $C =$ `NUM_STEPS_FOR_UPDATE = 4` time steps have occured and if our `memory_buffer` has enough experience tuples to fill a mini-batch. For example, if the mini-batch size is `64`, then our `memory_buffer` should have more than `64` experience tuples in order to pass the latter condition. If the conditions are met, then the `utils.check_update_conditions` function will return a value of `True`, otherwise it will return a value of `False`.

* **11 - 14**: If the `update` variable is `True` then we perform a learning update. The learning update consists of sampling a random mini-batch of experience tuples from our `memory_buffer`, setting the $y$ targets, performing gradient descent, and updating the weights of the networks. We will use the `agent_learn` function we defined in [Section 8](#8) to perform the latter 3.

* **15**: At the end of each iteration of the inner loop we set `next_state` as our new `state` so that the loop can start again from this new state. In addition, we check if the episode has reached a terminal state (i.e. we check if `done = True`). If a terminal state has been reached, then we break out of the inner loop.

* **16**: At the end of each iteration of the outer loop we update the value of $\epsilon$, and check if the environment has been solved. We consider that the environment has been solved if the agent receives an average of `200` points in the last `100` episodes. If the environment has not been solved we continue the outer loop and start a new episode.

Finally, we wanted to note that we have included some extra variables to keep track of the total number of points the agent received in each episode. This will help us determine if the agent has solved the environment and it will also allow us to see how our agent performed during training. We also use the `time` module to measure how long the training takes.


