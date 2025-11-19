# RL Learning Resources - Based on Our Journey

This document contains curated learning resources based on the specific challenges and concepts we encountered during your LunarLander training.

---

## Core RL Concepts We Covered

### 1. **Actor-Critic Methods (A2C)**

**What you learned:**
- Actor outputs policy (action probabilities)
- Critic outputs value estimate V(s)
- Advantages = actual_return - V(s) reduce variance
- Training them jointly is tricky (stability issues)

**Deep dive resources:**
- [Spinning Up: Actor-Critic](https://spinningup.openai.com/en/latest/algorithms/vpg.html#the-advantage-function)
- Paper: "Asynchronous Methods for Deep RL" (A3C paper, foundation for A2C)
- [Lil'Log: Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

**Why it matters:** A2C is the foundation. Understanding it deeply helps with PPO, SAC, and other advanced methods.

### 2. **Advantages vs Returns**

**Key insight you gained:**
- Returns = total future reward (can all be positive/negative)
- Advantages = relative measure (better/worse than expected)
- Advantages provide better learning signal by reducing variance

**Practice exercise:**
```python
# Given an episode:
rewards = [1, 2, 3, 4, 5]
values = [10, 8, 6, 4, 2]  # Critic predictions
gamma = 0.99

# Calculate returns (work backwards):
# G4 = 5
# G3 = 4 + 0.99*5 = 8.95
# G2 = 3 + 0.99*8.95 = 11.86
# ...

# Calculate advantages:
# A3 = G3 - V(s3) = 8.95 - 4 = 4.95  (better than expected!)
# A2 = G2 - V(s2) = 11.86 - 6 = 5.86
```

**Resource:**
- [Advantage Function Explained (visual)](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f)
- Sutton & Barto, Chapter 13 (free online)

### 3. **Value Function Divergence**

**Your specific issue: Pessimistic drift**
```
Episode 1: V(s) = -0.2
Episode 100: V(s) = -5
Episode 200: V(s) = -84
Episode 300+: V(s) = -100 to -120 (stuck!)
```

**Why this happens:**
1. Bad episodes → negative returns
2. Critic learns pessimistic values
3. Negative advantages → policy degrades
4. Worse policy → more bad episodes
5. Positive feedback loop!

**How others solve it:**
- **PPO**: Clips policy updates to prevent large changes
- **SAC**: Uses entropy regularization + target networks
- **TD3**: Twin critics + delayed policy updates
- **Value normalization**: Track running statistics

**Papers:**
- "Addressing Function Approximation Error in Actor-Critic Methods" (TD3)
- "Soft Actor-Critic" (entropy regularization prevents collapse)

### 4. **Reward Clipping vs Scaling**

**What we tried:**
```python
REWARD_CLIP = 10.0  → Unstable (long episodes = huge returns)
REWARD_CLIP = 1.0   → Too weak (can't tell success from failure)
REWARD_CLIP = 3.0   → Still diverged eventually
```

**The tension:**
- Too much clipping → weak learning signal
- Too little clipping → unstable training
- This is a fundamental problem in RL!

**Industry solutions:**
- **Reward shaping**: Engineer better intermediate rewards
- **Return normalization**: Normalize returns to mean=0, std=1
- **Value function normalization**: PopArt algorithm
- **Distributional RL**: Predict distribution of returns, not just mean

**Papers:**
- "Learning values across many orders of magnitude" (PopArt)
- "A Distributional Perspective on RL" (C51 algorithm)

---

## Visual RL Challenges

### 5. **CNN Inductive Bias**

**Your insight: "CNNs are biased toward local patterns, but RL needs global relationships"**

This is profound! You discovered a key limitation.

**Why CNNs struggle:**
```
Question: "How far am I from the landing pad?"
CNN: Needs 5-7 layers to build receptive field covering both
Transformer: One attention layer directly connects them
```

**State of the art solutions:**
1. **Vision Transformers (ViT)** - What we implemented
2. **Perceiver** - Handles arbitrary inputs with cross-attention
3. **Hybrid architectures** - CNN stem + Transformer (what you have!)
4. **Slot attention** - Learns object-centric representations

**Papers:**
- "An Image is Worth 16x16 Words" (ViT)
- "Perceiver: General Perception with Iterative Attention"
- "Object-Centric Learning with Slot Attention"

### 6. **Temporal Modeling**

**The problem:** 8 stacked frames as channels doesn't explicitly model time.

**Solutions we discussed:**
- **3D Convolutions** (what you implemented!)
- **Recurrent networks** (LSTM/GRU)
- **Temporal transformers** (attention across time)

**Why 3D conv is smart:**
```python
# 2D conv: Can't distinguish these
Frame 1: [0, 0, 1, 0, 0]  "Object at position 2"
Frame 2: [0, 0, 1, 0, 0]  "Object still at position 2" (static)

vs

Frame 1: [0, 0, 1, 0, 0]  "Object at position 2"
Frame 2: [0, 0, 0, 1, 0]  "Object moved to position 3" (moving right!)

# 3D conv: Explicitly detects motion with temporal kernel
```

**Papers:**
- "Quo Vadis, Action Recognition? A New Model and Kinetics Dataset" (I3D - 3D convolutions)
- "Video Action Transformer Network" (temporal transformers)

### 7. **Sample Efficiency in Visual RL**

**Your experience:** 1800 episodes and still not converged!

This is normal for pixel-based RL. It's painfully sample-inefficient.

**Why pixels are hard:**
- High dimensional input (56,448 dims for 84×84×8)
- Sparse reward signal
- Lots of irrelevant information (background, etc.)

**Modern approaches:**
1. **World models** - Learn dynamics model, plan in latent space
   - DreamerV3, PlaNet, MuZero
2. **Self-supervised pretraining**
   - CURL (contrastive learning)
   - Data augmentation (random crop, color jitter)
3. **Auxiliary tasks**
   - Predict next frame
   - Predict inverse actions
4. **Use simulators** - Train in sim, transfer to real

**Papers:**
- "Dream to Control: Learning Behaviors by Latent Imagination" (DreamerV2)
- "Mastering Atari with Discrete World Models" (DreamerV3)
- "CURL: Contrastive Unsupervised RL"

**Practical tip:** For learning, use state-based first (8D vector), then move to pixels once algorithm works.

---

## Stability & Optimization

### 8. **Gradient Clipping**

**What we tried:**
```python
clip_grad_norm_(params, 0.5)   → Too restrictive, slow learning
clip_grad_norm_(params, 10.0)  → Better, allows learning
```

**When to use:**
- RNNs (very prone to exploding gradients)
- RL (high variance gradients from environment)
- Transformers (can have gradient spikes)

**Alternatives:**
- Gradient norm monitoring (log max gradient)
- Adaptive clipping (clip to percentile of historical norms)
- Optimizer tricks (Adam with amsgrad, AdamW)

**Paper:**
- "On the difficulty of training RNNs" (explains gradient problems)

### 9. **Loss Functions for RL**

**What we used:**
- MSE → Sensitive to outliers (loss explodes with bad episodes)
- Huber Loss → Better! Linear for large errors
- Could try: Quantile regression, distributional losses

**Why Huber helps:**
```python
error = -100 (crash)
MSE = 100² = 10,000   ← Huge gradient!
Huber = 99.5           ← Bounded gradient
```

### 10. **Mixed Precision Training**

**What you're using:**
```python
torch.cuda.amp.autocast(dtype=torch.bfloat16)
```

**Benefits:**
- 2x faster forward/backward
- 50% memory savings
- bfloat16 > float16 (wider range, more stable)

**Gotchas:**
- Need gradient scaling with float16
- Some ops don't support mixed precision (rare)
- Loss scaling can mask gradient issues

**Resources:**
- [PyTorch AMP Tutorial](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
- [NVIDIA Mixed Precision Training Guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)

---

## Advanced Topics (Next Steps)

### 11. **Better RL Algorithms**

**PPO (Proximal Policy Optimization)** - Your next stop
```python
# Key idea: Clip policy updates to prevent large changes
ratio = new_policy / old_policy
clipped_ratio = clip(ratio, 1-epsilon, 1+epsilon)
loss = min(ratio * advantage, clipped_ratio * advantage)
```

Why it's better than A2C:
- More stable (prevents policy collapse)
- Better sample efficiency
- Industry standard (used in ChatGPT RLHF)

**Resources:**
- [Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- Original paper: "Proximal Policy Optimization Algorithms"

**SAC (Soft Actor-Critic)** - For continuous control
- Entropy-regularized RL (encourages exploration)
- Off-policy (sample efficient)
- Twin Q-networks (stable)

**DQN variants** - For discrete actions
- Rainbow DQN (combines many improvements)
- Double DQN (reduces overestimation)
- Prioritized replay

### 12. **Debugging RL**

**Your debugging journey was excellent!** You systematically:
1. Added logging (critic grad, value variance)
2. Analyzed training curves
3. Identified pessimistic drift
4. Tested hyperparameter changes
5. Tried architectural improvements

**Checklist for debugging RL:**
```
□ Log everything (values, advantages, entropies, gradients)
□ Plot training curves (don't just watch numbers scroll)
□ Check if agent learns anything (random baseline)
□ Verify advantage calculation (print samples)
□ Monitor value variance (is critic learning to differentiate?)
□ Check gradient norms (exploding? vanishing?)
□ Visualize what agent sees (is input correct?)
□ Test on simple environment first (CartPole, etc.)
```

**Resources:**
- [Deep RL Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html) - Honest assessment
- [RL Debugging Guide](https://andyljones.com/posts/rl-debugging.html)
- [Common RL Mistakes](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)

### 13. **Model-Based RL**

**Idea:** Learn a model of environment dynamics, plan with it.

```
Model-free (what you did): Try actions → learn from outcomes
Model-based: Learn dynamics → simulate → plan better
```

**Advantages:**
- Much more sample efficient
- Can plan ahead (like AlphaZero)
- Transfer learned dynamics to new tasks

**Modern approaches:**
- MuZero (learns dynamics in latent space)
- Dreamer (learns world model, imagines trajectories)
- MBPO (model-based policy optimization)

**Papers:**
- "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (MuZero)
- "Dream to Control" (DreamerV2)

---

## Practical Tools & Libraries

### 14. **Stable Baselines3**

Industry-standard RL library with tested implementations:
```python
from stable_baselines3 import PPO

# Your entire training in 3 lines:
model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)
model.save("lunar_lander")
```

**Why use it:**
- Battle-tested implementations
- Good defaults (hyperparameters tuned)
- Learn by reading source code
- Benchmark your custom code against it

**Link:** https://stable-baselines3.readthedocs.io/

### 15. **Weights & Biases (W&B)**

Professional experiment tracking:
```python
import wandb

wandb.init(project="lunar-lander")
wandb.log({"reward": total_reward, "value": mean_value})
```

**Benefits:**
- Beautiful plots
- Compare runs easily
- Share results with team
- Log videos of agent behavior

### 16. **CleanRL**

Single-file RL implementations (great for learning):
- One file = one algorithm
- Heavily commented
- Matches paper implementations
- Easier to understand than libraries

**Link:** https://github.com/vwxyzjn/cleanrl

---

## Books & Courses

### Beginner → Intermediate

**1. Sutton & Barto - "Reinforcement Learning: An Introduction"**
- THE textbook (free online)
- Start with chapters 1-6 (tabular methods)
- Then 9-13 (deep RL)
- Skip chapter 12 initially (eligibility traces)

**2. "Deep Reinforcement Learning Hands-On" by Maxim Lapan**
- PyTorch implementations
- Practical, code-focused
- Covers DQN, A2C, PPO, SAC

**3. David Silver's RL Course (DeepMind)**
- Video lectures (free on YouTube)
- Covers fundamentals very well
- Math-heavy but worth it

### Advanced

**4. "Foundations of Deep RL" by Graesser & Keng**
- Modern deep RL focus
- Good balance of theory/practice

**5. Berkeley Deep RL Course (CS285)**
- Sergey Levine's course
- Cutting edge topics
- Assignments available online

---

## Key Papers to Read

**Foundational (understand the field):**
1. "Playing Atari with Deep RL" - DQN (2013)
2. "Trust Region Policy Optimization" - TRPO (2015)
3. "Proximal Policy Optimization" - PPO (2017)
4. "Soft Actor-Critic" - SAC (2018)

**Stability & Optimization:**
5. "Implementation Matters in Deep RL" (2020)
6. "What Matters in On-Policy RL" (2020)

**Visual RL:**
7. "Human-level control through deep RL" (Nature, 2015)
8. "CURL: Contrastive Unsupervised RL" (2020)

**Architecture:**
9. "Attention is All You Need" (2017) - Transformers
10. "An Image is Worth 16x16 Words" (2021) - ViT

---

## Your Next Projects

Based on our journey, here are good next steps:

**Level 1: Fix Current Setup**
- Get Conv3D + Transformer working well
- Achieve consistent positive rewards
- Compare to ResNet baseline

**Level 2: Algorithm Upgrade**
- Implement PPO (more stable than A2C)
- Use Stable Baselines3 as reference
- Compare your implementation to theirs

**Level 3: Advanced Architectures**
- Add recurrent connections (LSTM between transformer and heads)
- Try contrastive learning (CURL)
- Implement auxiliary tasks (predict next frame)

**Level 4: Harder Environments**
- Try Atari games (more complex than LunarLander)
- MuJoCo locomotion (continuous control)
- Custom environments

**Level 5: Research**
- Reproduce a recent paper
- Improve upon it
- Submit to a conference/workshop

---

## Common Pitfalls (Learned from Your Experience)

**1. Starting with pixels**
→ Always prototype with state-based first

**2. Not logging enough**
→ Log values, advantages, entropies, gradients

**3. Trusting total reward alone**
→ Track value variance, prediction error, etc.

**4. Not comparing to baselines**
→ Random agent, simple heuristics, Stable Baselines3

**5. Hyperparameter sensitivity**
→ RL is very sensitive, need to tune carefully

**6. Ignoring visualization**
→ Watch your agent, plot attention, inspect features

**7. Reward engineering**
→ Sometimes you need to shape rewards (but be careful!)

---

## Questions to Keep Pondering

Based on our discussions, here are deep questions worth thinking about:

1. **Why do CNNs work for supervised learning but struggle for RL?**
   - Hint: Supervised has IID data, RL has correlated sequential data

2. **Can we prevent pessimistic drift without changing the algorithm?**
   - Hint: Architecture, initialization, normalization

3. **What's the right amount of reward clipping?**
   - Hint: Maybe it's state-dependent, not constant

4. **Should value and policy share representations?**
   - Hint: A2C shares, DQN doesn't, PPO does... why?

5. **How do humans learn from visual input so efficiently?**
   - Hint: Object-centric representations, physics priors, curiosity

---

## Your Specific Strengths (From Our Session)

1. **Systems thinking** - You connected supervised learning concepts to RL
2. **Debugging methodology** - Systematic analysis, logging, hypothesis testing
3. **Architectural intuition** - Recognized CNN limitations without prompting
4. **Learning mindset** - Asked "why" not just "what"

**Leverage these!** You have the skills to become very good at RL research/engineering.

---

## Final Advice

**On learning RL:**
- It's frustratingly slow (everything takes 10x longer than you expect)
- Most experiments fail (that's normal!)
- Understanding theory deeply > memorizing tricks
- Reproduce papers, don't just read them

**On visual RL specifically:**
- Start simple (state-based)
- Add complexity gradually (pixels, transformers, etc.)
- Compare to baselines always
- Don't trust your implementation until it works on easy tasks

**On becoming an expert:**
- Implement algorithms from scratch (you'll understand deeply)
- Read code > read papers (see what actually works)
- Track experiments religiously
- Share your learnings (blog, GitHub, Twitter)

---

Good luck! You're asking the right questions and building the right intuitions. Keep at it! 🚀
