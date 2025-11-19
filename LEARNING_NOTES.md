# RL Training Stability - Learning Notes

## Problem We Were Solving

### Initial Symptoms
```
Episode   10 | Steps: 121 | Total Reward: -312.90 | Actor Loss: -0.0021 | Critic Loss: 12854.3740
Episode   24 | Steps: 114 | Total Reward: -427.06 | Actor Loss: -0.0005 | Critic Loss: 23290.8887
Episode   41 | Steps: 118 | Total Reward: -452.72 | Actor Loss: 0.0012 | Critic Loss: 24662.2383
Episode  130 | Steps:  80 | Total Reward: -471.31 | Actor Loss: -0.0026 | Critic Loss: 34149.5859
```

### Key Issues
1. **Critic Loss Exploding** - Going from hundreds to tens of thousands
2. **Actor Loss Tiny** - Around ±0.001 to 0.004 (barely learning)
3. **High Variance** - Rewards swinging wildly from -60 to -400+
4. **No Improvement** - Agent kept crashing, not learning

---

## Root Causes Explained

### 1. The Critic Loss Explosion

**What the critic does:**
- Predicts the value function V(s) = "expected future return from this state"
- Trained via MSE loss: `loss = (predicted_value - actual_return)²`

**Why it exploded:**
```python
# Episode with crash (actual return = -400)
predicted_value = -50  # Critic's guess
actual_return = -400   # What actually happened
error = -400 - (-50) = -350
loss = (-350)² = 122,500  # MASSIVE!
```

When you have:
- Large negative returns (-400)
- High learning rate (1e-5)
- MSE loss (squares the error)

→ Critic overcorrects → Next episode it's way off again → Vicious cycle

### 2. Why High Learning Rates Are Dangerous Here

**Learning rate controls update size:**
```python
new_value = old_value - learning_rate * gradient
```

With high variance returns and high LR:
- Episode 1: Return = -50 → Critic learns "state is bad"
- Episode 2: Return = -400 → Critic overreacts "state is TERRIBLE"
- Episode 3: Return = -80 → Critic is confused, now predicts too negative
- Predictions swing wildly, never converge

**The fix:** Lower learning rate = smaller, more stable updates

### 3. The Actor-Critic Coupling Problem

Actor learns from **advantages**:
```python
advantage = actual_return - critic_prediction
actor_loss = -(log_prob * advantage)
```

When critic is unstable:
- Advantages become meaningless noise
- Actor gets bad gradient signals
- Policy doesn't improve
- Bad episodes continue
- Critic stays unstable

It's a feedback loop!

---

## Understanding Bootstrapped Returns

### What Are Returns?

**Return (Gt)** = Total discounted future reward from time t:
```python
Gt = rt + γ*rt+1 + γ²*rt+2 + γ³*rt+3 + ...
```

Example with rewards [10, 5, 3] and γ=0.99:
```
G0 = 10 + 0.99*5 + 0.99²*3 = 17.88
G1 = 5 + 0.99*3 = 7.97
G2 = 3
```

### Two Ways to Calculate Returns

#### Monte Carlo (what we're using):
```python
Gt = sum of actual rewards until episode ends
```
- ✅ **Pro:** Unbiased (true values)
- ❌ **Con:** High variance, only learn at episode end

#### Bootstrapped (TD learning):
```python
Gt ≈ rt + γ*V(st+1)
     ↑       ↑
  actual   critic's guess
```
- ✅ **Pro:** Lower variance, learn every step
- ❌ **Con:** Biased by critic's errors

### Why High Variance Matters

With Monte Carlo returns on a bad task:
```python
# Code calculates true returns (main.py:203-205)
returns = calculate_returns_and_advantages(episode_rewards, values, GAMMA)

# If episode crashes immediately:
episode_rewards = [-100, -200, -127.50]  # LunarLander crash penalties
returns = [-400.17, -301.23, -127.50]    # Massive negative values
```

Critic tries to fit these huge, noisy targets → unstable training

---

## Solutions Applied

### 1. Reward Clipping (Most Important!)

**What it does:**
```python
# Before
reward = -400  # Huge crash penalty

# After (main.py:178-180)
clipped_reward = np.clip(reward, -10, +10)  # = -10
```

**Why this works:**
- Bounds the maximum return magnitude
- Critic loss = (predicted - return)² stays manageable
- Even worst episodes produce bounded losses
- Agent still learns relative comparisons: "crash bad (-10) vs land good (+10)"

**Tradeoff:**
- Agent learns on "clipped" rewards, not true rewards
- But behavior is often the same! Policy gradient cares about *which actions are better*, not exact reward magnitudes

**Math:**
```python
# Without clipping
worst_return = -400
critic_loss = (-400 - predicted)² ≈ 160,000 (if predicted ≈ 0)

# With clipping to ±10
worst_return = -10 * episode_length ≈ -1000 (if all rewards clipped to -10)
# But typically much less because many steps have small rewards
critic_loss = (-100 - predicted)² ≈ 10,000 (16x smaller!)
```

### 2. Simplified Learning Rate

**Before:**
```python
BACKBONE_LEARNING_RATE = 1e-7
ACTOR_LEARNING_RATE = 6e-7
CRITIC_LEARNING_RATE = 1e-6  # Highest, but still causing issues
```

**After:**
```python
LEARNING_RATE = 1e-6  # Single rate for everything
```

**Why:**
- Simpler to tune
- Separate rates weren't solving the root problem (reward scale)
- All components can learn together at stable pace

**Is it standard for critic to learn faster?**
- **No!** Common misconception
- Same rate often works fine
- Higher critic rate only helps if: critic lags behind fast-improving policy
- Lower critic rate helps if: value estimates are unstable (our case)

### 3. Reduced Entropy Coefficient

**Before:**
```python
ENTROPY_COEFF = 0.05
```

**After:**
```python
ENTROPY_COEFF = 0.01  # 5x reduction
```

**What entropy does:**
```python
# Entropy measures randomness of action distribution
entropy = -sum(p * log(p)) for all actions

# High entropy = uniform distribution (lots of exploration)
# Low entropy = peaked distribution (agent commits to actions)

# Loss includes entropy bonus (main.py:217)
total_loss = actor_loss + critic_loss - (entropy * ENTROPY_COEFF)
#                                        ↑
#                            Negative = reward for being random
```

**Why reduce it:**
- High exploration → more random actions → more crashes
- Lower entropy → agent can exploit what it learns faster
- Once agent finds "don't crash immediately", stick with it!

**Tradeoff:**
- Less exploration might miss optimal policy
- But stability > optimality when you're crashing every episode

---

## Key Takeaways

### 1. Reward Scale Matters!
- RL algorithms are sensitive to reward magnitude
- Large rewards → large gradients → instability
- **Always normalize or clip rewards** when training from scratch

### 2. The Critic Stability Problem
- Critic learns from bootstrapped/MC returns (high variance)
- Unstable critic → bad advantages → actor can't learn
- **Stabilize critic first** (clipping, lower LR, smaller losses)

### 3. Learning Rate Tuning
- Higher LR ≠ faster learning in RL
- High variance + high LR = oscillation, not convergence
- Start conservative, increase if stuck

### 4. Exploration vs Exploitation
- More exploration ≠ better learning
- Early on: need stability to learn *anything*
- Later: can increase exploration once basics work

### 5. Monitor These Metrics
```
✅ Good signs:
- Critic loss decreasing over time
- Actor loss stable (not oscillating wildly)
- Occasional positive reward episodes
- Episode length increasing

❌ Bad signs:
- Critic loss exploding (>10,000)
- Actor loss near zero (no learning)
- Rewards not improving after 100+ episodes
- High variance with no trend
```

---

## Common RL Stability Tricks

### Reward Engineering
1. **Clipping** (what we did): Bound rewards to range
2. **Normalization**: Scale to mean=0, std=1
3. **Reward shaping**: Add intermediate rewards

### Loss/Gradient Control
1. **Gradient clipping** (we have this): `clip_grad_norm_(params, 0.5)`
2. **Value loss clipping**: Clip critic loss directly
3. **Huber loss**: Less sensitive to outliers than MSE

### Architecture Choices
1. **Separate networks**: Don't share weights between actor/critic
2. **Target networks**: Use old critic for advantage calculation
3. **Layer normalization**: Normalize activations

### Algorithm Variants
1. **PPO**: Clips policy updates (very stable)
2. **TD(λ)**: Mix Monte Carlo and bootstrapping
3. **GAE**: Generalized Advantage Estimation (variance reduction)

---

## Further Reading

- **Sutton & Barto Chapter 13**: Policy Gradient Methods
- **Spinning Up in Deep RL**: OpenAI's educational resource
- **Stable Baselines3**: Well-tested RL implementations
- **"Deep RL Doesn't Work Yet"** (blog): Common pitfalls

---

## Experiment Ideas

Once training is stable, try:

1. **Remove clipping gradually**: Increase REWARD_CLIP from 10 → 20 → 50 → inf
2. **Increase entropy**: See if more exploration helps after basics learned
3. **Separate networks**: Don't share feature_extractor between actor/critic
4. **Different γ**: Try 0.95 or 0.999 (changes how far agent "looks ahead")
5. **Add baseline**: Subtract running mean from returns
