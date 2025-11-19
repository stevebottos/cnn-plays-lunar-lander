# A2C vs PPO: Key Algorithmic Differences

## Overview

Both A2C (Advantage Actor-Critic) and PPO (Proximal Policy Optimization) are policy gradient methods that use an actor-critic architecture. However, PPO is significantly more stable and sample-efficient due to its constrained policy updates.

## Core Differences

### 1. Policy Update Strategy

**A2C (main.py)**:
- **On-policy, single-step updates**: Updates the policy immediately after each episode
- **Direct gradient ascent**: Maximizes expected return with no constraints
- Loss: `-(log_prob * advantage).mean()`
- Problem: Large policy updates can destabilize training (the "cliff" problem)

**PPO (main_ppo.py)**:
- **On-policy, batched updates**: Collects multiple episodes before updating
- **Clipped surrogate objective**: Constrains how much the policy can change
- Loss: `-min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)`
- Solution: Prevents catastrophically large policy updates via clipping

### 2. The Trust Region Concept

The fundamental innovation of PPO is implementing a "trust region" - a safe zone where we know policy updates won't destroy performance.

**A2C limitation**:
```python
# A2C directly uses log probability * advantage
actor_loss = -(log_probs * advantages).mean()
```
This can take arbitrarily large steps in policy space, leading to:
- Sudden performance collapse
- Pessimistic value drift (critic predicting increasingly negative values)
- Difficulty recovering from bad updates

**PPO solution**:
```python
# PPO clips the probability ratio
ratio = torch.exp(log_probs - old_log_probs)  # π_new / π_old
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1-ε, 1+ε) * advantages  # Clipped between 0.8 and 1.2 (if ε=0.2)
actor_loss = -torch.min(surr1, surr2).mean()
```
This ensures the new policy stays close to the old policy:
- If advantage is positive (good action): Allow ratio up to 1.2 (20% increase in probability)
- If advantage is negative (bad action): Allow ratio down to 0.8 (20% decrease in probability)
- Beyond these bounds: gradient becomes zero (no incentive to move further)

### 3. Data Efficiency

**A2C**:
- Each episode's data is used **once** for a single gradient update
- Throws away experience immediately after use
- Requires constant environment interaction

**PPO**:
- Collects multiple episodes (COLLECT_EPISODES = 4 in our code)
- Performs multiple optimization epochs (PPO_EPOCHS = 4) on the same data
- Uses mini-batches (BATCH_SIZE = 64) for better gradient estimates
- Extracts more learning from each environment interaction

Example: After collecting 4 episodes with 400 total timesteps, PPO performs:
- 4 epochs × (400 / 64 batches) ≈ 25 gradient updates
- A2C would only do 4 updates on the same data

### 4. Advantage Estimation

**A2C (main.py)**:
- Uses basic Monte Carlo returns
- Higher variance, slower learning
```python
# Calculate return backwards
R = 0
for r in reversed(rewards):
    R = r + gamma * R
```

**PPO (main_ppo.py)**:
- Uses Generalized Advantage Estimation (GAE)
- Better bias-variance tradeoff via exponential weighting
- GAE_LAMBDA parameter controls the tradeoff (0.95 in our code)
```python
# GAE combines multiple n-step returns
delta = r + gamma * V(s_t+1) - V(s_t)  # TD error
gae = delta + gamma * lambda * gae     # Exponentially weighted
```

GAE formula: `A_t = Σ (γλ)^l * δ_{t+l}` where δ is the TD error

Benefits:
- λ=0: Pure TD (low variance, high bias)
- λ=1: Pure Monte Carlo (high variance, low bias)
- λ=0.95: Sweet spot for most problems

### 5. Learning Rate Tolerance

**A2C**:
- LEARNING_RATE = 1e-5 (very small)
- Higher rates (3e-4) cause critic gradient explosion
- Training at 1e-5 is slow but necessary for stability

**PPO**:
- LEARNING_RATE = 3e-4 (30x higher!)
- Clipping makes training stable even with aggressive learning rates
- Faster learning without divergence

### 6. Hyperparameters Comparison

| Parameter | A2C (main.py) | PPO (main_ppo.py) | Why Different? |
|-----------|---------------|-------------------|----------------|
| Learning Rate | 1e-5 | 3e-4 | PPO's clipping enables higher LR |
| Update Frequency | Every episode | Every 4 episodes | PPO batches for efficiency |
| Epochs per Update | 1 | 4 | PPO reuses data multiple times |
| Clip Epsilon | N/A | 0.2 | PPO's core innovation |
| GAE Lambda | N/A | 0.95 | PPO uses GAE for better advantages |
| Batch Size | Full episode | 64 timesteps | PPO uses mini-batches |

## Training Stability Analysis

### A2C Problems (Observed in Logs)

From `training_log_conv3d_transformer.txt`:
1. **Pessimistic drift**: Values drift negative (-8 → -24 over 10k episodes)
2. **Value collapse**: Value std drops (0.24 → 0.01), critic becomes overconfident
3. **High variance**: Episode rewards jump wildly (-200 to +73)
4. **Slow improvement**: Only 8% positive episodes after 10k episodes

Why these happen:
- No constraint on policy updates → large changes → bad experiences
- Critic learns from bad experiences → becomes pessimistic
- Pessimistic critic → negative advantages → policy becomes worse
- Vicious cycle forms (pessimistic drift feedback loop)

### PPO Expected Improvements

1. **Reduced pessimistic drift**: Clipping prevents extreme policy shifts
2. **Stable value learning**: Multiple epochs smooth out value estimates
3. **Lower variance**: Better advantage estimates via GAE
4. **Faster convergence**: Higher learning rate + data reuse

## Computational Cost

**A2C**:
- Faster per-episode (immediate update)
- Less GPU memory (no rollout buffer)
- But needs more episodes to learn

**PPO**:
- Slower per-episode (batched updates)
- More GPU memory (stores 4 episodes + does 4 epochs)
- But learns faster per environment interaction

Trade-off: PPO is 10-20% slower wall-clock time but typically reaches good performance in 50% fewer environment steps.

## When to Use Each

**Use A2C when**:
- You need simplest possible implementation
- Environment interactions are very cheap
- You're doing research and want baseline algorithm
- You want to understand policy gradients fundamentals

**Use PPO when**:
- You want state-of-the-art performance
- Environment interactions are expensive (e.g., robotics, expensive simulators)
- You need stable, reliable training
- You're deploying in production

## Code Structure Comparison

### A2C Training Loop (main.py:341-476)
```python
for episode in range(NUM_EPISODES):
    # 1. Collect episode data
    while not done:
        action = sample_from_policy(state)
        state, reward = env.step(action)
        store(state, action, reward, value)

    # 2. Calculate returns (Monte Carlo)
    returns = compute_discounted_returns(rewards)

    # 3. Calculate advantages
    advantages = returns - values

    # 4. Single gradient update
    loss = actor_loss + critic_loss
    loss.backward()
    optimizer.step()
```

### PPO Training Loop (main_ppo.py:471-608)
```python
rollout_buffer = []

for episode in range(NUM_EPISODES):
    # 1. Collect episode data
    while not done:
        action = sample_from_policy(state)
        state, reward = env.step(action)
        rollout_buffer.append((state, action, reward, value, log_prob))

    # 2. Calculate advantages (GAE)
    advantages, returns = compute_gae(rewards, values)
    rollout_buffer.add(advantages, returns)

    # 3. When buffer is full (e.g., 4 episodes)
    if len(rollout_buffer) >= COLLECT_EPISODES:
        # 4. Multiple epochs of optimization
        for epoch in range(PPO_EPOCHS):
            # 5. Mini-batch updates
            for batch in minibatches(rollout_buffer, BATCH_SIZE):
                # 6. Clipped policy loss
                ratio = new_policy / old_policy
                clipped_loss = min(ratio * adv, clip(ratio) * adv)

                loss = clipped_loss + critic_loss
                loss.backward()
                optimizer.step()

        # 7. Clear buffer
        rollout_buffer.clear()
```

## Key Takeaways

1. **PPO's clipping is the killer feature**: It provides stability without the complexity of natural gradients (TRPO)

2. **A2C is educational**: Great for learning RL concepts, but PPO is better for real applications

3. **PPO is the modern standard**: Used by OpenAI (GPT-4 RLHF), DeepMind, and most production RL systems

4. **The performance gap is significant**: In our LunarLander task, expect PPO to:
   - Reach 50% positive episodes 3-5x faster than A2C
   - Maintain more stable value estimates
   - Achieve higher final performance

5. **Data efficiency matters**: PPO's ability to reuse data makes it especially valuable when environment interaction is the bottleneck

## Further Reading

- **PPO paper**: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- **GAE paper**: "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (Schulman et al., 2016)
- **A2C/A3C paper**: "Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al., 2016)
- **OpenAI Spinning Up**: Excellent explanations of both algorithms with code

## Diagnostic: Monitoring PPO Training

Watch for these metrics in logs:

1. **Clip Fraction**: Should be 0.1-0.3
   - Too low (<0.05): Increase learning rate or reduce CLIP_EPSILON
   - Too high (>0.5): Decrease learning rate or increase CLIP_EPSILON

2. **Value Loss**: Should decrease steadily
   - If increasing: Critic is struggling, consider reducing value_coeff

3. **Entropy**: Should decrease slowly
   - Rapid drop to zero: Policy becoming deterministic too fast, increase entropy_coeff
   - Stays high: Policy not learning, check other hyperparameters

4. **Reward Trend**: Should be noisy but upward trending
   - No improvement after 1000 episodes: Architectural issue or bad hyperparameters
