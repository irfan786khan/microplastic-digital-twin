
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import pickle
from collections import deque
import random
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MULTI-AGENT REINFORCEMENT LEARNING (MARL)")
print("MICROPLASTIC POLLUTION CONTROL")
print("="*80)



# Load Digital Twin data
try:
    with open('./data/digital_twin.pkl', 'rb') as f:
        digital_twin_data = pickle.load(f)
    print("   ✅ Digital Twin loaded")
except:
    print("   ⚠️ Digital Twin not found. Creating from scratch...")
    digital_twin_data = {
        'feature_cols': ['temperature', 'turbidity', 'ph', 'dissolved_oxygen', 'salinity', 'discharge', 'rainfall'],
        'scaler_X': None,
        'scaler_y': None,
        'actions': ['reduce_waste', 'improve_treatment', 'add_monitoring', 'increase_filtration', 'reduce_runoff'],
        'action_effects': {
            'reduce_waste': {'turbidity': 0.9, 'discharge': 0.95},
            'improve_treatment': {'turbidity': 0.85, 'dissolved_oxygen': 1.05},
            'add_monitoring': {'turbidity': 0.95, 'ph': 1.02},
            'increase_filtration': {'turbidity': 0.8, 'salinity': 0.95},
            'reduce_runoff': {'turbidity': 0.85, 'rainfall': 0.9}
        }
    }

feature_cols = digital_twin_data['feature_cols']
actions = digital_twin_data['actions']
action_effects = digital_twin_data['action_effects']

# Create simple Digital Twin environment for RL
class SimpleDigitalTwin:
    """Simplified Digital Twin for RL training"""

    def __init__(self, feature_cols, action_effects):
        self.feature_cols = feature_cols
        self.action_effects = action_effects
        self.n_features = len(feature_cols)
        self.state = None
        self.concentration = 5.0  # Initial concentration
        self.history = []
        self.step_count = 0

    def reset(self):
        """Reset environment"""
        self.state = np.array([0.0] * self.n_features)
        self.concentration = 5.0 + np.random.randn() * 0.5
        self.history = []
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        """Get current state"""
        return np.array([
            self.concentration / 10.0,  # Normalized concentration
            self.state[1] if len(self.state) > 1 else 0.5,
            self.state[2] if len(self.state) > 2 else 0.5,
            self.state[3] if len(self.state) > 3 else 0.5,
            self.state[4] if len(self.state) > 4 else 0.5,
            self.state[5] if len(self.state) > 5 else 0.5,
            self.step_count / 100.0  # Time step
        ])

    def step(self, action_indices):
        """
        Take multiple actions from different agents
        action_indices: list of actions from each agent
        """
        self.step_count += 1

        # Apply all actions
        total_effect = 1.0
        for action_idx in action_indices:
            if action_idx < len(self.actions):
                action = self.actions[action_idx]
                effect = self.action_effects.get(action, {})
                for feature, multiplier in effect.items():
                    total_effect *= multiplier

        # Update concentration with some randomness
        reduction = 0.1 * (1 - total_effect) + 0.05 * np.random.randn()
        self.concentration = max(0.1, self.concentration - reduction * self.concentration)

        # Calculate reward
        reward = -self.concentration / 10.0  # Lower concentration = higher reward

        # Check if done
        done = self.concentration < 0.2 or self.concentration > 15.0 or self.step_count > 50

        # Store history
        self.history.append({
            'concentration': self.concentration,
            'reward': reward,
            'actions': action_indices
        })

        return self._get_state(), reward, done, {'concentration': self.concentration}

# Create Digital Twin
dt = SimpleDigitalTwin(feature_cols, action_effects)
dt.actions = actions

print(f"   ✅ Digital Twin ready with {len(actions)} actions")



class DQNAgent:
    """Deep Q-Network Agent"""

    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.001, gamma=0.95, epsilon=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory = deque(maxlen=10000)
        self.batch_size = 64

        # Q-Network
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.target_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.update_target()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_network.to(self.device)
        self.target_network.to(self.device)

    def update_target(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Select action"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return q_values.argmax().item()

    def replay(self):
        """Train on batch"""
        if len(self.memory) < self.batch_size:
            return 0

        batch = random.sample(self.memory, self.batch_size)
        states = torch.tensor([b[0] for b in batch], dtype=torch.float32).to(self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(self.device)
        next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32).to(self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32).to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Loss
        loss = F.mse_loss(current_q.squeeze(), target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()



class MultiAgentSystem:
    """
    Multi-Agent Reinforcement Learning System
    """

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Create agents
        self.agents = {
            'industrial': DQNAgent(state_dim, action_dim, lr=0.001, epsilon=0.15),
            'wastewater': DQNAgent(state_dim, action_dim, lr=0.001, epsilon=0.15),
            'monitoring': DQNAgent(state_dim, action_dim, lr=0.001, epsilon=0.15),
            'policy': DQNAgent(state_dim, action_dim, lr=0.001, epsilon=0.10)
        }

        self.agent_names = list(self.agents.keys())
        self.history = []

    def get_actions(self, state):
        """Get actions from all agents"""
        actions = {}
        for name, agent in self.agents.items():
            actions[name] = agent.act(state)
        return actions

    def learn(self, states, actions, rewards, next_states, dones):
        """Train all agents"""
        losses = {}
        for i, (name, agent) in enumerate(self.agents.items()):
            agent.remember(states[i], actions[i], rewards[i], next_states[i], dones[i])
            loss = agent.replay()
            losses[name] = loss
        return losses

    def update_targets(self):
        """Update target networks"""
        for agent in self.agents.values():
            agent.update_target()

    def get_agent_actions(self, state):
        """Get actions with names"""
        actions = self.get_actions(state)
        return actions



def train_marl(episodes=100, steps_per_episode=50):
    """Train the multi-agent system"""

    # Initialize
    state_dim = 7  # concentration, features, time
    action_dim = len(actions)
    marl = MultiAgentSystem(state_dim, action_dim)
    dt = SimpleDigitalTwin(feature_cols, action_effects)
    dt.actions = actions

    # Training metrics
    episode_rewards = []
    episode_concentrations = []

    print("\n   Training Progress:")
    print("-"*70)

    for episode in range(episodes):
        state = dt.reset()
        total_reward = 0
        episode_conc = []
        episode_losses = []

        for step in range(steps_per_episode):
            # Get actions from all agents
            agent_actions = marl.get_actions(state)
            action_list = list(agent_actions.values())

            # Take step in environment
            next_state, reward, done, info = dt.step(action_list)

            # Store for each agent
            for i, (name, agent) in enumerate(marl.agents.items()):
                agent.remember(state, action_list[i], reward, next_state, done)
                loss = agent.replay()
                episode_losses.append(loss if loss else 0)

            total_reward += reward
            episode_conc.append(info['concentration'])
            state = next_state

            if done:
                break

        # Update target networks every 10 episodes
        if (episode + 1) % 10 == 0:
            marl.update_targets()

        # Store metrics
        episode_rewards.append(total_reward)
        episode_concentrations.append(np.mean(episode_conc) if episode_conc else 5.0)

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_conc = np.mean(episode_concentrations[-10:])
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            print(f"   Episode {episode+1:3d}/{episodes} | Avg Reward: {avg_reward:.3f} | Avg Concentration: {avg_conc:.3f} | Loss: {avg_loss:.4f}")

    print("-"*70)
    print("✅ Training complete!")

    return marl, episode_rewards, episode_concentrations

# Train the system
marl_system, rewards, concentrations = train_marl(episodes=100, steps_per_episode=50)



def evaluate_agents(marl_system, n_episodes=10):
    """Evaluate trained agents"""

    dt = SimpleDigitalTwin(feature_cols, action_effects)
    dt.actions = actions

    eval_results = []

    print("\n   Evaluation Results:")
    print("-"*70)

    for episode in range(n_episodes):
        state = dt.reset()
        total_reward = 0
        episode_conc = []

        for step in range(30):
            # Get actions
            agent_actions = marl_system.get_actions(state)
            action_list = list(agent_actions.values())

            # Take step
            next_state, reward, done, info = dt.step(action_list)
            total_reward += reward
            episode_conc.append(info['concentration'])
            state = next_state

            if done:
                break

        final_conc = episode_conc[-1] if episode_conc else dt.concentration
        eval_results.append({
            'episode': episode + 1,
            'total_reward': total_reward,
            'final_concentration': final_conc,
            'avg_concentration': np.mean(episode_conc) if episode_conc else final_conc
        })

        print(f"   Episode {episode+1:2d} | Reward: {total_reward:.3f} | Final Conc: {final_conc:.3f}")

    print("-"*70)

    return pd.DataFrame(eval_results)

eval_df = evaluate_agents(marl_system, n_episodes=10)



def analyze_agent_actions(marl_system, n_steps=100):
    """Analyze what actions each agent prefers"""

    dt = SimpleDigitalTwin(feature_cols, action_effects)
    dt.actions = actions

    action_counts = {name: [0] * len(actions) for name in marl_system.agent_names}

    state = dt.reset()

    for _ in range(n_steps):
        agent_actions = marl_system.get_actions(state)

        for name, action in agent_actions.items():
            action_counts[name][action] += 1

        action_list = list(agent_actions.values())
        next_state, reward, done, _ = dt.step(action_list)
        state = next_state

        if done:
            state = dt.reset()

    print("\n   Action Preferences:")
    print("-"*70)

    for name, counts in action_counts.items():
        total = sum(counts)
        percentages = [c/total*100 for c in counts] if total > 0 else [0]*len(actions)
        print(f"\n   {name.upper()} Agent:")
        for i, (action, pct) in enumerate(zip(actions, percentages)):
            bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
            print(f"      {action:<25} {pct:5.1f}% {bar}")

analyze_agent_actions(marl_system, n_steps=100)



def get_best_strategy(marl_system, n_trials=20):
    """Get the best pollution control strategy"""

    dt = SimpleDigitalTwin(feature_cols, action_effects)
    dt.actions = actions

    strategies = []

    for trial in range(n_trials):
        state = dt.reset()
        action_sequence = []
        total_reward = 0
        final_conc = 0

        for step in range(30):
            agent_actions = marl_system.get_actions(state)
            action_list = list(agent_actions.values())
            action_sequence.append(action_list)

            next_state, reward, done, info = dt.step(action_list)
            total_reward += reward
            state = next_state
            final_conc = info['concentration']

            if done:
                break

        strategies.append({
            'trial': trial + 1,
            'actions': action_sequence,
            'total_reward': total_reward,
            'final_concentration': final_conc
        })

    # Find best strategy
    best = max(strategies, key=lambda x: x['total_reward'])

    print("\n   📈 Best Strategy Found:")
    print(f"   - Trial: {best['trial']}")
    print(f"   - Total Reward: {best['total_reward']:.3f}")
    print(f"   - Final Concentration: {best['final_concentration']:.3f}")
    print(f"   - Number of Steps: {len(best['actions'])}")

    # Show recommended actions
    print("\n   📋 Recommended Action Sequence:")
    print("-"*70)

    for step, action_list in enumerate(best['actions'][:10]):
        action_names = [actions[a] for a in action_list]
        print(f"   Step {step+1:2d}: Industrial={action_names[0]:<20} | Treatment={action_names[1]:<20} | Monitor={action_names[2]:<20} | Policy={action_names[3]:<20}")

    if len(best['actions']) > 10:
        print(f"   ... and {len(best['actions']) - 10} more steps")

    print("-"*70)

    return best

best_strategy = get_best_strategy(marl_system, n_trials=20)



# Save all agent models
os.makedirs('./data/rl_models', exist_ok=True)

for name, agent in marl_system.agents.items():
    torch.save(agent.q_network.state_dict(), f'./data/rl_models/agent_{name}.pt')
    print(f"   ✅ Saved: agent_{name}.pt")

# Save training history
history_df = pd.DataFrame({
    'episode': range(1, len(rewards) + 1),
    'reward': rewards,
    'concentration': concentrations
})
history_df.to_csv('./data/rl_models/training_history.csv', index=False)
print(f"   ✅ Saved: training_history.csv")

# Save best strategy
strategy_df = pd.DataFrame({
    'step': range(1, len(best_strategy['actions']) + 1),
    'actions': [str(a) for a in best_strategy['actions']]
})
strategy_df.to_csv('./data/rl_models/best_strategy.csv', index=False)
print(f"   ✅ Saved: best_strategy.csv")



print(f"""
📊 Multi-Agent System:

   ┌─────────────────────────────────────────────────────────────────────────┐
   │ Agent                │ Role                           │ Status         │
   ├──────────────────────┼────────────────────────────────┼────────────────┤
   │ Industrial Control   │ Reduce discharge from industry │ ✅ Trained     │
   │ Wastewater Treatment │ Increase filtration efficiency  │ ✅ Trained     │
   │ Monitoring System    │ Optimal sampling locations      │ ✅ Trained     │
   │ Government Policy    │ Best pollution control strategy │ ✅ Trained     │
   └──────────────────────┴────────────────────────────────┴────────────────┘

📈 Training Performance:
   - Episodes: 100
   - Steps per episode: 50
   - Final Avg Reward: {np.mean(rewards[-10:]):.3f}
   - Final Avg Concentration: {np.mean(concentrations[-10:]):.3f}

📁 Output Files:
   ├── ./data/rl_models/agent_industrial.pt
   ├── ./data/rl_models/agent_wastewater.pt
   ├── ./data/rl_models/agent_monitoring.pt
   ├── ./data/rl_models/agent_policy.pt
   ├── ./data/rl_models/training_history.csv
   └── ./data/rl_models/best_strategy.csv

🎯 Best Strategy:
   - Reward: {best_strategy['total_reward']:.3f}
   - Final Concentration: {best_strategy['final_concentration']:.3f}
   - Steps: {len(best_strategy['actions'])}
""")

print("="*80)
print("✅ MULTI-AGENT RL COMPLETE")
print("="*80)



def get_pollution_control_strategy(concentration, agent_actions=None):
    """
    Get pollution control strategy from trained agents

    Parameters:
    - concentration: current pollution level (items/m³)
    - agent_actions: optional specific actions

    Returns:
    - Recommended actions from each agent
    """

    if agent_actions is None:
        # Use trained agents
        state = np.array([
            concentration / 10.0,
            0.5, 0.5, 0.5, 0.5, 0.5, 0.0
        ])

        actions = marl_system.get_actions(state)

        return {
            'industrial_control': actions['industrial'],
            'wastewater_treatment': actions['wastewater'],
            'monitoring_system': actions['monitoring'],
            'government_policy': actions['policy'],
            'recommended_actions': [actions[a] for a in marl_system.agent_names]
        }
    else:
        return agent_actions

# Test the function
print("\n   Testing strategy function...")
test_strategy = get_pollution_control_strategy(concentration=3.5)
print(f"   Industrial Control: {actions[test_strategy['industrial_control']]}")
print(f"   Wastewater Treatment: {actions[test_strategy['wastewater_treatment']]}")
print(f"   Monitoring System: {actions[test_strategy['monitoring_system']]}")
print(f"   Government Policy: {actions[test_strategy['government_policy']]}")

print("\n" + "="*80)
print("✅ MULTI-AGENT RL SYSTEM READY")
print("="*80)