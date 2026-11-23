# AI Warehouse Automation using Reinforcement Learning

An interactive **AI-powered warehouse robot simulation** built using **Deep Q-Learning (DQN)**, Python, PyTorch, and Streamlit.

The system demonstrates how reinforcement learning can be applied to warehouse automation for package pickup, navigation, obstacle avoidance, and delivery.

## 🚀 Project Overview

The warehouse contains multiple delivery zones and packages. The AI robot learns to navigate through the warehouse, collect the assigned package, and deliver it to the correct destination.

The system provides a real-time dashboard where the robot's movement, AI decisions, rewards, collisions, and delivery status can be observed.

## 🎯 Objectives

- Automate package pickup and delivery
- Navigate the warehouse efficiently
- Avoid obstacles and boundary collisions
- Minimize unnecessary movement
- Use reinforcement learning for decision-making
- Provide real-time visualization of AI decisions
- Demonstrate reward-based learning in a practical warehouse scenario

## 🧠 Reinforcement Learning Components

| Component | Implementation |
|---|---|
| Agent | Warehouse delivery robot |
| Environment | 10×10 warehouse grid |
| State | Robot, package, destination and nearby obstacles |
| Actions | Up, Down, Left, Right |
| Reward | Delivery reward, movement reward and collision penalties |
| Algorithm | Deep Q-Network (DQN) |

## 📦 Package & Delivery Zones

| Package | Destination | Category |
|---|---|---|
| P-101 | Zone A | Electronics |
| P-102 | Zone B | General Goods |
| P-103 | Zone C | Fragile Items |

The robot first navigates to the assigned package, picks it up, and then navigates to the corresponding delivery zone.

## 🤖 System Workflow

```text
Task Assigned
      ↓
Robot Observes Environment
      ↓
AI Selects Action
      ↓
Robot Navigates to Package
      ↓
Package Picked Up
      ↓
Robot Navigates to Destination
      ↓
Obstacle / Collision Handling
      ↓
Package Delivered
      ↓
Reward Received
```

## 🏭 Warehouse Features

- Multiple delivery zones
- Package pickup mechanism
- AI-controlled robot movement
- Deep Q-Network decision making
- Dynamic obstacle placement
- Collision detection
- Collision penalties
- Delivery rewards
- Live step and reward tracking
- AI Q-value visualization
- Real-time event log
- Streamlit dashboard

## 🛠️ Technologies Used

- Python
- PyTorch
- Streamlit
- NumPy
- Deep Reinforcement Learning
- Deep Q-Network (DQN)

## 📁 Project Structure

```text
AI-Warehouse-RL/
│
├── AI_Warehouse/
│   ├── app.py
│   ├── warehouse.py
│   ├── dqn_agent.py
│   ├── train.py
│   ├── dqn_model.pth
│   └── requirements.txt
│
└── README.md
```

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/preethisiva725/AI-Warehouse-RL.git
```

### 2. Open the project folder

```bash
cd AI-Warehouse-RL/AI_Warehouse
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 🎮 Using the Dashboard

1. Select the required delivery zone.
2. Start the AI delivery.
3. The robot navigates to the assigned package.
4. The robot picks up the package.
5. The robot navigates toward the selected destination.
6. Obstacles can be introduced during the demonstration.
7. The system records collisions and penalties.
8. The robot delivers the package.
9. The dashboard displays the delivery result and AI performance.

## 🏆 Reinforcement Learning Reward Concept

The robot receives feedback based on its actions.

Examples include:

- Positive reward for moving toward the package
- Positive reward for picking up the package
- Positive reward for moving toward the destination
- Large positive reward for successful delivery
- Negative reward for collisions
- Negative reward for inefficient movement
- Negative reward for boundary violations

This allows the agent to learn a better navigation policy through repeated interaction with the warehouse environment.

## 📊 AI Decision Visualization

The dashboard displays the estimated Q-values for available actions:

```text
UP       → Q-value
DOWN     → Q-value
LEFT     → Q-value
RIGHT    → Q-value
```

The selected action represents the AI's current decision based on its learned policy.

## 🔬 Training

The DQN model was trained across multiple warehouse delivery tasks involving Zones A, B, and C.

The trained model is included in:

```text
dqn_model.pth
```

This allows the application to run using the trained model without requiring model training every time.

## 🔮 Future Improvements

Possible extensions include:

- Multiple robots operating simultaneously
- Dynamic human/robot movement
- Larger warehouse environments
- Advanced path planning
- Real-time traffic management
- Multi-agent reinforcement learning
- Cloud deployment
- More realistic warehouse layouts
- Continuous learning from new warehouse data

## 👩‍💻 Project

**AI Warehouse Automation using Reinforcement Learning**

Built as an academic project demonstrating the practical application of reinforcement learning in warehouse automation and package delivery.
