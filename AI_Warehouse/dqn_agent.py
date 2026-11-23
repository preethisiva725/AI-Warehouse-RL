import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =========================================================
# DQN NETWORK
# =========================================================

class DQN(nn.Module):

    def __init__(self, state_size=15, action_size=4):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(state_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, action_size)
        )

    def forward(self, x):

        return self.network(x)


# =========================================================
# DQN AGENT
# =========================================================

class DQNAgent:

    def __init__(
        self,
        state_size=15,
        action_size=4
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.learning_rate = 0.001

        self.memory = deque(
            maxlen=20000
        )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = DQN(
            state_size,
            action_size
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        self.loss_function = nn.MSELoss()


    # =====================================================
    # ACTION
    # =====================================================

    def choose_action(self, state):

        if random.random() <= self.epsilon:

            return random.randrange(
                self.action_size
            )

        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():

            q_values = self.model(
                state_tensor
            )

        return int(
            torch.argmax(q_values).item()
        )


    # =====================================================
    # MEMORY
    # =====================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )


    # =====================================================
    # TRAIN
    # =====================================================

    def train(self, batch_size=32):

        if len(self.memory) < batch_size:

            return

        batch = random.sample(
            self.memory,
            batch_size
        )

        states = torch.FloatTensor(
            np.array(
                [x[0] for x in batch]
            )
        ).to(self.device)

        actions = torch.LongTensor(
            [x[1] for x in batch]
        ).to(self.device)

        rewards = torch.FloatTensor(
            [x[2] for x in batch]
        ).to(self.device)

        next_states = torch.FloatTensor(
            np.array(
                [x[3] for x in batch]
            )
        ).to(self.device)

        dones = torch.FloatTensor(
            [x[4] for x in batch]
        ).to(self.device)

        current_q = self.model(
            states
        ).gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():

            next_q = self.model(
                next_states
            ).max(1)[0]

            target_q = (
                rewards
                +
                (1 - dones)
                *
                self.gamma
                *
                next_q
            )

        loss = self.loss_function(
            current_q,
            target_q
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay


    # =====================================================
    # SAVE MODEL
    # =====================================================

    def save_model(self, path):

        torch.save(
            self.model.state_dict(),
            path
        )


    # =====================================================
    # LOAD MODEL
    # =====================================================

    def load_model(self, path):

        self.model.load_state_dict(
            torch.load(
                path,
                map_location=self.device
            )
        )

        self.model.eval()

        print(
            f"Model loaded from {path}"
        )


    # =====================================================
    # Q VALUES
    # =====================================================

    def get_q_values(self, state):

        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():

            q_values = self.model(
                state_tensor
            )

        return q_values.cpu().numpy()[0]