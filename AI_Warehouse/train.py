import random

from warehouse import WarehouseEnvironment
from dqn_agent import DQNAgent


# ==========================================================
# TRAINING SETTINGS
# ==========================================================

EPISODES = 1500
BATCH_SIZE = 32

ZONES = ["A", "B", "C"]

PACKAGE_FOR_ZONE = {
    "A": "P-101",
    "B": "P-102",
    "C": "P-103"
}


# ==========================================================
# CREATE ENVIRONMENT + AGENT
# ==========================================================

env = WarehouseEnvironment()
agent = DQNAgent()


successful_deliveries = 0

zone_success = {
    "A": 0,
    "B": 0,
    "C": 0
}


print()
print("=" * 65)
print("AI WAREHOUSE MULTI-ZONE DQN TRAINING")
print("=" * 65)
print()

print("Zones:")
print("  A = Electronics")
print("  B = General Goods")
print("  C = Fragile Items")
print()

print(f"Training Episodes: {EPISODES}")
print()


# ==========================================================
# TRAINING
# ==========================================================

for episode in range(1, EPISODES + 1):

    # ------------------------------------------------------
    # RANDOMLY SELECT DESTINATION
    # ------------------------------------------------------

    target_zone = random.choice(ZONES)

    package_id = PACKAGE_FOR_ZONE[target_zone]

    state = env.reset(
        target_zone=target_zone,
        package_id=package_id
    )

    total_reward = 0

    collision_count = 0

    picked_up = False

    delivered = False


    # ------------------------------------------------------
    # EPISODE
    # ------------------------------------------------------

    for step in range(env.max_steps):

        action = agent.choose_action(
            state,
            training=True
        )

        next_state, reward, done, info = env.step(
            action
        )

        agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )

        agent.train(
            BATCH_SIZE
        )

        state = next_state

        total_reward += reward


        # --------------------------------------------------
        # TRACK RESULTS
        # --------------------------------------------------

        if info["collision"]:
            collision_count += 1

        if info["package_picked"]:
            picked_up = True

        if info["delivered"]:
            delivered = True

        if done:
            break


    # ------------------------------------------------------
    # SUCCESS
    # ------------------------------------------------------

    if delivered:

        successful_deliveries += 1

        zone_success[target_zone] += 1


    # ------------------------------------------------------
    # PROGRESS
    # ------------------------------------------------------

    if episode % 10 == 0:

        print(
            f"Episode {episode:4d} | "
            f"Zone: {target_zone} | "
            f"Reward: {total_reward:7.1f} | "
            f"Steps: {env.steps:3d} | "
            f"Pickup: {picked_up} | "
            f"Delivered: {delivered} | "
            f"Collisions: {collision_count:2d} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )


# ==========================================================
# SAVE MODEL
# ==========================================================

agent.save_model(
    "dqn_model.pth"
)


# ==========================================================
# FINAL REPORT
# ==========================================================

print()
print("=" * 65)
print("MULTI-ZONE TRAINING COMPLETED")
print("=" * 65)

print()

print(
    f"Overall successful deliveries: "
    f"{successful_deliveries}/{EPISODES}"
)

print()

print(
    f"Zone A deliveries: "
    f"{zone_success['A']}"
)

print(
    f"Zone B deliveries: "
    f"{zone_success['B']}"
)

print(
    f"Zone C deliveries: "
    f"{zone_success['C']}"
)

print()

print(
    f"Final epsilon: "
    f"{agent.epsilon:.3f}"
)

print()

print(
    "Model saved to dqn_model.pth"
)

print()