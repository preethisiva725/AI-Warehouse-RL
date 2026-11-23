import streamlit as st
import numpy as np
import time

from warehouse import Warehouse
from dqn_agent import DQNAgent


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="AI Warehouse Control System",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# STYLE
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 1rem;
}

.title {
    font-size: 34px;
    font-weight: 800;
    color: #17202a;
}

.subtitle {
    color: #64748b;
    margin-bottom: 20px;
}

.section {
    font-size: 20px;
    font-weight: 800;
    margin-top: 10px;
}

.status {
    padding: 13px;
    border-radius: 9px;
    font-weight: 700;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# CONSTANTS
# =========================================================

MODEL_PATH = "dqn_model.pth"

ZONE_INFO = {
    "A": {
        "package": "P-101",
        "name": "Electronics"
    },
    "B": {
        "package": "P-102",
        "name": "General Goods"
    },
    "C": {
        "package": "P-103",
        "name": "Fragile Items"
    }
}

ACTION_NAMES = {
    0: "⬆ UP",
    1: "⬇ DOWN",
    2: "⬅ LEFT",
    3: "➡ RIGHT"
}

MOVES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}


# =========================================================
# SESSION STATE
# =========================================================

if "env" not in st.session_state:
    st.session_state.env = Warehouse()

if "agent" not in st.session_state:
    st.session_state.agent = DQNAgent(
        state_size=15,
        action_size=4
    )

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if "running" not in st.session_state:
    st.session_state.running = False

if "selected_zone" not in st.session_state:
    st.session_state.selected_zone = "B"

if "events" not in st.session_state:
    st.session_state.events = []

if "reward" not in st.session_state:
    st.session_state.reward = 0

if "collisions" not in st.session_state:
    st.session_state.collisions = 0

if "obstacle_mode" not in st.session_state:
    st.session_state.obstacle_mode = False

if "initialized" not in st.session_state:
    st.session_state.initialized = False

if "previous_position" not in st.session_state:
    st.session_state.previous_position = None


env = st.session_state.env
agent = st.session_state.agent


# =========================================================
# LOAD MODEL
# =========================================================

if not st.session_state.model_loaded:

    try:

        agent.load_model(MODEL_PATH)

        st.session_state.model_loaded = True

    except Exception as e:

        st.error(
            f"Could not load {MODEL_PATH}: {e}"
        )

        st.stop()


# =========================================================
# INITIAL TASK
# =========================================================

if not st.session_state.initialized:

    zone = st.session_state.selected_zone

    env.set_delivery_task(
        package_id=ZONE_INFO[zone]["package"],
        target_zone=zone
    )

    st.session_state.initialized = True


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_new_position(position, action):

    dr, dc = MOVES[action]

    return [
        position[0] + dr,
        position[1] + dc
    ]


def is_valid_position(position):

    r, c = position

    if (
        r < 0
        or r >= env.size
        or c < 0
        or c >= env.size
    ):
        return False

    if env.is_obstacle(position):
        return False

    return True


def get_distance(position, target):

    return (
        abs(position[0] - target[0])
        +
        abs(position[1] - target[1])
    )


def choose_ai_action():

    state = env.get_state()

    q_values = np.asarray(
        agent.get_q_values(state),
        dtype=np.float32
    )

    ranked_actions = list(
        np.argsort(q_values)[::-1]
    )

    valid_actions = [
        int(a)
        for a in ranked_actions
        if is_valid_position(
            get_new_position(
                env.robot,
                int(a)
            )
        )
    ]

    if not valid_actions:

        return int(ranked_actions[0]), q_values

    # -----------------------------------------------------
    # CURRENT OBJECTIVE
    # -----------------------------------------------------

    if not env.package_picked:

        target = list(env.package)

    else:

        target = env.get_target_position()

    current_distance = get_distance(
        env.robot,
        target
    )

    # -----------------------------------------------------
    # FIRST: DQN PREFERENCE + PROGRESS
    # -----------------------------------------------------

    improving_actions = []

    for action in valid_actions:

        new_position = get_new_position(
            env.robot,
            action
        )

        new_distance = get_distance(
            new_position,
            target
        )

        if new_distance < current_distance:

            improving_actions.append(
                action
            )

    # If there is a move that gets closer,
    # choose the highest-Q one among those.
    if improving_actions:

        improving_actions.sort(
            key=lambda a: q_values[a],
            reverse=True
        )

        return improving_actions[0], q_values

    # -----------------------------------------------------
    # IF DIRECT PROGRESS IS BLOCKED
    # -----------------------------------------------------

    # Let DQN decide how to navigate around
    # the obstacle.
    return valid_actions[0], q_values


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">🤖 AI Warehouse Control System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Reinforcement Learning based package pickup, navigation and delivery'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🎯 Delivery Task")

    zone = st.selectbox(
        "Select Destination Zone",
        ["A", "B", "C"],
        index=["A", "B", "C"].index(
            st.session_state.selected_zone
        ),
        disabled=st.session_state.running
    )

    if zone != st.session_state.selected_zone:

        st.session_state.selected_zone = zone

        env.set_delivery_task(
            package_id=ZONE_INFO[zone]["package"],
            target_zone=zone
        )

        st.session_state.running = False
        st.session_state.reward = 0
        st.session_state.collisions = 0
        st.session_state.events = []
        st.session_state.previous_position = None

        st.rerun()

    st.info(
        f"""
**Package:** {ZONE_INFO[zone]["package"]}

**Destination:** Zone {zone}

**Type:** {ZONE_INFO[zone]["name"]}
"""
    )

    st.divider()

    st.header("🚦 Robot Control")

    if not st.session_state.running:

        if st.button(
            "▶ Start AI Delivery",
            use_container_width=True
        ):

            st.session_state.running = True

            st.session_state.events.append(
                "🤖 AI delivery started"
            )

            st.rerun()

    else:

        if st.button(
            "⏹ Stop Robot",
            use_container_width=True
        ):

            st.session_state.running = False

            st.session_state.events.append(
                "⏹ Robot stopped"
            )

            st.rerun()

    if st.button(
        "🔄 Reset Delivery",
        use_container_width=True
    ):

        env.set_delivery_task(
            package_id=ZONE_INFO[
                st.session_state.selected_zone
            ]["package"],
            target_zone=st.session_state.selected_zone
        )

        st.session_state.running = False
        st.session_state.reward = 0
        st.session_state.collisions = 0
        st.session_state.events = []
        st.session_state.obstacle_mode = False
        st.session_state.previous_position = None

        st.rerun()

    st.divider()

    st.header("🧱 Obstacle Control")

    if not st.session_state.obstacle_mode:

        if st.button(
            "➕ Place Obstacle",
            use_container_width=True
        ):

            st.session_state.obstacle_mode = True
            st.rerun()

    else:

        st.warning(
            "Click an empty warehouse cell."
        )

        if st.button(
            "❌ Cancel",
            use_container_width=True
        ):

            st.session_state.obstacle_mode = False
            st.rerun()


# =========================================================
# TOP METRICS
# =========================================================

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric("Robot", "R-01")

with m2:
    st.metric("Package", env.package_id)

with m3:
    st.metric(
        "Destination",
        f"Zone {env.target_zone}"
    )

with m4:
    st.metric("Steps", env.steps)

with m5:
    st.metric(
        "Collisions",
        st.session_state.collisions
    )


# =========================================================
# STATUS
# =========================================================

if env.delivered:

    st.markdown(
        '<div class="status" style="background:#dcfce7;'
        'color:#166534;">'
        '✅ DELIVERY SUCCESSFUL — Package reached the correct zone.'
        '</div>',
        unsafe_allow_html=True
    )

elif st.session_state.running:

    st.markdown(
        '<div class="status" style="background:#dbeafe;'
        'color:#1e40af;">'
        '🤖 AI ROBOT ACTIVE — DQN navigation in progress.'
        '</div>',
        unsafe_allow_html=True
    )

elif env.package_picked:

    st.markdown(
        '<div class="status" style="background:#fef3c7;'
        'color:#92400e;">'
        '📦 PACKAGE PICKED UP — Robot is carrying the package.'
        '</div>',
        unsafe_allow_html=True
    )

else:

    st.markdown(
        '<div class="status" style="background:#f1f5f9;'
        'color:#334155;">'
        '⏳ READY — Start the AI delivery.'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# MAIN AREA
# =========================================================

left, right = st.columns(
    [1.5, 1]
)


# =========================================================
# WAREHOUSE GRID
# =========================================================

with left:

    st.markdown(
        '<div class="section">🏭 Warehouse Floor</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "🤖 Robot → 📦 Package → Selected Zone"
    )

    for r in range(env.size):

        cols = st.columns(
            env.size,
            gap="small"
        )

        for c in range(env.size):

            with cols[c]:

                position = [r, c]

                symbol = "·"

                background = "#ffffff"
                border = "#94a3b8"

                if env.robot == position:

                    if env.package_picked:
                        symbol = "🤖📦"
                    else:
                        symbol = "🤖"

                    background = "#dbeafe"
                    border = "#2563eb"

                elif (
                    env.package == position
                    and not env.package_picked
                ):

                    symbol = "📦"

                    background = "#fef3c7"
                    border = "#f59e0b"

                elif env.is_obstacle(
                    position
                ):

                    symbol = "🧱"

                    background = "#fee2e2"
                    border = "#dc2626"

                elif position == env.zones["A"]:

                    symbol = "A"

                    background = "#dbeafe"
                    border = "#2563eb"

                elif position == env.zones["B"]:

                    symbol = "B"

                    background = "#dcfce7"
                    border = "#16a34a"

                elif position == env.zones["C"]:

                    symbol = "C"

                    background = "#fef9c3"
                    border = "#ca8a04"

                st.markdown(
                    f"""
                    <div style="
                        height:48px;
                        border:2px solid {border};
                        border-radius:8px;
                        background:{background};
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        font-size:21px;
                        font-weight:700;
                    ">
                    {symbol}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # -----------------------------------------
                # OBSTACLE BUTTON
                # -----------------------------------------

                if st.session_state.obstacle_mode:

                    if st.button(
                        " ",
                        key=f"obs_{r}_{c}",
                        use_container_width=True
                    ):

                        occupied = (
                            env.robot == position
                            or env.package == position
                            or position in env.zones.values()
                        )

                        if not occupied:

                            if env.is_obstacle(
                                position
                            ):

                                env.remove_obstacle(
                                    position
                                )

                                msg = (
                                    f"🧱 Obstacle removed "
                                    f"at {position}"
                                )

                            else:

                                env.add_obstacle(
                                    position
                                )

                                msg = (
                                    f"🧱 Obstacle placed "
                                    f"at {position}"
                                )

                            st.session_state.events.append(
                                msg
                            )

                        else:

                            st.session_state.events.append(
                                "⚠️ Cannot place obstacle there"
                            )

                        st.session_state.obstacle_mode = False

                        st.rerun()


# =========================================================
# RIGHT PANEL
# =========================================================

with right:

    st.markdown(
        '<div class="section">📋 Delivery Task</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"**Robot:** R-01"
    )

    st.write(
        f"**Package:** {env.package_id}"
    )

    st.write(
        f"**Destination:** Zone {env.target_zone}"
    )

    st.write(
        f"**Type:** "
        f"{ZONE_INFO[env.target_zone]['name']}"
    )

    st.write(
        f"**Package Picked:** "
        f"{'YES' if env.package_picked else 'NO'}"
    )

    st.write(
        f"**Delivered:** "
        f"{'YES' if env.delivered else 'NO'}"
    )

    st.divider()

    # =====================================================
    # AI DECISION
    # =====================================================

    st.markdown(
        '<div class="section">🧠 AI Decision Center</div>',
        unsafe_allow_html=True
    )

    state = env.get_state()

    q_values = np.asarray(
        agent.get_q_values(state),
        dtype=np.float32
    )

    ai_action = int(
        np.argmax(q_values)
    )

    st.write(
        f"**DQN Preferred Action:** "
        f"{ACTION_NAMES[ai_action]}"
    )

    q1, q2, q3, q4 = st.columns(4)

    with q1:
        st.metric(
            "UP",
            f"{q_values[0]:.2f}"
        )

    with q2:
        st.metric(
            "DOWN",
            f"{q_values[1]:.2f}"
        )

    with q3:
        st.metric(
            "LEFT",
            f"{q_values[2]:.2f}"
        )

    with q4:
        st.metric(
            "RIGHT",
            f"{q_values[3]:.2f}"
        )

    st.divider()

    # =====================================================
    # LIVE INFORMATION
    # =====================================================

    st.markdown(
        '<div class="section">📊 Live Performance</div>',
        unsafe_allow_html=True
    )

    st.metric(
        "Total Reward",
        round(
            st.session_state.reward,
            2
        )
    )

    st.metric(
        "Robot Position",
        str(env.robot)
    )

    if not env.package_picked:

        objective = env.package

        objective_name = (
            f"Package {env.package_id}"
        )

    else:

        objective = env.get_target_position()

        objective_name = (
            f"Zone {env.target_zone}"
        )

    st.write(
        f"**Current Objective:** {objective_name}"
    )

    st.write(
        f"**Objective Position:** {objective}"
    )

    st.divider()

    # =====================================================
    # EVENT LOG
    # =====================================================

    st.markdown(
        '<div class="section">📡 Event Monitor</div>',
        unsafe_allow_html=True
    )

    if st.session_state.events:

        for event in reversed(
            st.session_state.events[-8:]
        ):

            st.write(
                "• " + event
            )

    else:

        st.info(
            "Waiting for AI activity..."
        )


# =========================================================
# ZONE LEGEND
# =========================================================

st.divider()

st.markdown(
    '<div class="section">📍 Warehouse Zones</div>',
    unsafe_allow_html=True
)

z1, z2, z3 = st.columns(3)

with z1:

    st.info(
        "🔵 **ZONE A — Electronics**\n\n"
        "P-101"
    )

with z2:

    st.success(
        "🟢 **ZONE B — General Goods**\n\n"
        "P-102"
    )

with z3:

    st.warning(
        "🟡 **ZONE C — Fragile Items**\n\n"
        "P-103"
    )


# =========================================================
# AI MOVEMENT LOOP
# =========================================================

if st.session_state.running:

    # -----------------------------------------------------
    # DELIVERY COMPLETE
    # -----------------------------------------------------

    if env.delivered:

        st.session_state.running = False

        st.rerun()

    # -----------------------------------------------------
    # MAXIMUM STEPS
    # -----------------------------------------------------

    if env.steps >= env.max_steps:

        st.session_state.running = False

        st.session_state.events.append(
            "⚠️ Maximum delivery steps reached"
        )

        st.rerun()

    # =====================================================
    # CHOOSE ACTION
    # =====================================================

    chosen_action, q_values = choose_ai_action()

    # =====================================================
    # OBJECTIVE
    # =====================================================

    if not env.package_picked:

        objective = env.package

    else:

        objective = env.get_target_position()

    old_position = list(
        env.robot
    )

    old_distance = get_distance(
        old_position,
        objective
    )

    # =====================================================
    # EXECUTE ACTION
    # =====================================================

    next_state, reward, done, info = env.step(
        chosen_action
    )

    st.session_state.reward += reward

    new_position = list(
        env.robot
    )

    new_distance = get_distance(
        new_position,
        objective
    )

    # =====================================================
    # EVENT
    # =====================================================

    event = info.get(
        "event",
        f"AI performed {ACTION_NAMES[chosen_action]}"
    )

    st.session_state.events.append(
        event
    )

    # =====================================================
    # MOVEMENT LOG
    # =====================================================

    if old_position != new_position:

        st.session_state.events.append(
            f"🤖 Robot moved "
            f"{old_position} → {new_position}"
        )

    # =====================================================
    # COLLISION
    # =====================================================

    if info.get(
        "collision",
        False
    ):

        st.session_state.collisions += 1

        st.session_state.events.append(
            f"⚠️ Collision penalty: {reward}"
        )

    # =====================================================
    # PICKUP
    # =====================================================

    if (
        info.get(
            "package_picked",
            False
        )
    ):

        st.session_state.events.append(
            f"📦 {env.package_id} picked up"
        )

    # =====================================================
    # DELIVERY
    # =====================================================

    if info.get(
        "delivered",
        False
    ):

        st.session_state.events.append(
            f"🎉 {env.package_id} delivered "
            f"to Zone {env.target_zone}"
        )

        st.session_state.running = False

    # =====================================================
    # SLOW DOWN ONLY SLIGHTLY
    # =====================================================

    time.sleep(0.30)

    st.rerun()