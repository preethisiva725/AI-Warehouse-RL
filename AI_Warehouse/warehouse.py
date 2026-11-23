import numpy as np


class Warehouse:

    def __init__(self, size=10):

        self.size = size

        # =================================================
        # DESTINATION ZONES
        # =================================================

        self.zones = {
            "A": [1, 8],
            "B": [5, 8],
            "C": [8, 8]
        }

        self.zone_names = {
            "A": "Electronics",
            "B": "General Goods",
            "C": "Fragile Items"
        }

        # =================================================
        # PACKAGE MAPPING
        # =================================================

        self.package_options = {
            "P-101": "A",
            "P-102": "B",
            "P-103": "C"
        }

        # =================================================
        # FIXED OBSTACLES
        # =================================================

        self.fixed_obstacles = [
            [2, 3],
            [3, 3],
            [4, 3],
            [6, 4],
            [6, 5],
            [6, 6],
            [7, 6],
            [8, 6]
        ]

        self.max_steps = 100

        self.reset(
            target_zone="B",
            package_id="P-102"
        )

    # =====================================================
    # RESET
    # =====================================================

    def reset(
        self,
        target_zone=None,
        package_id=None
    ):

        if target_zone is None:

            target_zone = np.random.choice(
                ["A", "B", "C"]
            )

        if package_id is None:

            package_id = {
                "A": "P-101",
                "B": "P-102",
                "C": "P-103"
            }[target_zone]

        self.target_zone = target_zone
        self.package_id = package_id

        # Robot starts here
        self.robot = [0, 0]

        # Package starts here
        self.package = [1, 1]

        self.package_picked = False
        self.delivered = False

        self.steps = 0
        self.total_reward = 0

        self.obstacles = [
            list(x)
            for x in self.fixed_obstacles
        ]

        return self.get_state()

    # =====================================================
    # ASSIGN TASK
    # =====================================================

    def set_delivery_task(
        self,
        package_id,
        target_zone
    ):

        self.reset(
            target_zone=target_zone,
            package_id=package_id
        )

    # =====================================================
    # TARGET
    # =====================================================

    def get_target_position(self):

        return list(
            self.zones[self.target_zone]
        )

    # =====================================================
    # DISTANCE
    # =====================================================

    def distance(self, a, b):

        return (
            abs(a[0] - b[0])
            +
            abs(a[1] - b[1])
        )

    # =====================================================
    # OBSTACLE
    # =====================================================

    def is_obstacle(self, position):

        return list(position) in self.obstacles

    def add_obstacle(self, position):

        position = list(position)

        if (
            0 <= position[0] < self.size
            and
            0 <= position[1] < self.size
            and
            position not in self.obstacles
        ):

            self.obstacles.append(
                position
            )

    def remove_obstacle(self, position):

        position = list(position)

        if position in self.obstacles:

            self.obstacles.remove(
                position
            )

    # =====================================================
    # VALID MOVES
    # =====================================================

    def get_valid_actions(self):

        valid = []

        moves = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }

        for action, (dr, dc) in moves.items():

            nr = self.robot[0] + dr
            nc = self.robot[1] + dc

            position = [nr, nc]

            if (
                nr < 0
                or nr >= self.size
                or nc < 0
                or nc >= self.size
            ):
                continue

            if self.is_obstacle(position):
                continue

            valid.append(action)

        return valid

    # =====================================================
    # STATE
    # =====================================================

    def get_state(self):

        r, c = self.robot

        p_r, p_c = self.package

        t_r, t_c = self.get_target_position()

        state = [

            r / 9.0,
            c / 9.0,

            p_r / 9.0,
            p_c / 9.0,

            t_r / 9.0,
            t_c / 9.0,

            float(self.package_picked)
        ]

        # 8 surrounding cells

        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1)
        ]

        for dr, dc in directions:

            nr = r + dr
            nc = c + dc

            if (
                nr < 0
                or nr >= self.size
                or nc < 0
                or nc >= self.size
            ):

                state.append(-1.0)

            elif self.is_obstacle(
                [nr, nc]
            ):

                state.append(-1.0)

            else:

                state.append(0.0)

        return np.array(
            state,
            dtype=np.float32
        )

    # =====================================================
    # STEP
    # =====================================================

    def step(self, action):

        self.steps += 1

        old_position = list(
            self.robot
        )

        moves = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }

        dr, dc = moves[
            int(action)
        ]

        new_position = [
            self.robot[0] + dr,
            self.robot[1] + dc
        ]

        reward = -1

        collision = False
        collision_type = None

        event = (
            f"AI attempted "
            f"{old_position} → {new_position}"
        )

        # =================================================
        # BOUNDARY
        # =================================================

        outside = (
            new_position[0] < 0
            or new_position[0] >= self.size
            or new_position[1] < 0
            or new_position[1] >= self.size
        )

        if outside:

            reward = -10

            collision = True

            collision_type = "Boundary"

            event = (
                "⚠️ Robot reached warehouse boundary"
            )

        # =================================================
        # OBSTACLE
        # =================================================

        elif self.is_obstacle(
            new_position
        ):

            reward = -15

            collision = True

            collision_type = "Obstacle"

            event = (
                f"⚠️ Robot hit obstacle "
                f"at {new_position}"
            )

        # =================================================
        # VALID MOVEMENT
        # =================================================

        else:

            self.robot = new_position

            # ---------------------------------------------
            # GOING TO PACKAGE
            # ---------------------------------------------

            if not self.package_picked:

                old_distance = self.distance(
                    old_position,
                    self.package
                )

                new_distance = self.distance(
                    self.robot,
                    self.package
                )

                if new_distance < old_distance:

                    reward += 3

                elif new_distance > old_distance:

                    reward -= 2

                else:

                    reward -= 1

                # -----------------------------------------
                # PICKUP
                # -----------------------------------------

                if (
                    self.robot
                    ==
                    self.package
                ):

                    self.package_picked = True

                    reward += 30

                    event = (
                        f"📦 {self.package_id} "
                        f"picked up by Robot R-01"
                    )

            # ---------------------------------------------
            # CARRYING PACKAGE
            # ---------------------------------------------

            else:

                target = (
                    self.get_target_position()
                )

                old_distance = self.distance(
                    old_position,
                    target
                )

                new_distance = self.distance(
                    self.robot,
                    target
                )

                if new_distance < old_distance:

                    reward += 4

                elif new_distance > old_distance:

                    reward -= 2

                else:

                    reward -= 1

                # -----------------------------------------
                # DELIVERY
                # -----------------------------------------

                if (
                    self.robot
                    ==
                    target
                ):

                    self.delivered = True

                    reward += 100

                    event = (
                        f"✅ {self.package_id} "
                        f"delivered to Zone "
                        f"{self.target_zone}"
                    )

        # =================================================
        # DONE
        # =================================================

        done = False

        if self.delivered:

            done = True

        elif self.steps >= self.max_steps:

            done = True

            event = (
                "Maximum delivery steps reached"
            )

        self.total_reward += reward

        info = {

            "collision": collision,

            "collision_type":
                collision_type,

            "event":
                event,

            "position":
                list(self.robot),

            "package":
                list(self.package),

            "package_picked":
                self.package_picked,

            "target_zone":
                self.target_zone,

            "target_position":
                self.get_target_position(),

            "delivered":
                self.delivered,

            "steps":
                self.steps,

            "obstacle_count":
                len(self.obstacles)
        }

        return (
            self.get_state(),
            reward,
            done,
            info
        )

    # =====================================================
    # TEXT RENDER
    # =====================================================

    def render(self):

        print("\nWarehouse\n")

        for r in range(
            self.size
        ):

            row = []

            for c in range(
                self.size
            ):

                position = [
                    r,
                    c
                ]

                if position == self.robot:

                    if self.package_picked:

                        symbol = "🤖📦"

                    else:

                        symbol = "🤖"

                elif (
                    position == self.package
                    and not self.package_picked
                ):

                    symbol = "📦"

                elif self.is_obstacle(
                    position
                ):

                    symbol = "🧱"

                elif position == self.zones["A"]:

                    symbol = "A"

                elif position == self.zones["B"]:

                    symbol = "B"

                elif position == self.zones["C"]:

                    symbol = "C"

                else:

                    symbol = "."

                row.append(symbol)

            print(
                " ".join(row)
            )