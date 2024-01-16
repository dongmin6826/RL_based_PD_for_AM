RL_ALGORITHM = "PPO"
N_EPISODES = 100
N_EVAL_EPISODES = 10
MAX_N_PARTS = 20

# Set low and high values for the actions of the center coordination (X, Y, Z)
ACTION_SPACE_CENTER_COOR_LOW = 0  # unit: mm
ACTION_SPACE_CENTER_COOR_HIGH = 200  # unit: mm

# Set low and high values for the actions of the cutting plane angle (X, Y, Z)
ACTION_SPACE_CUT_PLANE_ANGLE_LOW = -180  # unit: degree
ACTION_SPACE_CUT_PLANE_ANGLE_HIGH = 180  # unit: degree
