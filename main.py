from envs.line_world.line_world import LineWorld
from rl_algorithms.dp.value_iteration import value_iteration
from rl_algorithms.dp.policy_iteration import policy_iteration

env = LineWorld()

V_vi, policy_vi = value_iteration(env)
print("Value Iteration: V =", V_vi)
print("Value Iteration: Policy =", policy_vi)

policy_pi, V_pi = policy_iteration(env)
print("Policy Iteration: Policy =", policy_pi)
print("Policy Iteration: V =", V_pi)