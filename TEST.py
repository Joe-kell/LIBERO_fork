
'''
Below scrip to identify key environment properties within Libero
'''



import os
import libero.libero
from libero.libero import benchmark, get_libero_path, get_default_path_dict
from libero.libero.envs import OffScreenRenderEnv, ControlEnv
import skrl
from skrl.agents.torch import ppo as ppo


benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_10" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()


# retrieve a specific task
task_id = 0
task = task_suite.get_task(task_id)
task_name = task.name
task_description = task.language


#Finds the name of tasks based on the self.tasks list, picks a task, which has many attributes, uses these to define quantities

#BELOW RETRIEVES THE TASK BDDL FILE - this means the env can be generated to match the task
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
    f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")


agent = ppo

agent_cfg = PPO_DEFAULT_CONFIG
print(agent_cfg)
exit()

# step over the environment
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128,
    "has_renderer": True,
    "has_offscreen_renderer": True
}
# env = OffScreenRenderEnv(**env_args)

env = ControlEnv(**env_args)
env.seed(0)
env.reset()
init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
init_state_id = 0
env.set_init_state(init_states[init_state_id])

dummy_action = [0.02] * 7
for step in range(1000):
    obs, reward, done, info = env.step(dummy_action)
    env.env.render()
env.close()