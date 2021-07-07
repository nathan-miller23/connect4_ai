from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.tune.registry import register_env
from ray.tune.logger import UnifiedLogger
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from kaggle_environments import make
from datetime import datetime
import gym, copy, os, dill, logging, ray, tempfile
import numpy as np


timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

def get_base_env(params):
    return make('connectx', params)

class ConnectFourMultiAgent(MultiAgentEnv):
    """
    Class used to wrap ConectFour environment in an Rllib compatible multi-agent environment
    """

    # List of all agent types currently supported
    supported_agents = ['ppo']

    # Flags
    ACTIVE = 'ACTIVE'
    INACTIVE = 'INACTIVE'
    DONE = 'DONE'

    # Default environment params used for creation
    DEFAULT_CONFIG = {
        # To be passed into 'make' call
        "env_params" : {
        },
        # To be passed into OvercookedMultiAgent constructor
        "multi_agent_params" : {
        }
    }

    def __init__(self, base_env, buffer_size=4):
        """
        base_env: Kaggle Environment
        """
        self.base_env = base_env
        self.buffer_size=buffer_size
        self.p1_buffer = []
        self.p2_buffer = []
        self._setup_observation_space()
        self.action_space = gym.spaces.Discrete(self.config['columns'])
        self.reset()

    @classmethod
    def from_base_env_params(cls, mdp_params, env_params={"horizon" : 400}, multi_agent_params={}):
        base_env = get_base_env(mdp_params, env_params)
        return cls(base_env, **multi_agent_params)

    @property
    def config(self):
        return self.base_env.configuration

    @property
    def board_shape(self):
        return (self.config['rows'], self.config['columns'])

    def default_featurize_fn(self, observation, player=1):
        board = np.array(observation['board']).reshape(self.config['rows'], self.config['columns'])
        timestep = observation['step']

        other_player = 2 if player == 1 else 1

        my_tiles = (board == player).astype(int)
        their_tiles = (board == other_player).astype(int)

        my_obs = np.array([my_tiles, their_tiles]).transpose(1, 2, 0)

        return my_obs
    
    def _validate_schedule(self, schedule):
        timesteps = [p[0] for p in schedule]
        values = [p[1] for p in schedule]

        assert len(schedule) >= 2, "Need at least 2 points to linearly interpolate schedule"
        assert schedule[0][0] == 0, "Schedule must start at timestep 0"
        assert all([t >=0 for t in timesteps]), "All timesteps in schedule must be non-negative"
        assert all([v >=0 and v <= 1 for v in values]), "All values in schedule must be between 0 and 1"
        assert sorted(timesteps) == timesteps, "Timesteps must be in increasing order in schedule"

        # To ensure we flatline after passing last timestep
        if (schedule[-1][0] < float('inf')):
            schedule.append((float('inf'), schedule[-1][1]))

    def _setup_observation_space(self):
        single_frame_shape = self._get_featurize_fn(1)(self.base_env.state[0]['observation']).shape
        obs_shape = list(single_frame_shape)
        obs_shape[-1] *= self.buffer_size
        low = np.ones(obs_shape) * 0
        high = np.ones(obs_shape) * 1
        self.observation_space = gym.spaces.Box(np.float32(low), np.float32(high), dtype=np.float32)

        for _ in range(self.buffer_size):
            self.p1_buffer.append(np.zeros(single_frame_shape))
            self.p2_buffer.append(np.zeros(single_frame_shape))

    def _get_featurize_fn(self, player):
        return lambda observation : self.default_featurize_fn(observation, player)

    def _get_obs_dict(self, state):
        p1_obs, p1_status = state[0]['observation'], state[0]['status']
        p2_obs, p2_status = state[1]['observation'], state[1]['status']


        obs_dict = {}
        if p1_status == self.ACTIVE:
            p1_obs = self._get_featurize_fn(1)(p1_obs)
            self.p1_buffer.pop(0)
            self.p1_buffer.append(p1_obs)
            obs = np.concatenate(self.p1_buffer, axis=-1)
            obs_dict[self.curr_agents[0]] = obs

        if p2_status == self.ACTIVE:
            p2_obs = self._get_featurize_fn(2)(p2_obs)
            self.p1_buffer.pop(0)
            self.p2_buffer.append(p2_obs)
            obs = np.concatenate(self.p2_buffer, axis=-1)
            obs_dict[self.curr_agents[1]] = obs
        
        return obs_dict

    def _get_reward_dict(self, state):
        p1_reward, p1_status = state[0]['reward'], state[0]['status']
        p2_reward, p2_status = state[1]['reward'], state[1]['status']

        reward_dict = {}
        if p1_status == self.ACTIVE:
            reward_dict[self.curr_agents[0]] = p1_reward
        if p2_status == self.ACTIVE:
            reward_dict[self.curr_agents[1]] = p2_reward

        return reward_dict

    def _get_infos_dict(self, state):
        p1_info, p1_status = state[0]['info'], state[0]['status']
        p2_info, p2_status = state[1]['info'], state[1]['status']

        info_dict = {}
        if p1_status == self.ACTIVE:
            info_dict[self.curr_agents[0]] = p1_info
        if p2_status == self.ACTIVE:
            info_dict[self.curr_agents[1]] = p2_info

        return info_dict

    def _get_done_dict(self, state):
        p1_status = state[0]['status']
        p2_status = state[1]['status']

        done = p1_status == self.DONE or p2_status == self.DONE

        return { '__all__' : done }

    def get_transition_data(self, state):
        obs = self._get_obs_dict(state)
        reward = self._get_reward_dict(state)
        info = self._get_infos_dict(state)
        done = self._get_done_dict(state)

        return obs, reward, done, info
    
    def _populate_agents(self):
        # Assign indices for ppo and non-ppo (ie bc, bc_opt, etc) agent
        # Note: we always include at least one ppo agent (i.e. bc_sp not supported for simplicity)
        # Note: We assume num players = 2
        agents = ['ppo', 'ppo']

        # Ensure agent names are unique
        agents[0] = agents[0] + '_0'
        agents[1] = agents[1] + '_1'
        
        return agents

    def _anneal(self, start_v, curr_t, end_t, end_v=0, start_t=0):
        if end_t == 0:
            # No annealing if horizon is zero
            return start_v
        else:
            off_t = curr_t - start_t
            # Calculate the new value based on linear annealing formula
            fraction = max(1 - float(off_t) / (end_t - start_t), 0)
            return fraction * start_v + (1 - fraction) * end_v

    def _anneal_from_schedule(self, timestep, schedule):
        p_0 = schedule[0]
        p_1 = schedule[1]
        i = 2
        while timestep > p_1[0] and i < len(schedule):
            p_0 = p_1
            p_1 = schedule[i]
            i += 1
        start_t, start_v = p_0
        end_t, end_v = p_1
        new_factor = self._anneal(start_v, timestep, end_t, end_v, start_t)
        return new_factor


    def step(self, action_dict):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format
        
        returns:
            observation (dict): formatted to be standard input for self.agent_idx's policy
            rewards (dict): by-agent timestep reward
            dones (dict): by-agent done flags
            infos (dict): by-agent info dictionaries
        """
        action = [action_dict.get(self.curr_agents[0], 0), action_dict.get(self.curr_agents[1], 0)]
        assert all(self.action_space.contains(a) for a in action), "%r (%s) invalid"%(action, type(action))
        self.base_env.step(action)

        obs, rewards, dones, infos = self.get_transition_data(self.base_env.state)
        return obs, rewards, dones, infos

    def reset(self):
        """
        """
        self.base_env.reset()
        self.curr_agents = self._populate_agents()
        return self._get_obs_dict(self.base_env.state)
    
    @classmethod
    def from_config(cls, env_config):
        """
        Factory method for generating environments in style with rllib guidlines

        env_config (dict):  Must contain keys 'mdp_params', 'env_params' and 'multi_agent_params', the last of which
                            gets fed into the OvercookedMultiAgent constuctor

        Returns:
            OvercookedMultiAgent instance specified by env_config params
        """
        assert env_config and "env_params" in env_config and "multi_agent_params" in env_config

        # "start_state_fn" and "horizon"
        env_params = env_config["env_params"]
        # "reward_shaping_factor"
        multi_agent_params = env_config["multi_agent_params"]
        base_env = get_base_env(env_params)

        return cls(base_env, **multi_agent_params)




def gen_trainer_from_params(params):
    # All ray environment set-up
    if not ray.is_initialized():
        init_params = {
            "ignore_reinit_error" : True,
            "_temp_dir" : params['ray_params']['temp_dir'],
            "log_to_driver" : params['verbose'],
            "logging_level" : logging.INFO if params['verbose'] else logging.CRITICAL
        }
        ray.init(**init_params)
    register_env("overcooked_multi_agent", params['ray_params']['env_creator'])
    ModelCatalog.register_custom_model(params['ray_params']['custom_model_id'], params['ray_params']['custom_model_cls'])

    # Parse params
    training_params = params['training_params']
    environment_params = params['environment_params']
    policy_params = params['policy_params']
    multi_agent_params = params['environment_params']['multi_agent_params']

    env = ConnectFourMultiAgent.from_config(environment_params)

    import tensorflow as tf
    if tf.executing_eagerly():
        print("AHHHHHH")

    # Returns a properly formatted policy tuple to be passed into ppotrainer config
    def gen_policy(policy_type="ppo"):
        # supported policy types thus far
        assert policy_type in ConnectFourMultiAgent.supported_agents

        curr_policy_params = policy_params[policy_type]
        policy_cls = curr_policy_params['cls']
        policy_config = curr_policy_params['config']

        policy_observation_space = None
        if policy_type == 'ppo' or policy_type == 'ensemble_ppo':
            policy_observation_space = env.observation_space
        return (policy_cls, policy_observation_space, env.action_space, policy_config)

    # Rllib compatible way of setting the directory we store agent checkpoints in
    logdir_prefix = params['logdir_prefix'] if 'logdir_prefix' in params else "{0}_{1}_{2}".format(params["experiment_name"], params['training_params']['seed'], timestr)
    def custom_logger_creator(config):
        """
        Creates a Unified logger that stores results in <params['results_dir']>/<params["experiment_name"]>_<seed>_<timestamp>
        """
        results_dir = params['results_dir']
        if not os.path.exists(results_dir):
            try:
                os.makedirs(results_dir)
            except Exception as e:
                print("error creating custom logging dir: {}. Falling back to default logdir {}".format(str(e), DEFAULT_RESULTS_DIR))
                results_dir = DEFAULT_RESULTS_DIR
        logdir = params['logdir'] if 'logdir' in params else tempfile.mkdtemp(
            prefix=logdir_prefix, dir=results_dir)
        logger = UnifiedLogger(config, logdir, loggers=None)
        return logger

    # Create rllib compatible multi-agent config based on params
    multi_agent_config = {}
    all_policies = set(['ppo'])
    multi_agent_config['policies'] = { policy : gen_policy(policy) for policy in all_policies }

    def select_policy(agent_id):
        agent_type = '_'.join(agent_id.split('_')[:-1])
        assert agent_type in ConnectFourMultiAgent.supported_agents
        return agent_type

    multi_agent_config['policy_mapping_fn'] = select_policy
    multi_agent_config['policies_to_train'] = ['ppo']

    trainer = PPOTrainer(env="overcooked_multi_agent", config={
        "multiagent": multi_agent_config,
        "env_config" : environment_params,
        **training_params
    }, logger_creator=custom_logger_creator)
    return trainer






### Serialization ###

def save_trainer_config(params, save_path):
    # Save params used to create trainer in /path/to/checkpoint_dir/config.pkl
    config = copy.deepcopy(params)
    config_path = os.path.join(os.path.dirname(save_path), "config.pkl")

    # Note that we use dill (not pickle) here because it supports function serialization
    with open(config_path, "wb") as f:
        dill.dump(config, f)

    return config_path

def save_trainer(trainer, params, path=None):
    """
    Saves a serialized trainer checkpoint at `path`. If none provided, the default path is
    ~/ray_results/<experiment_results_dir>/checkpoint_<i>/checkpoint-<i>

    Note that `params` should follow the same schema as the dict passed into `gen_trainer_from_params`
    """
    # Save trainer
    save_path = trainer.save(path)

    # Save params
    save_trainer_config(params, save_path)
    return save_path

def load_trainer_config(save_path, **params_to_override):
    checkpoint_dir = os.path.dirname(save_path)
    experiment_dir = os.path.dirname(checkpoint_dir)
    config_path = os.path.join(checkpoint_dir, "config.pkl")
    with open(config_path, "rb") as f:
        # We use dill (instead of pickle) here because we must deserialize functions
        config = dill.load(f)
    
    # Override this param to lower overhead in trainer creation
    config['training_params']['num_workers'] = 0
    config['logdir'] = experiment_dir

    # Override any other params specified by user
    config = override_dict(config, **params_to_override)

    return config

def load_trainer(save_path, **params_to_override):
    """
    Returns a ray compatible trainer object that was previously saved at `save_path` by a call to `save_trainer`
    Note that `save_path` is the full path to the checkpoint FILE, not the checkpoint directory
    """
    # Ensure tf is executing in graph mode
    import tensorflow as tf
    if tf.executing_eagerly():
        tf.compat.v1.disable_eager_execution()
    
    # Read in params used to create trainer
    config = load_trainer_config(save_path, **params_to_override)

    # Get un-trained trainer object with proper config
    trainer = gen_trainer_from_params(config)

    # Load weights into dummy object
    trainer.restore(save_path)
    return trainer

def override_dict(map, **args_to_override):
    map = copy.deepcopy(map)
    for arg, val in args_to_override.items():
        updated = recursive_dict_update(map, arg, val)
        if not updated:
            print("WARNING, no value for specified argument {} found in schema. Adding as top level parameter".format(arg))
    return map

def recursive_dict_update(map, key, value):
    if type(map) != dict:
        return False
    if key in map:
        map[key] = value
        return True
    return any([recursive_dict_update(child, key, value) for child in map.values()])