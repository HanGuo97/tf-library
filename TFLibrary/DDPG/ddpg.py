"""
Modified from
https://github.com/pemami4911/deep-rl/blob/master/ddpg/ddpg.py
"""

import warnings
import numpy as np
from namedlist import namedlist
import tensorflow_utils as tf_utils
from utils.misc_utils import ReplayBuffer
from utils.misc_utils import OrnsteinUhlenbeckActionNoise


from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

from tensorflow.python.ops import variables
from tensorflow.python.ops import gradients_impl
from tensorflow.python.training import adam as adam_ops
from tensorflow.python.training import gradient_descent as gd_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.framework import dtypes
from tensorflow.contrib import layers as contrib_layers


Experience = namedlist("Experience",
    ("State", "Action", "Reward", "NextState"))


class SingleEpisodeDDPGController(object):
    def __init__(self,
                 sess,
                 num_units,
                 num_actions,
                 batch_size,
                 learning_rate,
                 tau,
                 gamma,
                 actor_activation=math_ops.sigmoid,
                 critic_activation=None,
                 opitmizer_name="adam",
                 max_gradient_norm=5.0,
                 actor_scope=None,
                 critic_scope=None,
                 actor_target_scope=None,
                 critic_target_scope=None,
                 # noise
                 noise_mu=None,
                 # replay buffer
                 buffer_size=1000,
                 buffer_random_seed=123):
                 
        actor = ActorNetwork(
            sess=sess,
            num_units=num_units,
            num_actions=num_actions,
            batch_size=batch_size,
            learning_rate=learning_rate,
            tau=tau,
            activation=actor_activation,
            opitmizer_name=opitmizer_name,
            max_gradient_norm=max_gradient_norm,
            actor_scope=actor_scope,
            target_scope=actor_target_scope)

        critic = CriticNetwork(
            sess=sess,
            num_units=num_units,
            num_actions=num_actions,
            batch_size=batch_size,
            learning_rate=learning_rate,
            tau=tau,
            gamma=gamma,
            activation=critic_activation,
            opitmizer_name=opitmizer_name,
            max_gradient_norm=max_gradient_norm,
            critic_scope=critic_scope,
            target_scope=critic_target_scope)

        # Initialize target network weights
        sess.run(variables.global_variables_initializer())
        actor.update_target()
        critic.update_target()

        if noise_mu is None:
            noise_mu = np.zeros([num_actions])
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=noise_mu)

        # Initialize replay memory
        replay_buffer = ReplayBuffer(
            buffer_size=buffer_size,
            random_seed=buffer_random_seed)

        self._actor = actor
        self._critic = critic
        self._actor_noise = actor_noise
        self._replay_buffer = replay_buffer

        self._batch_size = batch_size

        self._last_state = None
        self._last_action = None

    def act(self, state):
        action = self._actor.predict(inputs=state)

        # save un-noisy inputs
        self._last_state = state
        self._last_action = action

        noisy_action = action + self._actor_noise()
        return noisy_action

    def update(self, new_state, observed_reward):
        actor = self._actor
        critic = self._critic
        batch_size = self._batch_size
        replay_buffer = self._replay_buffer

        squeeze = lambda X: np.squeeze(X)
        replay_buffer.add(
            s=squeeze(self._last_state),
            a=squeeze(self._last_action),
            r=observed_reward,
            # no termination
            t=False, s2=squeeze(new_state))

        # Keep adding experience to the memory until
        # there are at least minibatch size samples
        debug = None
        if replay_buffer.size() > batch_size:
            (s_batch,  # state
             a_batch,  # action
             r_batch,  # reward
             t_batch,  # termination
             s2_batch) = replay_buffer.sample_batch(batch_size)

            # Calculate targets
            # if t_batch[k]:  # if terminates
            # TD_targets.append(r_batch[k]) # no  + gamma x Q
            warnings.warn("This Code Assumes No Termination of Episodes")
            actor_targets = actor.predict_target(s2_batch)
            TD_targets, _ = critic.compute_TD_target(
                rewards=r_batch,
                next_inputs=s2_batch,
                next_actions=actor_targets)

            # update critics based on ||TD_target - Q||
            pred_Q, _ = critic.train(
                inputs=s_batch,
                actions=a_batch,
                TD_target=TD_targets)

            # what is this for?
            # ep_ave_max_q += np.amax(pred_Q)

            # Update the actor policy using the sampled gradient
            # use the newly computed action not from replay buffer
            pred_actions = actor.predict(s_batch)
            critic_grads = critic.action_gradients(
                inputs=s_batch,
                actions=pred_actions)
            actor.train(
                inputs=s_batch,
                action_gradients=critic_grads[0])

            # Update target networks
            actor.update_target()
            critic.update_target()


            experience = Experience(
                State=s_batch, Action=a_batch,
                Reward=r_batch, NextState=s2_batch)
            # print(experience)
        
            debug = {
                "experience": experience,
                "actor_targets": actor_targets,
                "TD_targets": TD_targets,
                "pred_actions": pred_actions,
                "critic_grads": critic_grads}

        return self.act(new_state), debug


class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self,
                 sess,
                 num_units,
                 num_actions,
                 batch_size,
                 learning_rate,
                 tau,
                 activation=math_ops.tanh,
                 batch_norm=True,
                 opitmizer_name="adam",
                 max_gradient_norm=5.0,
                 actor_scope=None,
                 target_scope=None):
        self._sess = sess
        self._num_units = num_units
        self._num_actions = num_actions
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._tau = tau
        self._activation = activation
        self._batch_norm = batch_norm
        self._actor_scope = actor_scope
        self._target_scope = target_scope

        if activation and not callable(activation):
            raise TypeError("Expected `activation` to be callable")
        

        # Actor Network
        (actor_inputs,
         actor_outputs,
         actor_params,
         actor_normalized_inputs) = self._build_network(actor_scope)

        # Target Network
        (target_inputs,
         target_outputs,
         target_params,
         target_normalized_inputs) = self._build_network(target_scope)

        num_trainable_vars = len(actor_params) + len(target_params)

        # Op for periodically updating target
        # network with online network weights
        # theta_target = tau x theta_target + (1-tau) theta
        target_update_op = update_target_params(
            params=actor_params, target_params=target_params, tau=tau)
        
        # This gradient will be provided by the critic network
        # define: grad(x, y): gradient of y w.r.t x
        # define: action = Actor(s | theta) + Noise

        # grad(actor, J) = E[grad(action, Critic(s,a)) x grad(actor, Actor(s))]
        # the placeholder if for the gradient of action provided from Critic
        action_gradients = array_ops.placeholder(
            dtype=dtypes.float32,
            shape=[None, num_actions],
            name="actor_gradients")
        
        # Combine the gradients here
        actor_gradients = gradients_impl.gradients(
            # grad(actor, Actor(s))
            ys=actor_outputs, xs=actor_params,
            # grad(action, Critic(s,a))
            # negate the action-value gradient
            # since we want the actor to follow
            # the action-value gradients
            grad_ys=-action_gradients)
        
        # clip gradients
        if max_gradient_norm:
            actor_gradients, _, _ = tf_utils.gradient_clip(
                actor_gradients, max_gradient_norm=max_gradient_norm)

        # Optimization Op
        optimizer = create_optimizer(opitmizer_name, learning_rate)
        optimization_op = optimizer.apply_gradients(
            zip(actor_gradients, actor_params))

        self._actor_params = actor_params
        self._actor_inputs = actor_inputs
        self._actor_outputs = actor_outputs
        self._actor_normalized_inputs = actor_normalized_inputs
        
        self._target_params = target_params
        self._target_inputs = target_inputs
        self._target_outputs = target_outputs
        self._target_normalized_inputs = target_normalized_inputs
        
        self._action_gradients = action_gradients
        self._actor_gradients = actor_gradients

        self._optimizer = optimizer
        self._optimization_op = optimization_op
        self._target_update_op = target_update_op
        self._num_trainable_vars = num_trainable_vars

    def _build_network(self, scope=None):
        with vs.variable_scope(scope, "ActorNetwork") as s:
            inputs = array_ops.placeholder(
                shape=[None, self._num_units],
                dtype=dtypes.float32, name="inputs")
            kernel = vs.get_variable(
                name="network_kernel",
                shape=[self._num_units, self._num_actions])
            bias = vs.get_variable(
                name="network_bias",
                shape=[self._num_actions])

            if self._batch_norm:
                normalized_inputs = contrib_layers.batch_norm(
                    inputs=inputs,
                    is_training=True,
                    # force the updates in place
                    # but have a speed penalty
                    updates_collections=None)
            else:
                normalized_inputs = inputs

            # for easier fetching
            normalized_inputs = array_ops.identity(
                normalized_inputs, name="normalized_inputs")

            # one layer without linearity
            outputs = math_ops.matmul(normalized_inputs, kernel)
            outputs = nn_ops.bias_add(outputs, bias, name="outputs")
            
            if self._activation is not None:
                outputs = self._activation(outputs, name="outputs_activated")

        parameters = variables.trainable_variables(s.name)
        
        return inputs, outputs, parameters, normalized_inputs

    def train(self, inputs, action_gradients):
        return self._sess.run(
            fetches=self._optimization_op,
            feed_dict={
                self._actor_inputs: inputs,
                self._action_gradients: action_gradients})

    def predict(self, inputs):
        actor_outputs = self._sess.run(
            fetches=self._actor_outputs,
            feed_dict={
                self._actor_inputs: inputs})
        return actor_outputs

    def predict_target(self, inputs):
        target_outputs = self._sess.run(
            fetches=self._target_outputs,
            feed_dict={
                self._target_inputs: inputs})

        return target_outputs

    def actor_gradients(self, inputs, gradiens):
        return self._sess.run(
            fetches=self._actor_gradients,
            feed_dict={
                self._actor_inputs: inputs,
                self._action_gradients: gradiens})

    def update_target(self):
        self._sess.run(fetches=self._target_update_op)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """
    def __init__(self,
                 sess,
                 num_units,
                 num_actions,
                 batch_size,
                 learning_rate,
                 tau,
                 gamma,
                 activation=None,
                 batch_norm=True,
                 opitmizer_name="adam",
                 max_gradient_norm=5.0,
                 critic_scope=None,
                 target_scope=None):
        if activation and not callable(activation):
            raise TypeError("Expected `activation` to be callable")

        self._sess = sess
        self._num_units = num_units
        self._num_actions = num_actions
        self._learning_rate = learning_rate
        self._tau = tau
        self._gamma = gamma
        self._activation = activation
        self._batch_norm = batch_norm
        self._critic_scope = critic_scope
        self._target_scope = target_scope

        # Create the critic network
        (critic_inputs,
         critic_actions,
         critic_outputs,
         critic_params,
         critic_normalized_inputs) = self._build_network(critic_scope)

        # Target Network
        (target_inputs,
         target_actions,
         target_outputs,
         target_params,
         target_normalized_inputs) = self._build_network(target_scope)

        # Op for periodically updating target network
        # with online network weights
        target_update_op = update_target_params(
            params=critic_params, target_params=target_params, tau=tau)

        # Network target (y_i)
        # y_t = reward_t + gamma Q"(s_t+1, U"(s_t+1))
        TD_target = array_ops.placeholder(
            dtype=dtypes.float32,
            shape=[None, 1],
            name="TD_target")

        # Define loss and optimization Op
        # Loss = Sum( (y_t - Q(s_t, a_t))^2 )
        critic_loss = nn_ops.l2_loss(TD_target - critic_outputs)
        critic_loss = math_ops.div(critic_loss, batch_size)
        
        # Gradients
        critic_gradients = gradients_impl.gradients(critic_loss, critic_params)
        # clip gradients
        if max_gradient_norm:
            critic_gradients, _, _ = tf_utils.gradient_clip(
                critic_gradients, max_gradient_norm=max_gradient_norm)
        
        # optimization
        optimizer = create_optimizer(opitmizer_name, learning_rate)
        optimization_op = optimizer.apply_gradients(
            zip(critic_gradients, critic_params))
        

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        action_grads = gradients_impl.gradients(critic_outputs, critic_actions)

        self._critic_params = critic_params
        self._critic_inputs = critic_inputs
        self._critic_actions = critic_actions
        self._critic_outputs = critic_outputs
        self._critic_normalized_inputs = critic_normalized_inputs
        
        self._target_params = target_params
        self._target_inputs = target_inputs
        self._target_actions = target_actions
        self._target_outputs = target_outputs
        self._target_normalized_inputs = target_normalized_inputs
        
        self._TD_target = TD_target
        self._critic_loss = critic_loss
        self._target_update_op = target_update_op
        
        
        self._optimization_op = optimization_op
        self._action_grads = action_grads


    def _build_network(self, scope=None):
        with vs.variable_scope(scope, "CriticNetwork") as s:
            inputs = array_ops.placeholder(
                shape=[None, self._num_units],
                dtype=dtypes.float32, name="inputs")
            actions = array_ops.placeholder(
                shape=[None, self._num_actions],
                dtype=dtypes.float32, name="actions")

            inputs_kernel = vs.get_variable(
                name="inputs_network_kernel",
                shape=[self._num_units, 1])
            actions_kernel = vs.get_variable(
                name="actions_network_kernel",
                shape=[self._num_actions, 1])
            bias = vs.get_variable(
                name="network_bias",
                shape=[1])

            if self._batch_norm:
                normalized_inputs = contrib_layers.batch_norm(
                    inputs=inputs,
                    is_training=True,
                    # force the updates in place
                    # but have a speed penalty
                    updates_collections=None)
            else:
                normalized_inputs = inputs

            # for easier fetching
            normalized_inputs = array_ops.identity(
                normalized_inputs, name="normalized_inputs")

            # one layer without linearity
            _input = math_ops.matmul(normalized_inputs, inputs_kernel)
            _actions = math_ops.matmul(actions, actions_kernel)
            outputs = nn_ops.bias_add(_input + _actions, bias, name="outputs")

            if self._activation is not None:
                outputs = self._activation(outputs, name="outputs_activated")

            parameters = variables.trainable_variables(s.name)

        return inputs, actions, outputs, parameters, normalized_inputs

    def train(self, inputs, actions, TD_target):
        return self._sess.run(
            fetches=[
                self._critic_outputs,
                self._optimization_op],
            feed_dict={
                self._critic_inputs: inputs,
                self._critic_actions: actions,
                self._TD_target: TD_target})

    def predict(self, inputs, actions):
        critic_outputs = self._sess.run(
            fetches=self._critic_outputs,
            feed_dict={
                self._critic_inputs: inputs,
                self._critic_actions: actions})
        return critic_outputs

    def predict_target(self, inputs, actions):
        target_outputs = self._sess.run(
            fetches=self._target_outputs,
            feed_dict={
                self._target_inputs: inputs,
                self._target_actions: actions})
        return target_outputs

    def compute_TD_target(self, rewards, next_inputs, next_actions):
        """compute y = r + gamma Q"(s_t+1, U"(s_t+1))
        
        Args:
            rewards: [batch]
            next_inputs: [batch, num_units]
            next_actions: [batch, actions]

        Returns:
            TD_target: [batch, 1]
            next_target_outputs: [batch, 1]
        """
        # [batch, 1]
        next_target_outputs = self.predict_target(
            inputs=next_inputs, actions=next_actions)

        # [batch, 1]
        expanded_rewards = np.expand_dims(rewards, axis=1)
        
        # [batch, 1]
        TD_target = expanded_rewards + self._gamma * next_target_outputs
        return TD_target, next_target_outputs


    def action_gradients(self, inputs, actions):
        return self._sess.run(
            fetches=self._action_grads,
            feed_dict={
                self._critic_inputs: inputs,
                self._critic_actions: actions})

    def update_target(self):
        self._sess.run(self._target_update_op)



def update_target_params(params, target_params, tau):
    if not len(params) == len(target_params):
        raise ValueError("actor and target should have same params")
    target_update_op = [
        target_params[i].assign(
            math_ops.multiply(params[i], tau) +
            math_ops.multiply(target_params[i], 1. - tau))
        for i in range(len(target_params))]
    return target_update_op


def create_optimizer(opitmizer_name, learning_rate):
    # Optimization Op
    if opitmizer_name == "sgd":
        opitmizer_name = gd_ops.GradientDescentOptimizer(learning_rate)
    elif opitmizer_name == "adam":
        opitmizer_name = adam_ops.AdamOptimizer(learning_rate)
    else:
        raise ValueError("Unknown opitmizer_name ", opitmizer_name)

    return opitmizer_name
