from __future__ import print_function

import tensorflow as tf
import autograd
import autograd.numpy as np   # Thinly-wrapped version of Numpy

from DDPG import ddpg
from utils import test_utils

print("Using Tensorflow ", tf.__version__)


def ddpg_test():
    sess = tf.Session()
    actor = ddpg.ActorNetwork(
        sess=sess,
        num_units=128,
        batch_size=32,
        num_actions=9,
        learning_rate=5,
        tau=0.001,
        # using SGD for easier debugging
        opitmizer_name="sgd",
        actor_scope="Actor",
        target_scope="ActorTarget")

    critic = ddpg.CriticNetwork(
        sess=sess,
        num_units=128,
        num_actions=9,
        batch_size=32,
        learning_rate=5,
        tau=0.001,
        gamma=0.99,
        # using SGD for easier debugging
        opitmizer_name="sgd",
        critic_scope="Critic",
        target_scope="CriticTarget")

    sess.run(tf.global_variables_initializer())

    test_utils.display(actor._actor_params)
    test_utils.display(actor._target_params)

    test_utils.display(critic._critic_params)
    test_utils.display(critic._target_params)

    # fetch all variables
    global_variables = {}
    for var in tf.global_variables():
        global_variables[var.name] = var


    test_actor(actor, critic, sess, global_variables)


def test_actor(actor, critic, sess, global_variables):
    """Test Actor Gradients
        
        U: Actor
        Q: Critic

        grad(theta) J ~= grad(a) Q(s=s,a=U(s)) x grad(theta) U(s)

    """
    # Test The Objective Gradient
    # ============================================
    print("TESTING: test_actor_gradients\n\n")

    # Set Up
    # --------------------------------------------------
    state = test_utils.random_vector([32, 128])
    variables = sess.run(global_variables)
    # it's too complicated to replicate the BatchNorm
    # so just use the provided one
    actor_state = sess.run(
        "Actor/normalized_inputs:0", {"Actor/inputs:0": state})
    critic_state = sess.run(
        "Critic/normalized_inputs:0", {"Critic/inputs:0": state})

    # Actual Gradients
    # --------------------------------------------------
    # U(s)
    actual_action = actor.predict(inputs=state)
    # Q(s,a=U(s))
    actual_Q = critic.predict(inputs=state, actions=actual_action)
    # symbolic grad(a) Q, and grad U
    actual_critic_grads = critic.action_gradients(
        inputs=state, actions=actual_action)
    actual_actor_grads = actor.actor_gradients(
        inputs=state, gradiens=actual_critic_grads[0])

    # Expected Gradients
    # --------------------------------------------------
    # grad(W, b) U
    grad_actor_wrt_params = autograd.elementwise_grad(actor_fn, argnum=[1, 2])
    # grad(A) Q
    grad_critic_wrt_action = autograd.elementwise_grad(critic_fn, argnum=1)

    expected_action = actor_fn(
        S=actor_state,
        W=variables["Actor/network_kernel:0"],
        b=variables["Actor/network_bias:0"])

    expected_Q = critic_fn(
        S=critic_state,
        A=expected_action,
        Ws=variables["Critic/inputs_network_kernel:0"],
        Wa=variables["Critic/actions_network_kernel:0"],
        b=variables["Critic/network_bias:0"])

    expected_grad_actor_wrt_params = grad_actor_wrt_params(
        actor_state,
        variables["Actor/network_kernel:0"],
        variables["Actor/network_bias:0"])

    expected_grad_critic_wrt_action = grad_critic_wrt_action(
        critic_state,
        expected_action,
        variables["Critic/inputs_network_kernel:0"],
        variables["Critic/actions_network_kernel:0"],
        variables["Critic/network_bias:0"])

    # sum over batch
    # from [batch, gradient]
    # to [gradient]
    expected_batch_grad_critic_wrt_action = np.mean(
        expected_grad_critic_wrt_action, axis=0)

    expected_grad_J_wrt_W = (
        -expected_grad_actor_wrt_params[0] *
        expected_batch_grad_critic_wrt_action)

    expected_grad_J_wrt_b = (
        -expected_grad_actor_wrt_params[1] *
        expected_batch_grad_critic_wrt_action)

    # actions and Q
    test_utils.check_equality(actual_action, expected_action)
    test_utils.check_equality(actual_Q, expected_Q)
    # critic gradients
    assert len(actual_critic_grads) == 1
    test_utils.check_equality(actual_critic_grads[0],
                              expected_grad_critic_wrt_action)
    # actor gradients
    test_utils.check_equality(actual_actor_grads[0], expected_grad_J_wrt_W)
    test_utils.check_equality(actual_actor_grads[1], expected_grad_J_wrt_b)

    

    # Test Updating The Model
    # ============================================
    print("\n\n\nTESTING: updating the model\n\n")
    # old parameters
    old_actor_params = sess.run(actor._actor_params)
    old_actor_target_params = sess.run(actor._target_params)
    # update model
    actor.train(
        inputs=state,
        action_gradients=actual_critic_grads[0])
    # new parameters
    new_actor_params = sess.run(actor._actor_params)
    new_actor_target_params = sess.run(actor._target_params)


    # actors should be ** different **
    [test_utils.check_inequality(n, o)
        for n, o in zip(old_actor_params,
                        new_actor_params)]
    
    # targets should be same
    [test_utils.check_equality(n, o)
        for n, o in zip(old_actor_target_params,
                        new_actor_target_params)]

    # manually compute updates
    lr = actor._learning_rate
    actor_params_diff = [n - o
        for n, o in zip(new_actor_params, old_actor_params)]
    
    # negative gradients as in gradient descent
    scaled_gradients = [lr * - grad
        for grad in actual_actor_grads]

    [test_utils.check_equality(a, g)
        for a, g in zip(actor_params_diff, scaled_gradients)]




def actor_fn(S, W, b):
    """A = tanh(S' W + b)"""
    A = np.tanh(np.matmul(S, W) + b)
    return A


def critic_fn(S, A, Ws, Wa, b):
    """Q = S' Ws + A Wa + b"""
    Q = np.matmul(S, Ws) + np.matmul(A, Wa) + b
    return Q
