from __future__ import print_function

import tensorflow as tf
import autograd
import autograd.numpy as np   # Thinly-wrapped version of Numpy

from TFLibrary.DDPG import ddpg
from TFLibrary.utils import test_utils

print("Using Tensorflow ", tf.__version__)


def ddpg_test():
    sess = tf.Session()
    actor = ddpg.ActorNetwork(
        sess=sess,
        num_units=128,
        batch_size=32,
        num_actions=9,
        learning_rate=50,
        tau=0.001,
        # using SGD for easier debugging
        batch_norm=True,
        opitmizer_name="sgd",
        max_gradient_norm=None,
        actor_scope="Actor",
        target_scope="ActorTarget")

    critic = ddpg.CriticNetwork(
        sess=sess,
        num_units=128,
        num_actions=9,
        batch_size=32,
        learning_rate=50,
        tau=0.001,
        gamma=0.99,
        # using SGD for easier debugging
        batch_norm=True,
        opitmizer_name="sgd",
        max_gradient_norm=None,
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
    test_critic(actor, critic, sess, global_variables)
    test_target_network_updates(actor, critic, sess)


def test_actor(actor, critic, sess, global_variables):
    """Test Actor
        
        U: Actor
        Q: Critic

        grad(theta) J ~= grad(a) Q(s=s,a=U(s)) x grad(theta) U(s)

    """
    # Test The Objective Gradient
    # ============================================
    test_utils.test_message("Actor")

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
    # from [batch, gradient] to [gradient]
    expected_batch_grad_critic_wrt_action = np.mean(
        expected_grad_critic_wrt_action, axis=0)

    expected_grad_J_wrt_W = (
        -expected_grad_actor_wrt_params[0] *
        expected_batch_grad_critic_wrt_action)

    expected_grad_J_wrt_b = (
        -expected_grad_actor_wrt_params[1] *
        expected_batch_grad_critic_wrt_action)

    # actions and Q
    test_utils.check_equality(actual_action, expected_action, "Action")
    test_utils.check_equality(actual_Q, expected_Q, "Q")
    # critic gradients
    assert len(actual_critic_grads) == 1
    test_utils.check_equality(
        actual_critic_grads[0],
        expected_grad_critic_wrt_action, "CriticGradients")
    # actor gradients
    test_utils.check_equality(
        actual_actor_grads[0], expected_grad_J_wrt_W, "ObjectiveGradients1")
    test_utils.check_equality(
        actual_actor_grads[1], expected_grad_J_wrt_b, "ObjectiveGradients2")

    

    # Test Updating The Model
    # ============================================
    test_utils.test_message("TESTING: updating the model")
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
    [test_utils.check_inequality(n, o, "ActorParamsUpdate")
        for n, o in zip(old_actor_params,
                        new_actor_params)]
    
    # targets should be same
    [test_utils.check_equality(n, o, "ActorTargetParamsUpdate")
        for n, o in zip(old_actor_target_params,
                        new_actor_target_params)]

    # manually compute updates
    actor_params_diff = [n - o
        for n, o in zip(new_actor_params, old_actor_params)]
    
    # negative gradients as in gradient descent
    lr = actor._learning_rate
    scaled_gradients = [lr * - grad
        for grad in actual_actor_grads]

    [test_utils.check_equality(a, g, "ActorParamsUpdate")
        for a, g in zip(actor_params_diff, scaled_gradients)]


def test_critic(actor, critic, sess, global_variables):
    """Test Critic
        
        U: Actor
        U": Actor Target
        Q: Critic
        Q": Critic Target
        s': next state
        
        q = r + gamma Q"(s', U"(s'))
        Loss = 1/2 SUM[(q - Q(s,a))^2]
    """
    # Test The Objective Gradient
    # ============================================
    test_utils.test_message("Critic")

    # Set Up
    # --------------------------------------------------
    reward = test_utils.random_vector([])
    state = test_utils.random_vector([32, 128])
    next_state = test_utils.random_vector([32, 128])
    variables = sess.run(global_variables)
    # it's too complicated to replicate the BatchNorm
    # so just use the provided one
    actor_state = sess.run(
        "Actor/normalized_inputs:0",
        {"Actor/inputs:0": state})
    critic_state = sess.run(
        "Critic/normalized_inputs:0",
        {"Critic/inputs:0": state})
    actor_next_state = sess.run(
        "ActorTarget/normalized_inputs:0",
        {"ActorTarget/inputs:0": next_state})
    critic_next_state = sess.run(
        "CriticTarget/normalized_inputs:0",
        {"CriticTarget/inputs:0": next_state})

    # Actual Loss
    # --------------------------------------------------
    # U(s) and U"(s')
    actual_action = actor.predict(inputs=state)
    actual_next_action = actor.predict_target(inputs=next_state)
    
    # Q(s,a), Q"(s',a') and q (TD_target)
    actual_Q = critic.predict(inputs=state, actions=actual_action)
    actual_TD_target, actual_next_Q = critic.compute_TD_target(
        rewards=reward,
        next_inputs=next_state,
        next_actions=actual_next_action)

    # 1/2 (q - Q)^2
    actual_critic_loss = sess.run(
        critic._critic_loss,
        {critic._critic_inputs: state,
         critic._critic_actions: actual_action,
         critic._TD_target: actual_TD_target})

    # Expected Loss
    # --------------------------------------------------
    # U(s) and U"(s')
    expected_action = actor_fn(
        S=actor_state,
        W=variables["Actor/network_kernel:0"],
        b=variables["Actor/network_bias:0"])
    expected_next_action = actor_fn(
        S=actor_next_state,
        W=variables["ActorTarget/network_kernel:0"],
        b=variables["ActorTarget/network_bias:0"])
    
    # Q(s,a) and Q(s',a')
    expected_Q = critic_fn(
        S=critic_state,
        A=expected_action,
        Ws=variables["Critic/inputs_network_kernel:0"],
        Wa=variables["Critic/actions_network_kernel:0"],
        b=variables["Critic/network_bias:0"])
    expected_next_Q = critic_fn(
        S=critic_next_state,
        A=expected_next_action,
        Ws=variables["CriticTarget/inputs_network_kernel:0"],
        Wa=variables["CriticTarget/actions_network_kernel:0"],
        b=variables["CriticTarget/network_bias:0"])

    expected_TD_target = reward + critic._gamma * expected_next_Q
    expected_critic_loss = test_utils.L2_distance(
        expected_Q, expected_TD_target)
    expected_critic_loss = expected_critic_loss / 32

    test_utils.check_equality(
        actual_action, expected_action, "Action")
    test_utils.check_equality(
        actual_next_action, expected_next_action, "NextAction")
    test_utils.check_equality(
        actual_Q, expected_Q, "Q")
    test_utils.check_equality(
        actual_next_Q, expected_next_Q, "NextQ")
    test_utils.check_equality(
        actual_TD_target, expected_TD_target, "TDTarget")
    test_utils.check_equality(
        float(actual_critic_loss), expected_critic_loss, "CriticLoss")




    # Test Updating The Model
    # ============================================
    test_utils.test_message("TESTING: updating the model")

    # Expected Updates
    # --------------------------------------------------
    # fetch the gradients ** before ** update
    lr = critic._learning_rate
    # negative gradients as in gradient descent
    expected_critic_loss_grad = -lr * (expected_Q - expected_TD_target) / 32
    expected_critic_loss_grad = expected_critic_loss_grad.astype(np.float32)

    # grad(Ws,Wa,b) Q, using Jacobians
    # for some unknown reasons autograd.jacobian
    # does not support multiple argnums
    grad_critic_wrt_Ws = autograd.jacobian(critic_fn, argnum=2)
    grad_critic_wrt_Wa = autograd.jacobian(critic_fn, argnum=3)
    grad_critic_wrt_b = autograd.jacobian(critic_fn, argnum=4)

    # [output_shapes, variable_shapes]
    # [32,1, 128,1] to [32, 128]
    expected_grad_critic_wrt_Ws = grad_critic_wrt_Ws(
        critic_state,
        expected_action,
        variables["Critic/inputs_network_kernel:0"],
        variables["Critic/actions_network_kernel:0"],
        variables["Critic/network_bias:0"]).squeeze()
    # [32,1, 9,1] to [32, 9]
    expected_grad_critic_wrt_Wa = grad_critic_wrt_Wa(
        critic_state,
        expected_action,
        variables["Critic/inputs_network_kernel:0"],
        variables["Critic/actions_network_kernel:0"],
        variables["Critic/network_bias:0"]).squeeze()
    # [32,1, 1] to [32,]
    expected_grad_critic_wrt_b = grad_critic_wrt_b(
        critic_state,
        expected_action,
        variables["Critic/inputs_network_kernel:0"],
        variables["Critic/actions_network_kernel:0"],
        variables["Critic/network_bias:0"]).squeeze()

    # to get gradients, multiply the Jacobian with loss
    # [num_units, batch] x [batch, 1] = [num_units, 1]
    expected_grad_critic_wrt_Ws = np.matmul(
        expected_grad_critic_wrt_Ws.transpose(), expected_critic_loss_grad)
    expected_grad_critic_wrt_Wa = np.matmul(
        expected_grad_critic_wrt_Wa.transpose(), expected_critic_loss_grad)
    expected_grad_critic_wrt_b = np.matmul(
        expected_grad_critic_wrt_b.transpose(), expected_critic_loss_grad)

    # Actual Updates
    # --------------------------------------------------
    # old parameters
    old_critic_params = sess.run(critic._critic_params)
    old_critic_target_params = sess.run(critic._target_params)
    # update model
    critic.train(
        inputs=state,
        actions=actual_action,
        TD_target=actual_TD_target)
    # new parameters
    new_critic_params = sess.run(critic._critic_params)
    new_critic_target_params = sess.run(critic._target_params)
    # manually compute updates
    actual_grad_critic = [n - o
        for n, o in zip(new_critic_params, old_critic_params)]
    
    # critic should be ** different **
    [test_utils.check_inequality(n, o, "CriticParamsUpdate")
        for n, o in zip(new_critic_params,
                        old_critic_params)]
    # targets should be same
    [test_utils.check_equality(n, o, "CriticTargetParamsUpdate")
        for n, o in zip(new_critic_target_params,
                        old_critic_target_params)]

    # actor gradients
    test_utils.check_equality(
        actual_grad_critic[0], expected_grad_critic_wrt_Ws, "CriticUpdates1")
    test_utils.check_equality(
        actual_grad_critic[1], expected_grad_critic_wrt_Wa, "CriticUpdates2")
    test_utils.check_equality(
        actual_grad_critic[2], expected_grad_critic_wrt_b, "CriticUpdates3")


def test_target_network_updates(actor, critic, sess):
    # Test The Objective Gradient
    # ============================================
    test_utils.test_message("Target Network Updates")

    # Actor
    # --------------------------------------------------
    # old parameters
    old_actor_params = sess.run(actor._actor_params)
    old_actor_target_params = sess.run(actor._target_params)
    # update model
    actor.update_target()
    # new parameters
    new_actor_params = sess.run(actor._actor_params)
    new_actor_target_params = sess.run(actor._target_params)

    # Critic
    # --------------------------------------------------
    # old parameters
    old_critic_params = sess.run(critic._critic_params)
    old_critic_target_params = sess.run(critic._target_params)
    # update model
    critic.update_target()
    # new parameters
    new_critic_params = sess.run(critic._critic_params)
    new_critic_target_params = sess.run(critic._target_params)


    # actors and critics should not be updated
    [test_utils.check_equality(n, o, "ActorUpdates")
        for n, o in zip(new_actor_params, old_actor_params)]
    [test_utils.check_equality(n, o, "CriticUpdates")
        for n, o in zip(new_critic_params, old_critic_params)]

    # actors and critics targets should be updated
    [test_utils.check_inequality(n, o, "ActorTargetUpdates")
        for n, o in zip(new_actor_target_params, old_actor_target_params)]
    [test_utils.check_inequality(n, o, "CriticTargetUpdates")
        for n, o in zip(new_critic_target_params, old_critic_target_params)]


    # new theta" = tau theta + (1 - tau) theta"
    # new theta" - (1 - tau) theta" = tau theta
    actor_tau = actor._tau
    critic_tau = critic._tau
    # LHS
    scaled_actor_params = [actor_tau * n for n in new_actor_params]
    scaled_critic_params = [critic_tau * n for n in new_critic_params]
    # RHS
    actor_target_params_diff = [n - (1 - actor_tau) * o
        for n, o in zip(new_actor_target_params, old_actor_target_params)]
    critic_target_params_diff = [n - (1 - critic_tau) * o
        for n, o in zip(new_critic_target_params, old_critic_target_params)]

    # check difference
    [test_utils.check_equality(diff, sa, "ActorTargetUpdates")
        for diff, sa in zip(actor_target_params_diff, scaled_actor_params)]

    [test_utils.check_equality(diff, sc, "CriticTargetUpdates")
        for diff, sc in zip(critic_target_params_diff, scaled_critic_params)]


def actor_fn(S, W, b):
    """A = tanh(S W + b)"""
    A = np.tanh(np.matmul(S, W) + b)
    return A


def critic_fn(S, A, Ws, Wa, b):
    """Q = S Ws + A Wa + b"""
    Q = np.matmul(S, Ws) + np.matmul(A, Wa) + b
    return Q


if __name__ == "__main__":
    ddpg_test()
