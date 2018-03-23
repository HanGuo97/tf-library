import tensorflow as tf


def gradient_clip(gradients, max_gradient_norm):
    """Clipping gradients of a model."""
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
        gradients, max_gradient_norm)
    gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
    gradient_norm_summary.append(
        tf.summary.scalar("clipped_gradient",
                          tf.global_norm(clipped_gradients)))

    return clipped_gradients, gradient_norm_summary, gradient_norm


def untested_warning():
    tf.logging.warning("THIS IS NOT TESTED")


def add_summary(summary_writer, global_step, tag, value):
    """Add a new summary to the current summary_writer.
    Useful to log things that are not part of the training graph,
    e.g., tag=BLEU.
    """
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    summary_writer.add_summary(summary, global_step)


def get_config():
    """Returns config for tf.session"""
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    return config


def load_ckpt(saver, sess, ckpt_dir=None, ckpt_file=None):
    if not ckpt_dir and not ckpt_file:
        return

    if not ckpt_file:
        ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
        if ckpt_file is None:
            return

    saver.restore(sess, ckpt_file)
    tf.logging.info("Loaded checkpoint %s" % ckpt_file)
    return ckpt_file
