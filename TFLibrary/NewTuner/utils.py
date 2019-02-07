try:
    import cPickle
    pickle = cPickle
except ImportError:
    cPickle = None
    import pickle

try:
    DEFAULT_PROTOCOL = pickle.DEFAULT_PROTOCOL
except AttributeError:
    DEFAULT_PROTOCOL = pickle.HIGHEST_PROTOCOL


# Empty Deliminator is used by the socket lib, so must be bytes
EMPTY = b""
HELLO_MESSAGE = "HELLO"
READY_MESSAGE = "READY"


def is_internal_message(message):
    return message in [EMPTY, HELLO_MESSAGE, READY_MESSAGE]


def assert_empty(empty):
    if empty != EMPTY:
        raise ValueError("Empty should be empty")


def assert_safe_no_receiver(contents, receiver_address):
    # The message is only seen by Router
    no_receiver = receiver_address is None
    if is_internal_message(contents) != no_receiver:
        raise ValueError(
            "Internal contents should only be sent without receiver, "
            "likewise, external contents need specified receiver. "
            "Contents {} Address {}".format(contents, receiver_address))



def is_hello_message(message):
    return message == HELLO_MESSAGE


def serialize_pyobj(obj, protocol=DEFAULT_PROTOCOL):
    return pickle.dumps(obj, protocol)


def deserialize_pyobj(messages):
    return pickle.loads(messages)
