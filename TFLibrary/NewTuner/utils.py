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


def assert_empty(empty):
    if empty != EMPTY:
        raise ValueError("Empty should be empty")


def is_hello_message(message):
    return message == HELLO_MESSAGE


def serialize_pyobj(obj, protocol=DEFAULT_PROTOCOL):
    return pickle.dumps(obj, protocol)


def deserialize_pyobj(messages):
    return pickle.loads(messages)
