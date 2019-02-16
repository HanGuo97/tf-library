from collections import namedtuple
from TFLibrary.NewTuner import utils


ClientEnvelop = namedtuple(
    "ClientEnvelop",
    ("header", "contents", "identity"))

ServerEnvelop = namedtuple(
    "ServerEnvelop",
    ("header", "contents", "identity"))

VALID_CLIENT_HEADERS = ["HELLO", "POST", "GET"]
VALID_SERVER_HEADERS = ["HELLO", "READY", "POST"]


def make_client_envelop(header,
                        contents,
                        identity,
                        serialize=True):

    if header not in VALID_CLIENT_HEADERS:
        raise ValueError

    envelop = ClientEnvelop(
        header=header,
        contents=contents,
        identity=identity)

    if serialize:
        return utils.serialize_pyobj(envelop)
    else:
        return envelop


def make_server_envelop(header,
                        contents,
                        identity,
                        serialize=True):
    if header not in VALID_SERVER_HEADERS:
        raise ValueError

    envelop = ServerEnvelop(
        header=header,
        contents=contents,
        identity=identity)

    if serialize:
        return utils.serialize_pyobj(envelop)
    else:
        return envelop


def deserialize_client_envelop(serialied_envelop):
    envelop = utils.deserialize_pyobj(serialied_envelop)
    if not isinstance(envelop, ClientEnvelop):
        raise TypeError("Received object is not `ClientEnvelop`, "
                        "but {}".format(type(envelop)))

    return envelop


def deserialize_server_envelop(serialied_envelop):
    envelop = utils.deserialize_pyobj(serialied_envelop)
    if not isinstance(envelop, ServerEnvelop):
        raise TypeError("Received object is not `ServerEnvelop`, "
                        "but {}".format(type(envelop)))

    return envelop
