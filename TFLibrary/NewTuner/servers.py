import zmq
from zmq import ssh
from absl import logging
from TFLibrary.NewTuner import utils
from overrides import overrides
from collections import namedtuple
logging.set_verbosity(logging.INFO)

Envelop = namedtuple("Envelop", ("sender_address", "contents"))


class RemoteBase(object):
    """Metrics Server"""
    def __init__(self,
                 address,
                 ssh_server=None,
                 ssh_password=None,
                 identity="server"):

        self._address = address
        self._ssh_server = ssh_server
        self._ssh_password = ssh_password
        self._identity = identity

        self.setup()
        self.verify_connection()

    @property
    def identity(self):
        return self._identity
    
    def setup(self):
        # ZeroMQ Context
        context = zmq.Context.instance()
        # Define the socket using the "Context"
        socket = context.socket(zmq.REQ)

        self.connect(socket)
        self._context = context
        self._socket = socket

    def verify_connection(self):
        # First Send a Hello message to ensure the
        # connection is established
        self.send(contents=utils.HELLO_MESSAGE)
        envelop = self.receive()

        if not isinstance(envelop, Envelop):
            raise TypeError("received envelop must be `Envelop`")
        
        if envelop.contents == utils.HELLO_MESSAGE:
            logging.info("Connected to {}".format(self._address))
        else:
            raise ValueError("Received {}".format(envelop.contents))


    def send(self, messages, *args, **kwargs):
        raise NotImplementedError

    def receive(self, *args, **kwargs):
        raise NotImplementedError

    def connect(self, socket):
        if not self._ssh_server:
            socket.connect(self._address)
        else:  # --- Connect via SSH Tunnel ---
            ssh.tunnel_connection(
                socket,
                self._address,
                server=self._ssh_server,
                password=self._ssh_password)

        logging.info("Connecting to {}".format(self._address))

    def execute(self, contents):
        raise NotImplementedError


class BasicServer(RemoteBase):
    def __init__(self,
                 address,
                 ssh_server=None,
                 ssh_password=None,
                 identity="server"):
        super(BasicServer, self).__init__(
            address=address,
            ssh_server=ssh_server,
            ssh_password=ssh_password,
            identity=identity)

        self.start()

    def send(self, contents, receiver_address=None):
        # The message is only seen by Router
        no_receiver = receiver_address is None
        if utils.is_internal_message(contents) != no_receiver:
            raise ValueError(
                "Internal contents should only be sent without receiver, "
                "likewise, external contents need specified receiver")

        self._socket.send_multipart([
            receiver_address or utils.EMPTY,
            utils.serialize_pyobj(self.identity),
            utils.serialize_pyobj(contents)])

    def receive(self):
        messages = self._socket.recv_multipart()

        # [address][empty][request]
        if len(messages) != 3:
            raise ValueError("`len(messages)` should be 3")

        sender_address, empty, contents = messages
        utils.assert_empty(empty)
        contents = utils.deserialize_pyobj(contents)

        return Envelop(contents=contents,
                       sender_address=sender_address)

    def start(self):
        # Tell the Router the server is ready
        self.send(contents=utils.READY_MESSAGE)

        try:
            while True:
                envelop = self.receive()
                logging.info("{}: {}".format(self.identity, envelop.contents))
                contents_to_return = self.execute(envelop.contents)
                self.send(contents=contents_to_return,
                          receiver_address=envelop.sender_address)

        except zmq.ContextTerminated:
            # context terminated so quit silently
            return

    def execute(self, contents):
        return "OK"


class BasicClient(RemoteBase):
    def __init__(self,
                 address,
                 ssh_server=None,
                 ssh_password=None,
                 identity="server"):
        super(BasicClient, self).__init__(
            address=address,
            ssh_server=ssh_server,
            ssh_password=ssh_password,
            identity=identity)

        self.request()

    def send(self, contents):
        self._socket.send_multipart([
            utils.serialize_pyobj(self.identity),
            utils.serialize_pyobj(contents)])

    def receive(self):
        messages = self._socket.recv_multipart()
        # Response is [response]
        if len(messages) != 1:
            raise ValueError("`len(messages)` should be 1")

        contents = utils.deserialize_pyobj(messages[0])
        return Envelop(contents=contents,
                       sender_address=None)

    def request(self):
        # Tell the Router the server is ready
        self.send(contents=utils.READY_MESSAGE)
        envelop = self.receive()
        logging.info("{}: {}".format(self.identity, envelop.contents))
