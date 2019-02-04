import zmq
from zmq import ssh
from absl import logging
from TFLibrary.NewTuner import utils
from overrides import overrides
from collections import namedtuple

Envelop = namedtuple("Envelop", ("sender_address", "contents"))


class ServerBase(object):
    """Metrics Server"""
    def __init__(self,
                 address,
                 ssh_tunnel=False,
                 ssh_server=None,
                 ssh_password=None,
                 identity="server"):

        self._address = address
        self._ssh_tunnel = ssh_tunnel
        self._ssh_server = ssh_server
        self._ssh_password = ssh_password
        self._identity = identity

        self.setup()
        self.verify_connection()
        self.start()

    @property
    def identity(self):
        return self._identity
    
    def setup(self):
        raise NotImplementedError

    def verify_connection(self):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def send(self, messages):
        raise NotImplementedError

    def receive(self):
        raise NotImplementedError

    def connect(self, socket):
        if not self._ssh_tunnel:
            socket.connect(self._address)
        else:  # --- Connect via SSH Tunnel ---
            ssh.tunnel_connection(
                socket, self._address,
                server=self._server,
                password=self._password)

    def execute(self, contents):
        raise NotImplementedError


class BasicServer(ServerBase):
    def setup(self):
        # ZeroMQ Context
        context = zmq.Context.instance()
        # Define the socket using the "Context"
        socket = context.socket(zmq.REQ)
        
        self.connect(socket)
        logging.info("Connecting to %s" % self._address)

        self._context = context
        self._socket = socket

    def verify_connection(self):
        # First Send a Hello message to ensure the
        # connection is established
        self.send(contents=utils.HELLO_MESSAGE,
                  receiver_address=None)

        envelop = self.receive()
        if envelop.contents == utils.HELLO_MESSAGE:
            logging.info("Connected to %s" % envelop.contents[1])

    def send(self, contents, receiver_address):
        self._socket.send_multipart([
            receiver_address or utils.EMPTY,
            utils.serialize_pyobj(self.identity),
            utils.serialize_pyobj(contents)])

    def receive(self):
        messages = self._socket.recv_multipart()
        
        if len(messages) != 3:
            raise ValueError("`len(messages)` should be 3")

        # Now get next client request, route to LRU worker
        # Client request is [address][empty][request]
        sender_address, empty, contents = messages
        utils.assert_empty(empty)

        return Envelop(
            sender_address=sender_address,
            contents=utils.deserialize_pyobj(contents))

    def start(self):
        self.send(contents=utils.READY_MESSAGE,
                  receiver_address=None)
        try:
            while True:
                envelop = self.receive()
                logging.info("%s: %s" % (self.identity, envelop.contents))

                contents_to_return = self.execute(envelop.contents)
                self.send(contents=contents_to_return,
                          receiver_address=envelop.sender_address)

        except zmq.ContextTerminated:
            # context terminated so quit silently
            return

    def execute(self, contents):
        return "OK"
