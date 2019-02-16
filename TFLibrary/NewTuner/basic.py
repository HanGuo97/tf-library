import zmq
import time
from zmq import ssh
from absl import logging
from TFLibrary.NewTuner import utils
from TFLibrary.NewTuner import envelops

from overrides import overrides
from collections import namedtuple
logging.set_verbosity(logging.INFO)

Envelop = namedtuple("Envelop", ("sender_address", "contents"))


class RemoteBase(object):
    """Metrics Server"""
    def __init__(self,
                 address,
                 socket_type,
                 ssh_server=None,
                 ssh_password=None,
                 identity="server"):

        self._address = address
        self._socket_type = socket_type

        # SSH Tuneling
        self._ssh_server = ssh_server
        self._ssh_password = ssh_password
        
        # Info
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
        socket = context.socket(self._socket_type)

        self.connect(socket)
        self._context = context
        self._socket = socket

    def verify_connection(self):
        # First Send a Hello message to ensure the
        # connection is established
        self.send(header="HELLO")
        envelop = self.receive()

        if not isinstance(envelop, (envelops.ClientEnvelop,
                                    envelops.ServerEnvelop)):
            raise TypeError("received envelop must be `Envelop`")
        
        if envelop.header == "HELLO":
            logging.info("Connected to {}".format(self._address))
        else:
            raise ValueError("Received {}".format(envelop.header))


    def send(self, *args, **kwargs):
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
            socket_type=zmq.REQ,
            ssh_server=ssh_server,
            ssh_password=ssh_password,
            identity=identity)

        self.start()

    def send(self, header, contents=None):
        envelop = envelops.make_server_envelop(
            header=header,
            contents=contents,
            identity=self.identity,
            serialize=True)
 
        self._socket.send(envelop)

    def receive(self):
        return envelops.deserialize_server_envelop(
            serialied_envelop=self._socket.recv())

    def start(self):
        # Tell the Router the server is ready
        self.send(header="READY")

        try:
            while True:
                envelop = self.receive()
                logging.info("{}: {}".format(self.identity, envelop.contents))
                contents_to_return = self.execute(envelop.contents)
                self.send(header="POST", contents=contents_to_return)

        except zmq.ContextTerminated:
            # context terminated so quit silently
            return

    def execute(self, contents):
        time.sleep(30)
        return "OK"


class BasicClient(RemoteBase):
    """Tuner Client

    Client sends requests (i.e. instances of models) to the proxy, who
    then forwards requests to servers.

    Message Structure:
        Header: String
            The type of request, POST or GET
        Identity: String
            Identity of the client
        Contents: Object
            The contents to be sent.
    """
    def __init__(self,
                 address,
                 ssh_server=None,
                 ssh_password=None,
                 identity="client"):
        super(BasicClient, self).__init__(
            address=address,
            socket_type=zmq.REQ,
            ssh_server=ssh_server,
            ssh_password=ssh_password,
            identity=identity)

    def send(self, header, contents=None):
        envelop = envelops.make_client_envelop(
            header=header,
            contents=contents,
            identity=self.identity,
            serialize=True)
 
        self._socket.send(envelop)

    def receive(self):
        return envelops.deserialize_client_envelop(
            serialied_envelop=self._socket.recv())
