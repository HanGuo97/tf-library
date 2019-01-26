"""https://www.digitalocean.com/community/tutorials/
   how-to-work-with-the-zeromq-messaging-library"""

import zmq
from absl import logging
from TFLibrary.Remotes import remote_base
from TFLibrary.Remotes import remote_utils
from TFLibrary.utils.misc_utils import calculate_time
logging.set_verbosity(logging.DEBUG)


class ZMQServer(remote_base.RemoteServer):
    """Server Metrics Implemented Using ZeroMQ"""
    def setup(self):
        # ZeroMQ Context
        context = zmq.Context()

        # Define the socket using the "Context"
        socket = context.socket(zmq.REP)
        socket.bind(self._address)
        logging.info("Binded to %s" % self._address)

        # Save vars
        self._context = context
        self._socket = socket

    def start(self):
        logging.info("Starting Server...")

        while True:
            messages = self._receive()

            # If the message is Hello message, notify the
            # client that the connection is established
            if self._is_hello_messages(messages):
                logging.info("Connected to %s" % messages[1])
                hello_messages = self._get_hello_messages()
                self._send(hello_messages)

            # Messages not to be kwarged-dictionary so that
            # metric calls are more reliable
            if not isinstance(messages, dict):
                raise TypeError("Expected received message to be "
                                "dictionary, found ", type(messages))
            
            # Execute the metrics, return results and throw
            # back the Error messages
            try:
                with calculate_time("Metric", printer=logging.debug):
                    messages = self._call_func(**messages)

            except Exception as e:
                messages = remote_utils.format_error_message(e)

            # Serialize and Send
            self._send(messages)

    def _send(self, messages):
        self._socket.send_pyobj(messages)

    def _receive(self):
        messages_received = self._socket.recv_pyobj()
        return messages_received

    def _is_hello_messages(self, messages):
        if not isinstance(messages, (tuple, list)):
            return False

        if len(messages) != 2:
            raise NotImplementedError

        return messages[0] == remote_utils.HELLO_MSG

    def _get_hello_messages(self):
        return [remote_utils.HELLO_MSG, self.identity]


class ZMQClient(remote_base.RemoteClient):
    def connect(self):
        # ZeroMQ Context
        context = zmq.Context()

        # Define the socket using the "Context"
        socket = context.socket(zmq.REQ)
        socket.connect(self._address)
        logging.info("Connecting to %s" % self._address)

        self._context = context
        self._socket = socket

        # First Send a Hello message to ensure the
        # connection is established
        hello_messages = self._get_hello_messages()
        self._send(hello_messages)
        messages = self._receive()
        if self._is_hello_messages(messages):
            logging.info("Connected to %s" % messages[1])


    def _send(self, messages):
        self._socket.send_pyobj(messages)

    def _receive(self):
        messages_received = self._socket.recv_pyobj()

        if remote_utils.is_error(messages_received):
            logging.fatal(messages_received)

        return messages_received

    def _call(self, **kwargs):
        # Send a "message" using the socket
        with calculate_time("Client", printer=logging.debug):
            self._send(kwargs)
            messages = self._receive()

        return messages

    def _is_hello_messages(self, messages):
        if not isinstance(messages, (tuple, list)):
            return False

        if len(messages) != 2:
            raise NotImplementedError

        return messages[0] == remote_utils.HELLO_MSG

    def _get_hello_messages(self):
        return [remote_utils.HELLO_MSG, self.identity]
