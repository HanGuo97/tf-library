"""lbbroker3: Load balancing broker using zloop in Python"""

from __future__ import print_function

import time
import threading
from collections import namedtuple

import zmq
from zmq.eventloop.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream

from TFLibrary.NewTuner import utils

NBR_CLIENTS = 10
NBR_WORKERS = 3
ENCODE_METHOD = "ascii"

ServerMessages = namedtuple(
    "ServerMessages",
    ("worker_address",
     "client_address",
     "reply"))

ClientMessages = namedtuple(
    "ClientMessages",
    ("client_address", "request"))


def encode(string):
    return string.encode(ENCODE_METHOD)


def decode(string):
    return string.decode(ENCODE_METHOD)


def worker_thread(worker_url, i):
    """ Worker using REQ socket to do LRU routing """
    context = zmq.Context.instance()

    socket = context.socket(zmq.REQ)

    # set worker identity
    socket.identity = (u"Worker-%d" % (i)).encode('ascii')

    socket.connect(worker_url)

    # Tell the broker we are ready for work
    socket.send(utils.HELLO_MESSAGE)

    try:
        while True:

            address, empty, request = socket.recv_multipart()

            print("%s: %s\n" % (socket.identity.decode('ascii'),
                                utils.deserialize_pyobj(request)), end='')

            socket.send_multipart([address, utils.EMPTY, utils.serialize_pyobj('OK')])

    except zmq.ContextTerminated:
        # context terminated so quit silently
        return


def client_thread(client_url, i):
    """ Basic request-reply client using REQ socket """
    context = zmq.Context.instance()

    socket = context.socket(zmq.REQ)

    # Set client identity. Makes tracing easier
    socket.identity = (u"Client-%d" % (i)).encode('ascii')

    socket.connect(client_url)

    #  Send request, get reply
    socket.send(utils.serialize_pyobj("HELLO"))
    reply = socket.recv()

    print("%s: %s\n" % (socket.identity.decode('ascii'),
                        utils.deserialize_pyobj(reply)), end='')


class LRUQueue(object):
    """LRUQueue class using ZMQStream/IOLoop for event dispatching"""

    def __init__(self, backend_socket, frontend_socket):
        self._workers = []
        self.client_nbr = NBR_CLIENTS

        self._backend = ZMQStream(backend_socket)
        self._frontend = ZMQStream(frontend_socket)
        self._backend.on_recv(self._handle_backend)

        self._loop = IOLoop.instance()

    @property
    def available_workers(self):
        return len(self._workers)
        """Serialize worker and Client"""

    def _handle_backend(self, messages):
        if self.available_workers >= NBR_WORKERS:
            raise ValueError("Available Workers Wrong")

        # Queue worker address for LRU routing
        (server_says_hello,
         server_messages) = self.receive_from_servers(messages)
        # add worker back to the list of workers
        self._workers.append(server_messages.worker_address)

        # Third frame is READY or else a client reply address
        # If client reply, send rest back to frontend
        if not server_says_hello:
            self.send_to_clients(server_messages)

            self.client_nbr -= 1

            if self.client_nbr == 0:
                # Exit after N messages
                self._loop.add_timeout(time.time() + 1, self._loop.stop)

        if self.available_workers == 1:
            # on first recv, start accepting frontend messages
            self._frontend.on_recv(self._handle_frontend)

    def _handle_frontend(self, messages):
        # Receive and Process Messages
        client_messages = self.receive_from_clients(messages)
        #  Dequeue and drop the next worker address
        worker_address = self._workers.pop()
        # Route the messages to Servers
        self.send_to_servers(
            worker_address=worker_address,
            client_messages=client_messages)

        if self.available_workers == 0:
            # stop receiving until workers become available again
            self._frontend.stop_on_recv()

    def receive_from_clients(self, messages):
        if len(messages) != 3:
            raise ValueError("`len(messages)` should be 3")

        # Now get next client request, route to LRU worker
        # Client request is [address][empty][request]
        client_addr, empty, request = messages
        utils.assert_empty(empty)

        return ClientMessages(
            client_address=client_addr,
            request=utils.deserialize_pyobj(request))


    def receive_from_servers(self, messages):
        if len(messages) not in [3, 5]:
            raise ValueError("`len(messages)` should be in [3, 5]")

        # Queue worker address for LRU routing
        worker_address, empty, client_address = messages[:3]
        utils.assert_empty(empty)

        # Third frame is READY or else a client reply address
        # If client reply, send rest back to frontend
        server_says_hello = utils.is_hello_message(client_address)
        if not server_says_hello:
            empty, reply = messages[3:]
            utils.assert_empty(empty)
        else:
            reply = None

        return server_says_hello, ServerMessages(
            worker_address=worker_address,
            client_address=client_address,
            reply=utils.deserialize_pyobj(reply) if reply else None)


    def send_to_clients(self, server_messages):
        if not isinstance(server_messages, ServerMessages):
            raise TypeError

        self._frontend.send_multipart([
            server_messages.client_address, utils.EMPTY,
            utils.serialize_pyobj(server_messages.reply)])


    def send_to_servers(self, worker_address, client_messages):
        if not isinstance(client_messages, ClientMessages):
            raise TypeError

        self._backend.send_multipart([
            worker_address, utils.EMPTY,
            client_messages.client_address, utils.EMPTY,
            utils.serialize_pyobj(client_messages.request)])


def main():
    """main method"""

    url_worker = "ipc://backend.ipc"
    url_client = "ipc://frontend.ipc"

    # Prepare our context and sockets
    context = zmq.Context()
    frontend = context.socket(zmq.ROUTER)
    frontend.bind(url_client)
    backend = context.socket(zmq.ROUTER)
    backend.bind(url_worker)

    # create workers and clients threads
    for i in range(NBR_WORKERS):
        thread = threading.Thread(target=worker_thread, args=(url_worker, i, ))
        thread.daemon = True
        thread.start()

    for i in range(NBR_CLIENTS):
        thread_c = threading.Thread(target=client_thread,
                                    args=(url_client, i, ))
        thread_c.daemon = True
        thread_c.start()

    # create queue with the sockets
    queue = LRUQueue(backend, frontend)

    # start reactor
    IOLoop.instance().start()


if __name__ == "__main__":
    main()