"""lbbroker3: Load balancing broker using zloop in Python"""

from __future__ import print_function

import time
import threading
from collections import namedtuple

import zmq
from zmq.eventloop.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream

from TFLibrary.NewTuner import utils
from TFLibrary.NewTuner import servers

NBR_CLIENTS = 10
NBR_WORKERS = 3
ENCODE_METHOD = "ascii"

ServerEnvelop = namedtuple(
    "ServerEnvelop",
    ("server_address",
     "client_address",
     "server_identity",
     "contents"))

ClientEnvelop = namedtuple(
    "ClientEnvelop",
    ("client_address",
     "client_identity",
     "contents"))


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
            raise ValueError("Available Workers Wrong, %d != %d" % (NBR_CLIENTS, self.available_workers))

        # Queue worker address for LRU routing
        server_envelop = self.receive_from_servers(messages)

        # Third frame is READY or else a client reply address
        # If client reply, send rest back to frontend
        if server_envelop.contents == utils.HELLO_MESSAGE:
            self.send_to_servers(
                contents=utils.HELLO_MESSAGE,
                server_address=server_envelop.server_address,
                client_address=None)

            return
        
        # add worker back to the list of workers
        self._workers.append(server_envelop.server_address)

        if server_envelop.contents != utils.READY_MESSAGE:
            self.send_to_clients(
                contents=server_envelop.contents,
                client_address=server_envelop.client_address)

            self.client_nbr -= 1

            if self.client_nbr == 0:
                # Exit after N messages
                self._loop.add_timeout(time.time() + 1, self._loop.stop)

        if self.available_workers == 1:
            # on first recv, start accepting frontend messages
            self._frontend.on_recv(self._handle_frontend)

    def _handle_frontend(self, messages):
        # Receive and Process Messages
        client_envelop = self.receive_from_clients(messages)
        
        # Third frame is READY or else a client reply address
        # If client reply, send rest back to frontend
        if client_envelop.contents == utils.HELLO_MESSAGE:
            self.send_to_clients(
                contents=utils.HELLO_MESSAGE,
                client_address=client_envelop.client_address)

            return

        #  Dequeue and drop the next worker address
        server_address = self._workers.pop()
        
        # Route the messages to Servers
        self.send_to_servers(
            contents=client_envelop.contents,
            server_address=server_address,
            client_address=client_envelop.client_address)

        if self.available_workers == 0:
            # stop receiving until workers become available again
            self._frontend.stop_on_recv()

    def receive_from_clients(self, messages):
        if len(messages) != 4:
            raise ValueError("`len(messages)` should be 3")

        # Now get next client request, route to LRU worker
        # Client request is [address][empty][identity][request]
        client_address, empty, client_identity, contents = messages
        utils.assert_empty(empty)
        contents = utils.deserialize_pyobj(contents)

        return ClientEnvelop(
            client_address=client_address,
            client_identity=client_identity,
            contents=contents)


    def receive_from_servers(self, messages):
        if len(messages) != 5:
            raise ValueError("`len(messages)` should be 5")

        # Queue worker address for LRU routing
        (server_address, empty,
         client_address,
         server_identity,
         contents) = messages
        
        utils.assert_empty(empty)
        contents = utils.deserialize_pyobj(contents)
        if client_address == utils.EMPTY:
            if contents not in [utils.HELLO_MESSAGE,
                                utils.READY_MESSAGE]:
                raise ValueError(contents)
            
            client_address = None

        return ServerEnvelop(
            server_address=server_address,
            client_address=client_address,
            server_identity=server_identity,
            contents=contents)


    def send_to_clients(self, contents, client_address):
        if client_address is None:
            raise ValueError("`client_address` is None")

        self._frontend.send_multipart([
            client_address,
            utils.EMPTY,
            utils.serialize_pyobj(contents)])

    def send_to_servers(self, contents, server_address, client_address):
        if server_address is None:
            raise ValueError("`server_address` is None")

        self._backend.send_multipart([
            server_address,
            utils.EMPTY,
            client_address or utils.EMPTY,
            utils.EMPTY,
            utils.serialize_pyobj(contents)])


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
        thread_s = threading.Thread(target=servers.BasicServer,
                                    kwargs={"address": url_worker,
                                            "identity": "server-%d" % i})
        thread_s.daemon = True
        thread_s.start()

    for i in range(NBR_CLIENTS):
        thread_c = threading.Thread(target=servers.BasicClient,
                                    kwargs={"address": url_client,
                                            "identity": "client-%d" % i})
        thread_c.daemon = True
        thread_c.start()

    # create queue with the sockets
    LRUQueue(backend, frontend)

    # start reactor
    IOLoop.instance().start()


if __name__ == "__main__":
    main()
