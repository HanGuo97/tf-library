"""lbbroker3: Load balancing broker using zloop in Python"""

from __future__ import print_function

import time
from absl import logging
from zmq.eventloop.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream
from TFLibrary.NewTuner import utils
from TFLibrary.NewTuner import envelops
logging.set_verbosity(logging.INFO)


class LRUQueue(object):
    """LRUQueue class using ZMQStream/IOLoop for event dispatching"""

    def __init__(self,
                 backend_socket,
                 frontend_socket,
                 identity="LRUQueue"):

        loop = IOLoop.instance()
        backend = ZMQStream(backend_socket)
        frontend = ZMQStream(frontend_socket)
        backend.on_recv(self._handle_backend)
        frontend.on_recv(self._handle_frontend)

        self._loop = loop
        self._backend = backend
        self._frontend = frontend
        self._identity = identity

        self._results = []
        self._workers_busy = []
        self._workers_free = []
        self._requests_pending = []
        self._requests_running = []

    @property
    def identity(self):
        return self._identity

    def exit(self):
        # Exit after N messages
        self._loop.add_timeout(time.time() + 1, self._loop.stop)

    def send_requests_to_servers(self):
        # Send as many requests to servers as possible
        # until either no requests left or no available workers
        while len(self._requests_pending) and len(self._workers_free):
            # Dequeue and drop the next worker address and request
            server_address = self._workers_free.pop()
            client_envelop = self._requests_pending.pop()
            
            # Route the messages to Servers
            self.send_to_servers(
                header="POST",
                server_address=server_address,
                contents=client_envelop.contents)

            # Put them into busy queue
            self._workers_busy.append(server_address)
            self._requests_running.append(client_envelop)
        

    def _handle_backend(self, messages):
        """Handle Backend

        Cases:
            header: HELLO
                The server sent a message to verify connection.
                Reply the server with HELLO.

            header: READY
                The server is ready for receiving workloads.
                Append the worker into the queue.

            header: POST:
                The server finished its last work and sent the
                results. It also signals it's ready for new works.
                Put the results into the queue, and append the
                workder into the queue.

        """
        # Queue worker address for LRU routing
        (server_address,
         server_envelop) = self.receive_from_servers(messages)

        # Name
        server_info = "{} ({})".format(
            server_envelop.identity, server_address)

        if server_envelop.header == "HELLO":
            logging.info("Server {} Says HELLO".format(server_info))
            self.send_to_servers(header="HELLO",
                                 server_address=server_address)

            return

        if server_envelop.header == "READY":
            logging.info("Server {} is READY".format(server_info))

        if server_envelop.header == "POST":
            logging.info("Server {} finished".format(server_info))
            self._results.append({"contents": server_envelop.contents})


        # add worker back to the list of workers
        self._workers_free.append(server_address)
        # Send requests if possible
        self.send_requests_to_servers()

    def _handle_frontend(self, messages):
        """Handle Frontend

        Cases:
            header: HELLO
                The client tries to verify connection.
                Reply HELLO to the client.

            header: GET
                The client wants to query status.
                Reply with corresponding status.

            header: POST
                The client makes requests.
                Add the requests into queue.

        """
        # Receive and Process Messages
        (client_address,
         client_envelop) = self.receive_from_clients(messages)

        # Name
        client_info = "{} ({})".format(
            client_envelop.identity, client_address)
        
        if client_envelop.header == "HELLO":
            logging.info("Client {} Says HELLO".format(client_info))
            self.send_to_clients(
                header="HELLO",
                client_address=client_address)

            return

        if client_envelop.header == "GET":
            logging.info("Client {} sent GET".format(client_info))
            self.send_to_clients(
                header=client_envelop.header,
                client_address=client_address,
                contents={
                    "# Results": len(self._results),
                    "# Free Workers": len(self._workers_free),
                    "# Busy Workers": len(self._workers_busy),
                    "# Pending Requests": len(self._requests_pending),
                    "# Running Requests": len(self._requests_running),
                })

        if client_envelop.header == "POST":
            logging.info("Client {} sent POST".format(client_info))
            self._requests_pending.append(client_envelop)
            self.send_to_clients(
                header=client_envelop.header,
                client_address=client_address,
                contents="Request {} Received".format(
                    client_envelop.contents))


        # Send requests if possible
        self.send_requests_to_servers()

    def receive_from_clients(self, messages):
        """Receive messages from clients

        Received Messages:
            client_address
            EMPTY,
            ClientEnvelop,
        """
        if len(messages) != 3:
            raise ValueError("`len(messages)` should be 3")

        client_address, empty, serialied_envelop = messages
        utils.assert_empty(empty)
        client_envelop = envelops.deserialize_client_envelop(
            serialied_envelop=serialied_envelop)

        return client_address, client_envelop


    def receive_from_servers(self, messages):
        """Receive messages from servers

        Received Messages:
            server_address,
            EMPTY,
            ServerEnvelop
        
        where `client_address` can also be EMPTY is the message
        is for internal communication only
        """
        if len(messages) != 3:
            raise ValueError("`len(messages)` should be 3")

        # Queue worker address for LRU routing
        server_address, empty, serialied_envelop = messages
        utils.assert_empty(empty)
        server_envelop = envelops.deserialize_server_envelop(
            serialied_envelop=serialied_envelop)

        return server_address, server_envelop


    def send_to_clients(self, header, client_address, contents=None):
        """Send messages to clients

        Messages to Send:
            client_address,
            EMPTY,
            ClientEnvelop
        """
        if client_address is None:
            raise ValueError("`client_address` cannot be None")

        envelop = envelops.make_client_envelop(
            header=header,
            contents=contents,
            identity=self.identity,
            serialize=True)

        self._frontend.send_multipart([client_address,
                                       utils.EMPTY,
                                       envelop])

    def send_to_servers(self, header, server_address, contents=None):
        """Send messages to servers

        Messages to Send:
            server_address,
            EMPTY,
            ServerEnvelop

        where `client_address` can also be EMPTY if the message is
        for internal communication only.
        """
        if server_address is None:
            raise ValueError("`server_address` cannot None")

        envelop = envelops.make_server_envelop(
            header=header,
            contents=contents,
            identity=self.identity,
            serialize=True)

        self._backend.send_multipart([server_address,
                                      utils.EMPTY,
                                      envelop])
