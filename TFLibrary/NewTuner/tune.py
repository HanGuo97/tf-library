import zmq
import argparse
import threading
from absl import logging
from zmq.eventloop.ioloop import IOLoop
from TFLibrary.NewTuner import proxy
from TFLibrary.NewTuner import basic

URL_WORKER = "ipc://backend.ipc"
URL_CLIENT = "ipc://frontend.ipc"
NUM_CLIENTS = 10
NUM_WORKERS = 3
logging.set_verbosity(logging.INFO)


def start_proxy():
    context = zmq.Context()
    frontend = context.socket(zmq.ROUTER)
    backend = context.socket(zmq.ROUTER)
    frontend.bind(URL_CLIENT)
    backend.bind(URL_WORKER)

    # create queue with the sockets
    proxy.LRUQueue(backend, frontend)

    # start reactor
    IOLoop.instance().start()


def start_servers(n=NUM_WORKERS):
    # create workers and clients threads
    threads = []
    for i in range(n):
        thread_s = threading.Thread(target=basic.BasicServer,
                                    kwargs={"address": URL_WORKER,
                                            "identity": "server-%d" % i})
        thread_s.daemon = True
        thread_s.start()
        threads.append(thread_s)

    # Wait for all of them to finish
    for x in threads:
        x.join()


def make_request():
    client = basic.BasicClient(address=URL_CLIENT)
    client.send(header="POST", contents="DUMMY")
    envelop = client.receive()
    logging.info("{}: {}".format(client.identity,
                                 envelop.contents))


def get_status():
    client = basic.BasicClient(address=URL_CLIENT)
    client.send(header="GET", contents="STATUS")
    envelop = client.receive()

    logging.info("{} received response".format(client.identity))
    for key, val in envelop.contents.items():
        print("\t{}:\t{}".format(key, val))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str, default=None)
    # -----------------------------------------
    FLAGS, unparsed = parser.parse_known_args()

    if unparsed:
        raise ValueError(unparsed)

    if FLAGS.type == "proxy":
        start_proxy()
    if FLAGS.type == "server":
        start_servers()
    if FLAGS.type == "request":
        make_request()
    if FLAGS.type == "status":
        get_status()
