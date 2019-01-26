class RemoteServer(object):
    """Metrics Server"""
    def __init__(self, func, address, identity="server"):
        if not callable(func):
            raise TypeError("`func` must be Callable, "
                            "but found ", type(func))

        self._func = func
        self._address = address
        self._identity = identity

        self.setup()

    @property
    def identity(self):
        return self._identity
    
    def setup(self):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def _send(self, messages):
        raise NotImplementedError

    def _receive(self):
        raise NotImplementedError

    def _call_func(self, *_sentinel, **kwargs):
        if _sentinel:
            raise ValueError("`RemoteServer`._call_func should "
                             "be called with keyword args only")

        return self._func(**kwargs)



class RemoteClient(object):
    """Metrics Client"""
    def __init__(self, address, identity="client"):
        self._address = address
        self._identity = identity

        self.connect()

    @property
    def identity(self):
        return self._identity

    def connect(self):
        raise NotImplementedError

    def _send(self, messages):
        raise NotImplementedError

    def _receive(self):
        raise NotImplementedError

    def __call__(self, *_sentinel, **kwargs):
        if _sentinel:
            raise ValueError("`RemoteClient`.__call__ should "
                             "be called with keyword args only")

        return self._call(**kwargs)

    def _call(self, **kwargs):
        raise NotImplementedError
