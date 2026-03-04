import multiprocessing as mp


class Parallel(object):

    max_cores = mp.cpu_count()
  
    def __init__(
        self,
        n_cores: int = mp.cpu_count() - 1,
        initializer=None,
        initargs=(),
        activate: bool = False,
    ):
        self.n_cores = n_cores
        self.initializer = initializer
        self.initargs = initargs
        if activate:
            self.activate()
        else:
            self.__pool = None

    def activate(self):
        self.__pool = mp.Pool(
            self.n_cores,
            self.initializer,
            self.initargs,
        )

    def perform(self, func, *args):
        assert callable(func), "func must be a callable"
        assert len(args) > 0, "func inputs are not provided"
        if len(args) == 1:
            return self.__pool.map(func, *args)
        inputs = zip(*args)
        return self.__pool.starmap(func, inputs)

    def deactivate(self):
        self.__pool.close()
        self.__pool.join()
        self.__pool = None
