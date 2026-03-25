import multiprocessing as mp
from typing import Callable


class Parallel(object):
    """
    Utility wrapper for paralellizing tasks across multiple CPU cores.

    This class simplifies the management of a `multiprocessing.Pool`,
    providing a clean interface to activate, execute, and shut down
    parallel workers.

    Parameters
    ----------
    n_cores : int, default multiprocessing.cpu_count() - 1
        Number of cores to use.
    initializer : callable, optional
        A function to be called by each worker process when it starts.
    initargs : tuple, optional
        Arguments to be passed to the initializer.
    activate : bool, default False
        Whether to start the pool immediately upon instantiation.
    """

    max_cores = mp.cpu_count()
    """
    Total number of logical CPU cores available on the system.

    Calculated by ``multiprocessing.cpu_count()``
    """

    def __init__(
        self,
        n_cores: int = mp.cpu_count() - 1,
        initializer=None,
        initargs=(),
        activate: bool = False,
    ):
        """
        Initialize the Parallel handler.

        Parameters
        ----------
        n_cores : int, default mp.cpu_count() - 1
            Number of cores to use.
        initializer : callable, optional
            A function to be called by each worker process when it starts.
        initargs : tuple, optional
            Arguments to be passed to the initializer.
        activate : bool, default False
            Whether to start the pool immediately upon instantiation.
        """
        self.n_cores = n_cores
        self.initializer = initializer
        self.initargs = initargs
        if activate:
            self.activate()
        else:
            self.__pool = None

    def activate(self):
        """
        Create and start the multiprocessing pool.

        This method allocates system resources and spawns the worker
        processes. It must be called before ``perform`` can be used.
        """
        self.__pool = mp.Pool(
            self.n_cores,
            self.initializer,
            self.initargs,
        )

    def perform(self, func: Callable, *args):
        """
        Execute a function in parallel.

        If a single iterable is provided in ``args``, ``pool.map`` is used.
        If multiple iterables are provided, they are zipped together
        and executed via ``pool.starmap``.

        Parameters
        ----------
        func : callable
            The function to execute in parallel.
        *args : iterable
            The input data to be distributed among workers.

        Returns
        -------
        list
            The collected results from all worker processes.

        Raises
        ------
        AssertionError
            If ``func`` is not callable or no arguments are provided.
        """
        assert callable(func), "func must be a callable"
        assert len(args) > 0, "func inputs are not provided"
        if len(args) == 1:
            return self.__pool.map(func, *args)
        inputs = zip(*args)
        return self.__pool.starmap(func, inputs)

    def deactivate(self):
        """
        Gracefully shut down the process pool.

        Closes the pool to new tasks and waits for all current worker
        processes to exit. Resources are freed and the internal pool
        is set back to None.
        """
        self.__pool.close()
        self.__pool.join()
        self.__pool = None
