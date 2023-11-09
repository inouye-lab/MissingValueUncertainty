import logging
import os
from threading import Lock, Thread
from typing import List, Optional


class WorkQueue:
    """Simple work queue implementation for sharing task across threads"""

    _work: List[callable]
    """Internal work queue, should not be accessed directly, access via `getTask` and `addTask`"""
    _lock: Lock
    """Internal lock for the work queue"""

    def __init__(self):
        self._work = []
        self._lock = Lock()

    def getTask(self) -> Optional[callable]:
        """Threadsafe method to get and remove next task"""
        self._lock.acquire()
        try:
            if len(self._work) == 0:
                return None
            return self._work.pop()
        finally:
            self._lock.release()

    def addTask(self, task: callable) -> None:
        """Not threadsafe method to add new task"""
        self._work.append(task)


class Worker:
    """Thread that processes data from the work queue"""

    workQueue: WorkQueue
    """Thread will run until the work queue is empty"""
    index: int
    """Queue index for debug printing"""

    def __init__(self, workQueue: WorkQueue, index: int):
        self.workQueue = workQueue
        self.index = index

    def __call__(self, *args, **kwargs):
        """Run tasks in the work queue until the work queue is empty"""
        while True:
            task = self.workQueue.getTask()
            if task is None:
                break
            else:
                task()
        logging.info(f"Worker thread {self.index} stopping")


def distributeTasks(tasks: List[callable], numThreads: int) -> None:
    """
    Distributes the set of tasks over the desired number of threads
    :param tasks:       List of tasks to complete
    :param numThreads:  Number of threads to run them on. If -1, uses the maximum thread count
    """

    # if only a single task, just run it without dealing with threads
    if len(tasks) == 1:
        tasks[0]()
        return

    # -1 for threads means run on "maximum useful threads"
    # technically speaking, our maximum is cpu*core, but that seems harder to fetch
    if numThreads == -1:
        numThreads = os.cpu_count()

    # if we only want one thread, just run all tasks in main
    if numThreads == 1:
        logging.info(f"Running {len(tasks)} tasks on the main thread")
        for task in tasks:
            task()
        return

    # if we are allowed more threads than we have tasks, then just run each task on its own thread
    threads: List[Thread]
    if numThreads > len(tasks):
        logging.info(f"Running {len(tasks)} tasks each on their own thread")
        threads = [Thread(target=task) for task in tasks]
    else:
        # otherwise, make worker queues and queue up worker threads to run experiments
        logging.info(f"Running {len(tasks)} tasks on {numThreads} threads")
        workQueue = WorkQueue()
        for task in tasks:
            workQueue.addTask(task)
        threads = [Thread(target=Worker(workQueue, i+1)) for i in range(numThreads)]

    # threads are set up, so run them and wait for them to complete
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
