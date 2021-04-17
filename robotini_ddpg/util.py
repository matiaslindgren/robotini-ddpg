import logging
import queue
import time

def sleep_until(t):
    did_sleep = False
    while True:
        left = t - time.perf_counter()
        if left <= 0:
            return did_sleep
        time.sleep(left)
        did_sleep = True

def drain_queue(q):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return

def join_or_terminate(proc, timeout):
    proc.join(timeout=timeout)
    if proc.exitcode is None:
        logging.warning("Process %s did not terminate during join, killing process", proc.name)
        proc.terminate()

def reset_logger(name=None):
    remove_logger(name)
    l = logging.getLogger(name)
    l.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt=r"%(asctime)s:%(levelname)s:%(message)s", datefmt=r"%s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    l.addHandler(sh)
    return l

def remove_logger(name):
    l = logging.getLogger(name)
    while l.handlers:
        l.handlers.pop()
