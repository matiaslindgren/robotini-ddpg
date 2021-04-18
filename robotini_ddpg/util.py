import logging
import math
import os
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

def format_metric(m):
    return "\t\t {:s} = {:.3f}".format(m.name, m.result().numpy())

def metric_value_from_filename(path, key):
    v = next(v for v in path.name.split('_') if v.startswith(key))
    v = v.lstrip(key)
    return float(v)

def get_best_saved_policy(policy_dir, metric="AvgEvalReturn"):
    """
    E.g. from a policy_dir containing
        SavedPolicy_Step1_AvgEvalReturn13.4
        SavedPolicy_Step2_AvgEvalReturn73.1
        SavedPolicy_Step9_AvgEvalReturn100
        SavedPolicy_Step3_AvgEvalReturn0
    return 100.0
    """
    if not os.path.isdir(policy_dir):
        return None, -math.inf
    get_value = lambda path: metric_value_from_filename(path, metric)
    best_policy_path = max(os.scandir(policy_dir), key=get_value)
    return best_policy_path.path, get_value(best_policy_path)

def find_first(metrics, name):
    return next((m for m in metrics if m.name == name), None)
