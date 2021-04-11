import time

def sleep_until(t):
    did_sleep = False
    while True:
        left = t - time.perf_counter()
        if left <= 0:
            return did_sleep
        time.sleep(left)
        did_sleep = True
