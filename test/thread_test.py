# SuperFastPython.com
# report the default number of worker threads on your system
from concurrent.futures import ThreadPoolExecutor
# create a thread pool with the default number of worker threads
pool = ThreadPoolExecutor()
# report the number of worker threads chosen by default
print(pool._max_workers)