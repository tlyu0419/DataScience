from threading import Thread
from multiprocessing import Process
from multiprocessing import Pool

def loop():
    while True:
        pass

if __name__ == '__main__':

    for i in range(16):
        t = Process(target=loop)
        t.start()

    while True:
        pass