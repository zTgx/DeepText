import psutil
import os

def get_ram_usage():
    return psutil.Process(os.getpid()).memory_info().rss >> 20


if __name__ == '__main__':
    ram_usage = get_ram_usage()
    print(f"RAM used: {ram_usage} MB")