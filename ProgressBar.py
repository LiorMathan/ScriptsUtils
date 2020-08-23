import threading
import time
import os
from colored import fg, attr
import random


class ProgressBar:
    def __init__(self, work_needed: int, proggress_getter=None, bar_length=100, decimals=1, end_char='\r', suffix='', prefix='', color_mode=0):
        self.work_needed = work_needed
        self.proggress_getter = proggress_getter
        self.decimals = decimals
        self.bar_length = bar_length
        self.end_char = end_char
        self.stop = False
        self.last_progress = 0
        self.progress = 0
        self.suffix = suffix
        self.prefix = prefix
        if color_mode == 0:
            self.color_mode = random.randint(1, 6)

        else:
            self.color_mode = color_mode 
        
        self.currentcolor = fg(0)
        self.currentcolorlist = []
        self.colorjump = random.randint(0, 10)
        self.filledLength = 0
        self.lastfilledLength = 0

        self.work_done = 0

        self.thread = threading.Thread(target=self.__print_progress__)
        self.thread.setDaemon(True)

        self.clear_command = ''

        if os.sys.platform == "linux" or os.sys.platform == "linux2":
            self.clear_command = 'clear'
        elif os.sys.platform == "darwin":
            pass
        elif os.sys.platform == "win32":
            self.clear_command = 'cls'


    def init_periodic(self):
        self.thread.start()


    def print_progress(self, work_done, suffix='', prefix=''):
        if suffix == '':
            suffix = self.suffix

        if prefix == '':
            prefix = self.prefix     

        self.work_done = work_done
        self.progress = ("{0:." + str(self.decimals) + "f}").format(self.get_progress())

        self.filledLength = int(self.bar_length * work_done // self.work_needed)
        

        if self.color_mode == 1:
            if self.filledLength is not self.lastfilledLength:
                self.currentcolor = fg(random.randint(0, 255))

            bar = self.currentcolor + '█' * self.filledLength + attr(0) + '-' * (self.bar_length - self.filledLength)
        
        elif self.color_mode == 2:
            bar = ''
            for i in range(0, self.filledLength):
                bar += fg(random.randint(0, 255)) + '█' + attr(0)

            bar += '-' * (self.bar_length - self.filledLength)

        elif self.color_mode == 3:
            bar = fg(random.randint(0, 255)) + '█' * self.filledLength + attr(0) + '-' * (self.bar_length - self.filledLength)

        elif self.color_mode == 4:
            if self.filledLength is not self.lastfilledLength:
                for i in range(self.filledLength - self.lastfilledLength):
                    self.currentcolorlist.append(fg(random.randint(0, 255)))

            bar = ''
            for i in range(0, self.filledLength):
                bar += self.currentcolorlist[i] + '█' + attr(0)

            bar += '-' * (self.bar_length - self.filledLength)

        elif self.color_mode == 5:
            if self.filledLength is not self.lastfilledLength:
                for i in range(self.filledLength - self.lastfilledLength):
                    self.currentcolorlist.append(random.randint(0, 255))
                self.currentcolorlist.sort()
                
            bar = ''
            for i in range(0, self.filledLength):
                bar += fg(self.currentcolorlist[i]) + '█' + attr(0)

            bar += '-' * (self.bar_length - self.filledLength)

        else:
            bar = '█' * self.filledLength + '-' * (self.bar_length - self.filledLength)

        self.lastfilledLength = self.filledLength
        os.system(self.clear_command)
        print(f'\r{prefix} Progress |{bar}| {self.progress}% Complete {suffix}', end=self.end_char)


    def get_progress(self):
        return 100 * (self.work_done / float(self.work_needed))


    def __print_progress__(self):
        while not self.is_done() and not self.stop:
            self.print_progress(self.proggress_getter(), suffix=self.suffix, prefix=self.prefix)
            time.sleep(0.5)


    def stop_periodic(self):
        if self.thread.isAlive():
            self.thread.join()
        self.stop = True


    def is_done(self):
        return self.get_progress() == 100.0

    
def main():
    p = ProgressBar(1000, getter)
    p.init_periodic()
    
    while not p.is_done():
        pass


progress = 0
def getter():
    global progress

    progress += 1
    return progress

if __name__ == '__main__':
    main()