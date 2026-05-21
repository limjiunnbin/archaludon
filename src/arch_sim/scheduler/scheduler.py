class TaskScheduler:
    def init(self, core_num):
        self.bitmap = [0] * core_num