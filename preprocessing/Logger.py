class Logger:
    
    logger = None

    def log(self, *log_message):
        print(*log_message)

    @classmethod
    def getLogger(cls):
        if cls.logger is None:
            cls.logger = Logger()

        return cls.logger
