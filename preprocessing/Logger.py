class Logger:
    
    logger = None

    def __init__(self) -> None:
        self.log_messages = []

    def log(self, *log_messages):
        print(*log_messages)

        to_store = []
        for log_message in log_messages:
            to_store.append(str(log_message))
        self.log_messages.append(" ".join(to_store))

    def get_log_messages(self):
        return "\n".join(self.log_messages)

    @classmethod
    def get_logger(cls):
        if cls.logger is None:
            cls.logger = Logger()

        return cls.logger
