import logging

class ANSIFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    green = "\x1b[0;32m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    cyan = "\x1b[0;36m"
    reset = "\x1b[0m"
    header = "[%(name)s] - [%(levelname)s] "
    body = "%(message)s "
    tail = "[%(asctime)s]"

    FORMATS = {
        logging.DEBUG: grey + header + reset + body + cyan + tail + reset,
        logging.INFO: green + header + reset + body + cyan + tail + reset,
        logging.WARNING: yellow + header + reset + body + cyan + tail + reset,
        logging.ERROR: red + header + reset + body + cyan + tail + reset,
        logging.CRITICAL: bold_red + header + reset + body + cyan + tail + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)