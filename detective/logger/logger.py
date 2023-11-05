import logging

class Logger:
    _instance = None
    logger = None

    @staticmethod
    def initialize(logger):
        if Logger._instance is None:
            Logger._instance = Logger()
            Logger._instance.logger = logger
        return Logger._instance

    @staticmethod
    def get_instance():
        return Logger._instance

    @staticmethod
    def debug(message):
        if Logger._instance is not None:
            Logger._instance.logger.debug(message)

    @staticmethod
    def info(message):
        if Logger._instance is not None:
            Logger._instance.logger.info(message)

    @staticmethod
    def warning(message):
        if Logger._instance is not None:
            Logger._instance.logger.warning(message)

    @staticmethod
    def error(message):
        if Logger._instance is not None:
            Logger._instance.logger.error(message)