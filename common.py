import logging
import sys


def ConfigRootLogger(name='', version=None, level="info"):
    logger = logging.getLogger(name)
    if level == "debug":
        logger.setLevel(logging.DEBUG)
    elif level == "info":
        logger.setLevel(logging.INFO)
    elif level == "warning":
        logger.setLevel(logging.WARNING)
    format = logging.Formatter("T[%(asctime)s]-V[{}]-POS[%(module)s."
                                      "%(funcName)s(line %(lineno)s)]-PID[%(process)d] %(levelname)s"
                                      ">>> %(message)s  ".format(version),"%H:%M:%S")
    stdout_format = logging.Formatter("T[%(asctime)s]-V[{}]-POS[%(module)s."
                                      "%(funcName)s(line %(lineno)s)]-PID[%(process)d] %(levelname)s"
                                      ">>> %(message)s  ".format(version),"%H:%M:%S")

    file_handler = logging.FileHandler("log.txt")
    file_handler.setFormatter(format)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(stdout_format)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


ConfigRootLogger("main","1",level="info")
logger = logging.getLogger("main")