import logging


defaults = {
    "format":'%(asctime)s - %(process)d - %(thread)d - %(module)s - %(levelname)s - %(message)s',
    # "format":'%(asctime)s - %(process)d - %(thread)d - %(name)s - %(module)s - %(levelname)s - %(message)s',
}


def get_level(log_level):
    match log_level:
        case 0 | logging.ERROR:
            level = logging.ERROR
        case 1 | logging.WARNING:
            level = logging.WARNING
        case 2 | logging.INFO:
            level = logging.INFO
        case 3 | logging.DEBUG:
            level = logging.DEBUG
        case _:
            level = logging.INFO

    return level

