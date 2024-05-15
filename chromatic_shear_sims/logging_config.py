import logging


defaults = {
    "format":'%(asctime)s - %(process)d - %(thread)d - %(module)s - %(levelname)s - %(message)s',
    # "format":'%(asctime)s - %(process)d - %(thread)d - %(name)s - %(module)s - %(levelname)s - %(message)s',
}

# DEFAULT_FORMAT = '%(asctime)s - %(process)d - %(thread)d - %(module)s - %(levelname)s - %(message)s'
DEFAULT_FORMAT = '%(asctime)s - %(process)d - %(module)s - %(levelname)s - %(message)s'

def get_main_handler(name, log_level):
    main_formatter = logging.Formatter(DEFAULT_FORMAT)
    main_filter = logging.Filter(name)

    main_handler = logging.StreamHandler()
    main_handler.setLevel(log_level)
    main_handler.setFormatter(main_formatter)
    main_handler.addFilter(main_filter)

    return main_handler


def get_lib_handler(log_level):
    lib_formatter = logging.Formatter(DEFAULT_FORMAT)
    lib_filter = logging.Filter("chromatic_shear_sims")

    lib_handler = logging.StreamHandler()
    lib_handler.setLevel(log_level)
    lib_handler.setFormatter(lib_formatter)
    lib_handler.addFilter(lib_filter)

    return lib_handler


def setup_main_logger(logger, name, log_level):
    main_handler = get_main_handler(name, log_level)
    logger.addHandler(main_handler)
    logger.setLevel(log_level)

    return


def setup_lib_logger(logger, log_level):
    lib_handler = get_lib_handler(log_level)
    logger.addHandler(lib_handler)
    logger.setLevel(log_level)

    return


def setup_logging(main_logger, root_logger, name, log_level):
    setup_main_logger(main_logger, name, log_level)
    setup_lib_logger(root_logger, log_level)

    return


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



