HELLO_MSG = "HELLO"
ERROR_PREFIX = "ERROR"


def format_error_message(e):
    return "%s [%s] %s" % (ERROR_PREFIX,
                           type(e).__name__,
                           str(e))


def is_error(messages):
    if not isinstance(messages, str):
        return
    return messages.startswith(ERROR_PREFIX)
