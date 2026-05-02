"""
Helper functions for face recognition module.
"""

def log_output(message: str):
    """
    Log a message to the console with a consistent prefix.

    Parameters:
        message (str): The message to log.

    Returns:
        None. The function prints the message to the console.
    """
    print(f"{message}")


def log_error(message: str):
    """
    Log an error message to the console with a consistent prefix.

    Parameters:
        message (str): The error message to log.

    Returns:
        None. The function prints the error message to the console.
    """
    log_output(f"\033[91mError: {message}\033[0m")


def log_warning(message: str):
    """
    Log a warning message to the console with a consistent prefix.

    Parameters:
        message (str): The warning message to log.

    Returns:
        None. The function prints the warning message to the console.
    """
    log_output(f"\033[93mWarning: {message}\033[0m")


def log_info(message: str):
    """
    Log an informational message to the console with a consistent prefix.

    Parameters:
        message (str): The informational message to log.

    Returns:
        None. The function prints the informational message to the console.
    """
    log_output(f"\033[94mInfo: {message}\033[0m")