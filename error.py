class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class StructDecError(Error):

    def __init__(self, *args, **kwargs):
        default_message = 'Incorrect SD value!'

        if not (args or kwargs):
            args = (default_message,)

        super().__init__(*args, **kwargs)


class DetError(Error):

    def __init__(self, *args, **kwargs):
        default_message = 'Incorrect Det value!'

        if not (args or kwargs):
            args = (default_message,)

        super().__init__(*args, **kwargs)


class ArityError(Error):

    def __init__(self, *args, **kwargs):
        default_message = 'Arity is less than 2 or greater than the possible conjunction assignments!'

        if not (args or kwargs):
            args = (default_message,)

        super().__init__(*args, **kwargs)


class NoRandomness(Error):

    def __init__(self, *args, **kwargs):
        default_message = 'Low randomness in this configuration!'

        if not (args or kwargs):
            args = (default_message,)

        super().__init__(*args, **kwargs)


class RootVarError(Error):

    def __init__(self, *args, **kwargs):
        default_message = 'Root var not in scope!'

        if not (args or kwargs):
            args = (default_message,)

        super().__init__(*args, **kwargs)


class NoPartitioningFound(Error):

    def __init__(self, *args, **kwargs):
        default_message = 'No partitioning found! Try to change min_part_inst and conj_len hyper-parameters.'

        if not (args or kwargs):
            args = (default_message,)

        super().__init__(*args, **kwargs)