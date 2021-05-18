from enum import Enum

from .convert_for_loops_to_while import _make_for_loops_while
from .move_var_decls_to_top_of_scope import _move_var_decls_to_top_of_scope
from .name_function_applications import _name_unnamed_applications


class Transformations(Enum):
    MAKE_FOR_LOOPS_WHILE = _make_for_loops_while
    MOVE_DECLS_TO_TOP_OF_SCOPE = _move_var_decls_to_top_of_scope
    NAME_ALL_FUNCTION_APPLICATIONS = _name_unnamed_applications
