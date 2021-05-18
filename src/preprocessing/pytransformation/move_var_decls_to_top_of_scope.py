import ast


def _move_var_decls_to_top_of_scope(node, vars_already_scoped={}):
    """Initilizes all variables used in a scope to None
    at the begining of that scope.
    Parameters:
        node: ast node
        vars_already_scoped: hash table of variables that have already been
            reinitilized
    Returns:
        node with variables initialized at top of scope.
        vars_already_scoped: hashtable with variables already moved."""

    current_scope_new_vars = []

    if isinstance(node, ast.FunctionDef) or isinstance(node, ast.Module):
        for child_node in ast.iter_child_nodes(node):
            new_vars, vars_already_scoped = _find_vars_in_same_scope(
                                            child_node, vars_already_scoped)
            current_scope_new_vars = current_scope_new_vars + new_vars
        current_scope_new_vars.reverse()
        _move_initilizations_to_top_of_scope(node, current_scope_new_vars)

    return node, vars_already_scoped


def _find_vars_in_same_scope(node, vars_already_scoped):
    """finds all vars that need to be moved for reinitilization in this scope.
    Note: Will exceed maximum stack depth if really long AST's are used.
        Parameters:
            node: ast node to find all child variables
            vars_previously_declared: vars that have already been declared at
            a higher scope so no need to reinitilize.
        Returns:
            list of all vars that need to be reinitilized
    """
    def find_names_helper(node):
        """Helper function to recur down assignments and find names.
        Assignments can be subscripts or function calls, etc..."""
        if isinstance(node, ast.Name):
            return [node.id]
        else:
            names = []
            for child_node in ast.iter_child_nodes(node):
                names.extend(find_names_helper(child_node))
            return names

    current_scope_new_vars = []
    is_module = isinstance(node, ast.Module)
    is_function = isinstance(node, ast.FunctionDef)
    is_assign = isinstance(node, ast.Assign)
    is_for = isinstance(node, ast.For)
    # Add current node if it's a variable assignment or target in for loop, and
    # it hasn't been added to the initilization list.
    if is_assign or is_for:
        if is_for:
            targets = [node.target]
        else:
            targets = node.targets
        for target in targets:
            for var_name in find_names_helper(target):
                if var_name not in vars_already_scoped:
                    current_scope_new_vars.append(var_name)
                    vars_already_scoped[var_name] = 1

    # Recur down all children to find more vars, unless the node
    # opens a new scope.
    if not is_module and not is_function and not is_assign:
        for child_node in ast.iter_child_nodes(node):
            new_vars, vars_already_scoped = _find_vars_in_same_scope(
                                            child_node, vars_already_scoped)
            current_scope_new_vars = current_scope_new_vars + new_vars

    return current_scope_new_vars, vars_already_scoped


def _move_initilizations_to_top_of_scope(node, vars_to_re_init):
    '''Adds an initilization (to None) to the begining of the body
    of a node for every variable in vars_to_re_init
    Parameters:
        node: ast node. Should be module or function node.
        vars_to_re_init: list of variables to reinitilize to None.
    Returns:
        None -- Performs update to node in place.'''
    for var in vars_to_re_init:
        node.body = [
            ast.Assign(targets=[
                ast.Name(id=var, ctx=ast.Store()),
            ], value=ast.NameConstant(None))
        ] + node.body

    return None
