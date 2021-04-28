import ast


def _new_name(current_name, names_already_used):
    """Changes a base name into a unused name for a variable.
    Parameters:
        current_name: the base or current name of the variable.
        names_already_used: hashtable of names in use.
    Returns:
        name: string of new name.
    """
    incrementor = 0
    name = current_name + str(incrementor)
    while name in names_already_used:
        # remove old postfix
        name = name[:-len(str(incrementor))]
        # increment and add new prefix
        incrementor += 1
        name = name + str(incrementor)
    return name


def _get_all_used_variable_names(ast_node):
    """Returns a dictionry (for fast lookups) of all the names
    (variables, functions, modules, classes, etc) in use in the abstract syntax
    tree.
    Parameters:
        ast_node: Root node of the ast to get vars from
    Returns:
        dictionary of all variable names in use.
    """
    vars_used = []

    for node in ast.walk(ast_node):
        if isinstance(node, ast.Name):
            vars_used.append(node.id)
        elif isinstance(node, ast.FunctionDef):
            vars_used.append(node.name)
        elif isinstance(node, ast.Import):
            for name in node.names:
                vars_used.append(name.name)

    return dict.fromkeys(vars_used)
