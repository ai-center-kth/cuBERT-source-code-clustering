import ast

from .name_utils import _new_name


def _make_for_loops_while(parent_node, names_in_use):
    """Converts for loops into while loops.
    Creates an index variable and a call to the len() function as a test
    condition for the while loop.
    All for loop iterators must be indexable.
    DOES NOT SUPPOT NONINDEXABLE ITERATORS.
    Parameters:
        parent_node: ast node
    Returns:
        parent node with updates"""

    # Get every index of for loop objects in the body of parent node.
    # Could be done cleaner with a numpy .where, but we'd have to import numpy
    # pretty much just for that.
    try:
        indxs_for_loops = []
        for indx in range(len(parent_node.body)):
            current = parent_node.body[indx]
            if isinstance(current, ast.For):
                indxs_for_loops.append(indx)
            if hasattr(current, "body"):
                is_module = isinstance(current, ast.Module)
                is_func_def = isinstance(current, ast.FunctionDef)
                if not is_func_def and not is_module:
                    current, names_in_use = _make_for_loops_while(current,
                                                                  names_in_use)

    except AttributeError:
        # node has no body. No for loops in it.
        return parent_node, names_in_use

    num_lines_inserted = 0

    for for_loop_index in indxs_for_loops:
        for_loop_index = for_loop_index + num_lines_inserted
        for_loop = parent_node.body[for_loop_index]

        # Make loop incrementor variable.
        name_incrementor_variable = _new_name('loop_index', names_in_use)
        names_in_use[name_incrementor_variable] = 1

        # Make a call to built in len() function with the iterator
        # provided in the for loop.
        len_builtin_function = ast.Name(id='len', ctx=ast.Load)
        len_function_call = ast.Call(func=len_builtin_function,
                                     args=[for_loop.iter],
                                     keywords=[])

        # Test for while loop.
        left = ast.Name(id=name_incrementor_variable, ctx=ast.Load)
        compare_op = ast.Compare(left=left,
                                 ops=[ast.Lt()],
                                 comparators=[len_function_call])

        # Assign current value of loop to for loop target.
        index = ast.Index(ast.Name(id=name_incrementor_variable, ctx=ast.Load))
        value = ast.Subscript(for_loop.iter, index)
        target = [for_loop.target]
        assign_to_for_loop_target = ast.Assign(target, value)

        # Increment index variable.
        name = ast.Name(id=name_incrementor_variable)
        add_1_to_index_variable = ast.AugAssign(name, ast.Add(), ast.Num(1))

        # Construct while loop.
        while_loop = [assign_to_for_loop_target] + \
            for_loop.body + [add_1_to_index_variable]
        while_loop = ast.While(test=compare_op, body=while_loop, orelse=[])

        # Replace for with while loop.
        parent_node.body[for_loop_index] = while_loop

        # Insert loop incrementor variable before while loop and set to 0.
        inc_name = ast.Name(id=name_incrementor_variable, ctx=ast.Store)
        inc_0 = ast.Assign([inc_name], ast.Num(0))
        parent_node.body.insert(for_loop_index, inc_0)

        # Not the total lines inserted, only the lines inserted into the
        # parent's body. (so not the lines inside the loop)
        num_lines_inserted += 1

    return parent_node, names_in_use
