import ast
import astor


class String_Transformer(ast.NodeTransformer):
    """Iterates through source code in src, finding ever location that opens
    a new scope. At each scope level, performs the transformations in
    transformations_to_perform. The transformations should be specified in the
    via a list of transformations from the enum.
        Parameters:
        transformations_to_perform: list of functions
        Returns:
        String of transformed source code
    """

    def __init__(self, transformations_to_perform):
        self._transforms = transformations_to_perform
        self._names_in_use = {}

    def transform(self, src):
        ast_src = ast.parse(src)
        new_ast = self.visit(ast_src)
        self._names_in_use = {}
        return astor.to_source(new_ast)

    def _visit_new_scope(self, node):
        for transform in self._transforms:
            node, _ = transform(node, {})
        for child in ast.iter_child_nodes(node):
            self.visit(child)
        return node

    def visit_FunctionDef(self, node):
        return self._visit_new_scope(node)

    def visit_Module(self, node):
        return self._visit_new_scope(node)
