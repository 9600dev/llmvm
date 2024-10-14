import ast
from typing import Any, List, Optional

from llmvm.common.helpers import Helpers


class Source:
    class Callsite:
        def __init__(self, class_name, method_name, line, col):
            self.class_name = class_name
            self.method_name = method_name
            self.line = line
            self.col = col

        def __repr__(self):
            return f"Callsite(class_name={self.class_name!r}, method_name={self.method_name!r}, line={self.line}, col={self.col})"

    class Symbol:
        def __init__(self, name, underlying_type, params, type, parent, docstring, line, col, returns):
            self.name = name
            self.underlying_type = underlying_type
            self.params = params
            self.type = type
            self.parent = parent
            self.docstring = docstring
            self.line = line
            self.col = col
            self.returns = returns

        def __repr__(self):
            params_repr = ', '.join([f"{name}: {ptype}" for name, ptype in self.params]) if self.params else 'None'
            return (f"Symbol(name={self.name!r}, underlying_type={self.underlying_type!r}, "
                    f"params={params_repr}, type={self.type!r}, parent={self.parent!r}, "
                    f"docstring={self.docstring!r}, line={self.line}, col={self.col})")

        def __get_type_name(self, type_node):
            if isinstance(type_node, ast.Name):
                return type_node.id
            elif isinstance(type_node, ast.Attribute):
                return f"{self.__get_type_name(type_node.value)}.{type_node.attr}"
            elif isinstance(type_node, ast.Subscript):
                return f"{self.__get_type_name(type_node.value)}[{self.__get_type_name(type_node.slice)}]"
            elif isinstance(type_node, ast.Constant):
                return str(type_node.value)
            elif isinstance(type_node, ast.Tuple):
                return f"Tuple[{', '.join(self.__get_type_name(elt) for elt in type_node.elts)}]"
            else:
                return "Any"

        def method_definition(self):
            return f"def {self.name}({', '.join([f'{name}: {ptype}' for name, ptype in self.params])}) -> {self.__get_type_name(self.returns)}:"

        def class_definition(self):
            if not self.parent:
                return f"class {self.name}:"
            else:
                return f"class {self.name}({self.parent}):"

    def __init__(self, file_path):
        self.file_path = file_path
        self.source_code, self.tree = self._parse_file()

    def _parse_file(self):
        with open(self.file_path, 'r') as file:
            source_code = file.read()
            try:
                ast_tree = ast.parse(Helpers.escape_newlines_in_strings(source_code))
            except Exception as ex:
                return source_code, None
            return source_code, ast_tree

    def get_tree(self):
        return self.tree

    def get_methods(self, class_name: Optional[str]) -> List[Symbol]:
        if not self.tree:
            return []

        methods = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_name = item.name
                        params = []
                        for arg in item.args.args:
                            # Check if there is a type annotation
                            param_type = ast.get_source_segment(self.source_code, arg.annotation) if arg.annotation else None
                            params.append((arg.arg, param_type))
                        return_type = item.returns if item.returns else Any
                        symbol = Source.Symbol(
                            name=method_name,
                            underlying_type=class_name,
                            params=params,
                            type='method',
                            parent=class_name,
                            docstring=ast.get_docstring(item),
                            line=item.lineno,
                            col=item.col_offset,
                            returns=return_type
                        )
                        methods.append(symbol)
        return methods

    def get_method_source(self, method_name) -> str:
        if not self.tree:
            return ''

        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == method_name:
                return ast.get_source_segment(self.source_code, node)  # type: ignore
        raise ValueError(f"Method {method_name} not found in source code")

    def get_classes(self) -> List[Symbol]:
        if not self.tree:
            return []

        class_symbols = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                class_docstring = ast.get_docstring(node)
                symbol = Source.Symbol(
                    name=node.name,
                    underlying_type=None,  # Class definitions don't have an underlying type
                    params=[],  # Class definitions don't have parameters
                    type='class',
                    parent=None,  # Class definitions don't have a parent
                    docstring=class_docstring,
                    line=node.lineno,
                    col=node.col_offset,
                    returns=None,
                )
                class_symbols.append(symbol)
        return class_symbols

    @staticmethod
    def get_references(tree, method_name) -> List['Source.Callsite']:
        references = []
        current_class = None
        current_method = None

        class Visitor(ast.NodeVisitor):
            def visit_ClassDef(self, node):
                nonlocal current_class
                prev_class = current_class
                current_class = node.name
                self.generic_visit(node)
                current_class = prev_class

            def visit_FunctionDef(self, node):
                nonlocal current_method
                prev_method = current_method
                current_method = node.name
                self.generic_visit(node)
                current_method = prev_method

            def visit_Call(self, node):
                if (
                    (hasattr(node.func, 'id') and node.func.id == method_name)  # type: ignore
                    or (hasattr(node.func, 'attr') and node.func.attr == method_name)  # type: ignore
                ):
                    if current_class or current_method:
                        callsite = Source.Callsite(current_class, current_method, node.lineno, node.col_offset)
                        references.append(callsite)
                self.generic_visit(node)

        Visitor().visit(tree)
        return references
