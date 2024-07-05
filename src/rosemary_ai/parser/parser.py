from pathlib import Path

from lark import Lark

from .._utils._file_utils import read_and_close_file_to_root, read_and_close_file # noqa
from .namespace import Namespace
from .leaf_elements import rml_to_petal, rml_to_template, RosemaryNamespace
from .transformer import RmlElement, TreeToRmlTreeTransformer

GRAMMAR_PATH = "src/rosemary_ai/parser/rosemary.lark"


class RosemaryParser:
    def __init__(self, src_path: str):
        grammar = read_and_close_file_to_root(GRAMMAR_PATH)
        self.parser = Lark(grammar, start='rosemary')
        self.transformer = TreeToRmlTreeTransformer()

        self.imported_namespaces = {}
        self.path_stack = [Path(src_path).resolve()]

        rml_tree = self._src_to_rml_tree(read_and_close_file(src_path))
        self.namespace = self._rml_tree_to_namespace(rml_tree)

    def _rml_tree_to_namespace(self, tree: RmlElement, parent_namespace: RosemaryNamespace = None) -> RosemaryNamespace:
        namespace = Namespace(parent_namespace)
        for child in tree.children:
            if child.is_text:
                continue
            elif child.indicator == ('import',):
                if 'path' not in child.attributes or not child.attributes['path']:
                    raise ValueError('Import must have a path')
                for name, element in self._parse_file(child.attributes['path']).items():
                    namespace.append(name, element)
            elif child.indicator == ('corolla',):
                if 'name' not in child.attributes or not child.attributes['name']:
                    raise ValueError('Corolla must have a name')
                namespace.append(child.attributes['name'], self._rml_tree_to_namespace(child, namespace))
            elif child.indicator == ('petal',):
                if 'name' not in child.attributes or not child.attributes['name']:
                    raise ValueError('Petal must have a name')
                namespace.append(child.attributes['name'], rml_to_petal(child, namespace))
            elif child.indicator == ('template',):
                if 'name' not in child.attributes or not child.attributes['name']:
                    raise ValueError('Template must have a name')
                namespace.append(child.attributes['name'], rml_to_template(child, namespace))
            else:
                raise ValueError(f'Unknown element {child.indicator}')

        return namespace

    def _parse_file(self, path_str: str) -> RosemaryNamespace:
        assert self.path_stack
        path = (self.path_stack[-1].parent / Path(path_str)).resolve()

        if path in self.imported_namespaces:
            return self.imported_namespaces[path]

        self.path_stack += [path]
        rml_tree = self._src_to_rml_tree(read_and_close_file(path))
        namespace = self._rml_tree_to_namespace(rml_tree)
        self.imported_namespaces[path] = namespace
        self.path_stack.pop()
        return namespace

    def _src_to_rml_tree(self, src: str) -> RmlElement:
        return self.transformer.transform(self.parser.parse(src))
