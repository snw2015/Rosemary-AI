from pathlib import Path

from lark import Lark

from .._utils.file_utils import read_and_close_file_to_root, read_and_close_file, _get_proj_root  # noqa
from .namespace import Namespace
from .environment import rml_to_petal, rml_to_template, RosemaryNamespace
from .transformer import RmlElement, TreeToRmlTreeTransformer
from ..exceptions import RmlSyntaxException

GRAMMAR_PATH = "parser/rosemary.lark"
RML_COMMON_PATH = "rml_common/common.rml"


class RosemaryParser:
    def __init__(self, src_path: str):
        grammar = read_and_close_file_to_root(GRAMMAR_PATH)
        self.parser = Lark(grammar, start='rosemary')
        self.transformer = TreeToRmlTreeTransformer()

        self.imported_namespaces = {}
        self.src_path = src_path

        if src_path == 'common':
            rml_tree = self._src_to_rml_tree(read_and_close_file_to_root(RML_COMMON_PATH))
            self.path_stack = [_get_proj_root() / RML_COMMON_PATH]
        else:
            rml_tree = self._src_to_rml_tree(read_and_close_file(src_path))
            self.path_stack = [Path(src_path).resolve()]

        self.namespace = self._rml_tree_to_namespace(rml_tree)

    def _rml_tree_to_namespace(self, tree: RmlElement, parent_namespace: RosemaryNamespace = None) -> RosemaryNamespace:
        namespace = Namespace(parent_namespace)
        for child in tree.children:
            if child.is_text:
                continue
            elif child.indicator == ('import',):
                if 'path' not in child.attributes or not child.attributes['path']:
                    raise RmlSyntaxException('Import must have a path', self.src_path)
                for name, element in self._parse_file(child.attributes['path']).items():
                    namespace.append(name, element)
            elif child.indicator == ('corolla',):
                if 'name' not in child.attributes or not child.attributes['name']:
                    raise RmlSyntaxException('Corolla must have a name', self.src_path)
                namespace.append(child.attributes['name'], self._rml_tree_to_namespace(child, namespace))
            elif child.indicator == ('petal',):
                if 'name' not in child.attributes or not child.attributes['name']:
                    raise RmlSyntaxException('Petal must have a name', self.src_path)

                try:
                    namespace.append(child.attributes['name'], rml_to_petal(child, namespace, self.src_path))
                except Exception as e:
                    raise RmlSyntaxException('Failed to parse petal', self.src_path) from e
            elif child.indicator == ('template',):
                if 'name' not in child.attributes or not child.attributes['name']:
                    raise RmlSyntaxException('Template must have a name', self.src_path)
                try:
                    namespace.append(child.attributes['name'], rml_to_template(child, namespace, self.src_path))
                except Exception as e:
                    raise RmlSyntaxException('Failed to parse template', self.src_path) from e
            else:
                raise RmlSyntaxException(f'Unknown element {child.indicator}', self.src_path)

        return namespace

    def _parse_file(self, path_str: str) -> RosemaryNamespace:
        if path_str == 'common':
            return _COMMON_NAMESPACE

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
        try:
            tree = self.transformer.transform(self.parser.parse(src))
        except Exception as e:
            raise RmlSyntaxException('Failed to parse code', self.src_path) from e

        return tree


_COMMON_NAMESPACE = RosemaryParser('common').namespace
