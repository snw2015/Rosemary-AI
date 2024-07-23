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

    def _handle_import(self, child: RmlElement, namespace: Namespace):
        if 'path' not in child.attributes or not child.attributes['path']:
            raise RmlSyntaxException('Import must have a path', self.src_path)

        path = child.attributes['path']

        child_namespace = self._parse_file(path)

        if 'element' not in child.attributes:
            if 'as' not in child.attributes:
                for name, element in child_namespace.items():
                    namespace.append(name, element)
            else:
                if not child.attributes['as']:
                    raise RmlSyntaxException('Empty "as" attribute in <import>', self.src_path)

                namespace.append(child.attributes['as'], child_namespace)
        else:
            if not child.attributes['element']:
                raise RmlSyntaxException('Empty "element" attribute in <import>', self.src_path)

            element_names = list(map(str.strip, child.attributes['element'].split(',')))

            as_names = [None] * len(element_names)

            if 'as' in child.attributes:
                as_names = list(map(str.strip, child.attributes['as'].split(',')))
                if len(element_names) != len(as_names):
                    raise RmlSyntaxException(
                        f'Number of elements in "element" attribute ({len(element_names)}) '
                        f'does not match number of elements in "as" attribute ({len(as_names)})',
                        self.src_path
                    )

            for element_name, as_name in zip(element_names, as_names):
                try:
                    element = child_namespace[element_name]
                except KeyError:
                    raise RmlSyntaxException(
                        f'Element "{element_name}" not found in imported file "{path}"',
                        self.src_path
                    )

                if as_name is None:
                    namespace.append(element_name.split('.')[-1], element)
                else:
                    if not as_name:
                        raise RmlSyntaxException('Empty "as" attribute in <import>', self.src_path)

                    namespace.append(as_name, element)


    def _rml_tree_to_namespace(self, tree: RmlElement, parent_namespace: RosemaryNamespace = None) -> RosemaryNamespace:
        namespace = Namespace(parent_namespace)
        for child in tree.children:
            if child.is_text:
                continue
            elif child.indicator == ('import',):
                self._handle_import(child, namespace)
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
