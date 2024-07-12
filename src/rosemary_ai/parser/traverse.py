from copy import copy
from typing import List, Iterable

from ..exceptions import RmlFormatException
from ..multi_modal.image import Image
from .data_expression import DataExpression
from .executor import Executor
from .leaf_elements import Environment, RosemaryTemplate, Slot
from .transformer import RmlElement, TextToken


def _eval(repr_, context, need_copy=True):
    return DataExpression(repr_).evaluate(context, need_copy)


def traverse_all(env_stack: List[Environment], children: List[RmlElement], executor: Executor) -> bool:
    for child in children:
        if not traverse(env_stack, child, executor):
            return False
    return True


def _get_range_from_str(range_str: str, env: Environment):
    try:
        loop_range: range = _eval('range(' + range_str + ')', env.context)
        assert isinstance(loop_range, range)
        return loop_range
    except Exception:
        raise RmlFormatException(f'"range" is not a valid range expression: {range_str}.')


def _find_and_add_slot(element: RmlElement, new_slots: dict[str, Slot],
                       env: Environment):
    if len(element.indicator) != 1:
        raise RmlFormatException(f'Unexpected tag when generating slots: {".".join(element.indicator)}')

    indicator = element.indicator[0]
    if indicator == 'if':
        if 'cond' not in element.attributes:
            raise RmlFormatException('If must have a condition, given by "cond" attribute.')

        if env.eval(element.attributes['cond']):
            for child in element.children:
                _find_and_add_slot(child, new_slots, env)
    elif indicator == 'for':
        if 'range' in element.attributes:
            range_str = element.attributes['range']
            loop_range = _get_range_from_str(range_str, env)

            var_name = None
            has_var = 'var' in element.attributes
            if has_var:
                var_name = element.attributes['var']
                if not var_name:
                    raise RmlFormatException('Loop variable name must not be empty.')

            for i in loop_range:
                new_env = copy(env)
                if has_var:
                    new_env.context[var_name] = i
                for child in element.children:
                    _find_and_add_slot(child, new_slots, new_env)

        elif 'in' in element.attributes:
            loop_list = env.eval(element.attributes['in'])
            if not isinstance(loop_list, Iterable):
                raise RmlFormatException('"in" must be an iterable object.')

            var_name = None
            has_var = 'var' in element.attributes
            if has_var:
                var_name = element.attributes['var']
                if not var_name:
                    raise RmlFormatException('Loop variable name must not be empty.')

            for obj in loop_list:
                new_env = copy(env)
                if has_var:
                    new_env.context[var_name] = obj
                for child in element.children:
                    _find_and_add_slot(child, new_slots, new_env)
        else:
            raise RmlFormatException(
                'For must have a range or an iterable object,'
                ' given by "range" or "in" attribute.'
            )

    elif indicator in new_slots:
        slot = new_slots[indicator]
        var_context = {}
        for var_name in slot.variable_names:
            if var_name in element.attributes:
                var_context[var_name] = env.eval(element.attributes[var_name])
            else:
                raise RmlFormatException(f'Slot {indicator} lacks variable {var_name}.')
        slot.append(element, env, var_context)

    else:
        raise RmlFormatException(f'Slot not found: {indicator}')


def traverse(env_stack: List[Environment], element: RmlElement, executor: Executor) -> bool:
    assert env_stack
    curr_env = env_stack[-1]

    if element.is_text:
        for token in element.text_tokens:
            is_plain_text = token.type == TextToken.TYPE.PLAIN_TEXT
            if is_plain_text:
                succeed = executor.execute(token.text, curr_env.context)
            else:
                succeed = executor.execute(DataExpression(token.text), curr_env.context)

            if not succeed:
                return False
        return True
    else:
        if element.indicator == ('list',):
            executor.begin_scope('list')
            succeed = traverse_all(env_stack, element.children, executor)
            executor.end_scope('list')

            return succeed
        elif element.indicator == ('dict',):
            executor.begin_scope('dict')
            succeed = traverse_all(env_stack, element.children, executor)
            executor.end_scope('dict')

            return succeed
        elif element.indicator == ('list-item',):
            executor.begin_scope('list_item')
            succeed = traverse_all(env_stack, element.children, executor)
            executor.end_scope('list_item', succeed)

            return succeed
        elif element.indicator == ('dict-item',):
            if 'key' not in element.attributes and 'key_eval' not in element.attributes:
                raise RmlFormatException('Dict item must have a key, given by "key" or "key_eval" attribute.')

            key = None
            if 'key' in element.attributes:
                key = element.attributes['key']
            if 'key_eval' in element.attributes:
                key = _eval(element.attributes['key_eval'], curr_env.context)

            executor.begin_scope('dict_item', key)
            succeed = traverse_all(env_stack, element.children, executor)
            executor.end_scope('dict_item', succeed)

            return succeed
        elif element.indicator == ('br',):
            if element.children:
                raise RmlFormatException('br element cannot have children.')
            executor.execute('\n', curr_env.context)

            return True
        elif element.indicator == ('img',):
            if 'src' not in element.attributes and 'src_eval' not in element.attributes:
                raise RmlFormatException('Image must have a source, given by "src" or "src_eval" attribute.')

            src = None
            if 'src' in element.attributes:
                src = element.attributes['src']
            if 'src_eval' in element.attributes:
                src = _eval(element.attributes['src_eval'], curr_env.context)

            executor.execute(Image(src), curr_env.context)
            return True
        elif element.indicator == ('if',):
            if 'cond' not in element.attributes:
                raise RmlFormatException('If must have a condition, given by "cond" attribute.')
            if _eval(element.attributes['cond'], curr_env.context):
                return traverse_all(env_stack, element.children, executor)
        elif element.indicator == ('for',):
            if 'slot' in element.attributes:  # only allowed in templates
                slot_name = element.attributes['slot']
                slot = curr_env.slots[slot_name]

                if slot.is_inf:
                    raise RmlFormatException('Infinite slot is not allowed in for expansion.')

                while slot.has_next():
                    new_env = copy(curr_env)
                    slot_info = slot.pop()

                    new_env.slots[slot_name] = Slot([slot_info],
                                                    slot.variable_names)

                    var_context = slot_info[2]
                    new_env.context.update(var_context)
                    env_stack.append(new_env)
                    succeed = traverse_all(env_stack, element.children, executor)
                    env_stack.pop()

                    if not succeed:
                        return False
                return True

            elif 'range' in element.attributes:
                range_str = element.attributes['range']
                loop_range = _get_range_from_str(range_str, curr_env)

                end_if_failed = False
                if 'try' in element.attributes:
                    end_if_failed = DataExpression(element.attributes['try']).evaluate(curr_env.context)

                var_name = None
                if 'var' in element.attributes:
                    var_name = element.attributes['var']
                    if not var_name:
                        raise RmlFormatException('Loop variable name must not be empty.')

                if var_name:  # loop variable only exists in the loop
                    env_stack += [copy(curr_env)]

                loop_env = env_stack[-1]
                for i in loop_range:
                    snapshot = executor.get_snapshot()
                    if var_name:
                        loop_env.context[var_name] = i
                    succeed = traverse_all(env_stack, element.children, executor)
                    if not succeed:
                        if end_if_failed:
                            executor.back_to_snapshot(snapshot)
                            break
                        else:
                            return False

                if var_name:
                    env_stack.pop()

                return True

            elif 'in' in element.attributes:
                loop_list = DataExpression(element.attributes['in']).evaluate(curr_env.context)
                if not isinstance(loop_list, Iterable):
                    raise RmlFormatException('Loop list must be iterable.')

                end_if_failed = False
                if 'try' in element.attributes:
                    end_if_failed = DataExpression(element.attributes['try']).evaluate(curr_env.context)

                var_name = None
                if 'var' in element.attributes:
                    var_name = element.attributes['var']
                    if not var_name:
                        raise RmlFormatException('Loop variable name must not be empty.')

                if var_name:  # loop variable only exists in the loop
                    env_stack += [copy(curr_env)]

                loop_env = env_stack[-1]
                for obj in loop_list:
                    snapshot = executor.get_snapshot()
                    if var_name:
                        loop_env.context[var_name] = obj
                    succeed = traverse_all(env_stack, element.children, executor)
                    if not succeed:
                        if end_if_failed:
                            executor.back_to_snapshot(snapshot)
                            break
                        else:
                            return False
                if var_name:
                    env_stack.pop()

                return True

            else:
                raise RmlFormatException(
                    'For must have a slot, a range or an iterable object,'
                    ' given by "slot", "range" or "in" attribute.'
                )

        elif element.indicator == ('optional',):
            has_or = False
            for child in element.children:
                if child.indicator == ('or',):
                    has_or = True
                    break
            if not has_or:  # consider the whole element as optional
                snapshot = executor.get_snapshot()
                succeed = traverse_all(env_stack, element.children, executor)
                if succeed:
                    return True
                executor.back_to_snapshot(snapshot)
            else:  # choose first successful branch
                for child in element.children:
                    if child.indicator != ('or',):
                        raise RmlFormatException('Only "or" elements are allowed in an "optional" element.')

                    snapshot = executor.get_snapshot()

                    succeed = traverse_all(env_stack, child.children, executor)
                    if succeed:
                        return True
                    executor.back_to_snapshot(snapshot)

            required = ('required' in element.attributes and
                        DataExpression(element.attributes['required']).evaluate(curr_env.context))
            return not required
        else:
            # slots
            if len(element.indicator) == 1 and curr_env.slots:
                indicator = element.indicator[0]
                if indicator in curr_env.slots:
                    if element.children:
                        raise RmlFormatException('Slot element used in a template cannot have children.')

                    slot = curr_env.slots[indicator]
                    if not slot.has_next():
                        raise RmlFormatException(f'Elements found for slot "{indicator}" is not enough.')

                    slot_element, slot_env, _ = curr_env.slots[indicator].pop()

                    env_stack.append(slot_env)
                    succeed = traverse_all(env_stack, slot_element.children, executor)
                    env_stack.pop()

                    return succeed

            # templates
            indicator = element.indicator
            try:
                template: RosemaryTemplate = curr_env.namespace.get_by_indicator(indicator)
            except Exception:
                raise RmlFormatException(f'Unknown tag: {indicator}.')

            if not isinstance(template, RosemaryTemplate):
                raise RmlFormatException(
                    f'The given tag name cannot be interpreted as a template or slot: {indicator}.'
                )

            new_namespace = template.namespace
            context = {name: None for name in template.variable_names}
            for var_name in template.variable_names:
                if var_name in element.attributes:
                    context[var_name] = _eval(element.attributes[var_name], curr_env.context)

            slot_vars = template.slot_vars
            new_slots = {}

            if len(slot_vars) == 1 and list(slot_vars.keys())[0].startswith('@'):
                slot_name = list(slot_vars.keys())[0][1:]
                slot_var = []
                is_inf = True
                new_slots[slot_name] = Slot([(element, curr_env, {})], slot_var, is_inf)
            else:
                for slot_name in slot_vars.keys():
                    is_inf = False
                    slot_var = slot_vars[slot_name]
                    if slot_name.startswith('*'):
                        is_inf = True
                        slot_name = slot_name[1:]
                    new_slots[slot_name] = Slot([], slot_var, is_inf)
                for child in element.children:
                    _find_and_add_slot(child, new_slots, curr_env)

            for slot_element in new_slots.values():
                slot_element.reverse()

            new_env = Environment(context, new_slots, new_namespace)

            env_stack.append(new_env)
            succeed = traverse_all(env_stack, template.element.children, executor)
            env_stack.pop()

            return succeed

    return True
