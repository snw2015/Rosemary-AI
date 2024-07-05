from copy import copy
from typing import List

from ..multi_modal.image import Image
from .data_expression import DataExpression
from .executor import Executor
from .leaf_elements import Environment, RosemaryTemplate
from .transformer import RmlElement, TextToken


def _eval(repr_, context, need_copy=True):
    return DataExpression(repr_).evaluate(context, need_copy)


def traverse_all(env_stack: List[Environment], children: List[RmlElement], executor: Executor) -> bool:
    for child in children:
        if not traverse(env_stack, child, executor):
            return False
    return True


def traverse(env_stack: List[Environment], element: RmlElement, executor: Executor) -> bool:
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
            key = None
            if 'key' in element.attributes:
                key = element.attributes['key']
            if 'key_eval' in element.attributes:
                key = _eval(element.attributes['key_eval'], curr_env.context)

            executor.begin_scope('dict_item', key)
            succeed = traverse_all(env_stack, element.children, executor)
            executor.end_scope('dict_item', succeed)

            return succeed
        elif element.indicator == ('p',):
            succeed = traverse_all(env_stack, element.children, executor)
            executor.execute('\n', curr_env.context)
            return succeed
        elif element.indicator == ('img',):
            if 'src' not in element.attributes and 'src_eval' not in element.attributes:
                raise ValueError('Image must have a source')

            src = None
            if 'src' in element.attributes:
                src = element.attributes['src']
            if 'src_eval' in element.attributes:
                src = _eval(element.attributes['src_eval'], curr_env.context)

            executor.execute(Image(src), curr_env.context)
            return True
        elif element.indicator == ('if',):
            if 'cond' not in element.attributes:
                raise ValueError('If must have a condition')
            if _eval(element.attributes['cond'], curr_env.context):
                return traverse_all(env_stack, element.children, executor)
        elif element.indicator == ('for',):
            if 'range' not in element.attributes:
                raise ValueError('For must have a range')
            loop_range: range = _eval('range(' + element.attributes['range'] + ')', curr_env.context)
            assert isinstance(loop_range, range)

            end_if_failed = False
            if 'try' in element.attributes:
                end_if_failed = _eval(element.attributes['try'], curr_env.context)

            var_name = None
            if 'var' in element.attributes:
                var_name = element.attributes['var']

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
                        executor.set_snapshot(snapshot)
                        break
                    else:
                        return False

            if var_name:
                env_stack.pop()

            return True

        elif element.indicator == ('foreach',):
            if 'in' not in element.attributes:
                raise ValueError('For-each must have a target')
            loop_list = _eval(element.attributes['in'], curr_env.context)

            end_if_failed = False
            if 'try' in element.attributes:
                end_if_failed = _eval(element.attributes['try'], curr_env.context)

            var_name = None
            if 'var' in element.attributes:
                var_name = element.attributes['var']

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
                        executor.set_snapshot(snapshot)
                        break
                    else:
                        return False
            if var_name:
                env_stack.pop()

            return True
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
                executor.set_snapshot(snapshot)
            else:  # choose first successful branch
                for child in element.children:
                    if child.is_text:
                        continue
                    snapshot = executor.get_snapshot()
                    assert child.indicator == ('or',)
                    succeed = traverse_all(env_stack, child.children, executor)
                    if succeed:
                        return True
                    executor.set_snapshot(snapshot)

            required = ('required' in element.attributes and
                        _eval(element.attributes['required'], curr_env.context))
            return not required
        else:
            # slots
            if len(element.indicator) == 1 and curr_env.slots:
                indicator = element.indicator[0]
                is_slot = indicator in curr_env.slots
                if is_slot:
                    assert not element.children

                    slot_stack = curr_env.slots[indicator]
                    if not slot_stack:
                        return False
                    slot = curr_env.slots[element.indicator[0]].pop()

                    env_stack.pop()
                    succeed = traverse_all(env_stack, slot.children, executor)
                    env_stack.append(curr_env)

                    return succeed

            # templates
            indicator = element.indicator
            template: RosemaryTemplate = curr_env.namespace.get_by_indicator(indicator)
            assert isinstance(template, RosemaryTemplate)

            new_namespace = template.namespace
            context = {name: None for name in template.variable_names}
            for var_name in template.variable_names:
                if var_name in element.attributes:
                    context[var_name] = _eval(element.attributes[var_name], curr_env.context)

            slot_found = False
            new_slots = {name: [] for name in template.slot_names}
            for child in element.children:
                if len(child.indicator) == 1 and child.indicator[0] in new_slots:
                    new_slots[child.indicator[0]] += [child]
                    slot_found = True

            # make this one the default slot
            if not slot_found and len(new_slots) == 1:
                slot_name = list(new_slots.keys())[0]
                new_slots[slot_name] = [element]

            for slot_lists in new_slots.values():
                slot_lists.reverse()  # make it a stack

            new_env = Environment(context, new_slots, new_namespace)

            env_stack.append(new_env)
            succeed = traverse_all(env_stack, template.element.children, executor)
            env_stack.pop()

            return succeed

    return True
