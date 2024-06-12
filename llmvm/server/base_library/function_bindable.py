import ast
import re
import time
from typing import Any, Callable, Dict, Generator, List, Optional, cast

import astunparse

from llmvm.common.helpers import Helpers
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import (Assistant, Content, FunctionCall, LLMCall,
                                  Message, System, User)
from llmvm.server.ast_parser import Parser
from llmvm.server.starlark_execution_controller import ExecutionController
from llmvm.server.starlark_runtime import StarlarkRuntime

logging = setup_logging()


class FunctionBindable():
    def __init__(
        self,
        expr,
        func: str,
        agents: List[Callable],
        messages: List[Message],
        lineno: int,
        expr_instantiation,
        scope_dict: Dict[Any, Any],
        original_code: str,
        original_query: str,
        controller: ExecutionController,
        starlark_runtime: StarlarkRuntime,
    ):
        self.expr = expr
        self.expr_instantiation = expr_instantiation
        self.messages: List[Message] = messages
        self.func = func.replace('"', '')
        self.agents = agents
        self.lineno = lineno
        self.scope_dict = scope_dict
        self.original_code = original_code
        self.original_query = original_query
        self.controller = controller
        self.bound_function: Optional[Callable] = None
        self.starlark_runtime = starlark_runtime
        self._result = None

    def __call__(self, *args, **kwargs):
        if self._result:
            return self._result

    def __bind_helper(
        self,
        func: str,
    ) -> Message:
        # if we have a list, we need to use a different prompt
        if isinstance(self.expr, list):
            raise ValueError('llm_bind() does not support lists. You should rewrite the code to use a for loop instead.')

        # get a function definition fuzzy binding
        function_str = Helpers.in_between(func, '', '(')
        function_callable = [f for f in self.agents if function_str in str(f)]
        if not function_callable:
            raise ValueError('could not find function: {}'.format(function_str))

        function_callable = function_callable[0]
        function_definition = Helpers.get_function_description_flat(cast(Callable, function_callable))

        message = Helpers.prompt_message(
            prompt_name='llm_bind_global.prompt',
            template={
                'function_definition': function_definition,
            },
            user_token=self.controller.get_executor().user_token(),
            assistant_token=self.controller.get_executor().assistant_token(),
            append_token=self.controller.get_executor().append_token(),
        )
        return message

    def binder(
        self,
        expr,
        func: str,
    ) -> Generator['FunctionBindable', None, None]:
        bound = False
        global_counter = 0
        messages: List[Message] = []
        bindable = ''
        function_call: Optional[FunctionCall] = None

        def find_string_instantiation(target_string, source_code):
            parsed_ast = ast.parse(source_code)

            for node in ast.walk(parsed_ast):
                # Check for direct assignment
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                            if target_string in node.value.s:
                                return (node, None)
                        # Check for string instantiation in a list
                        elif isinstance(node.value, ast.List):
                            for element in node.value.elts:
                                if isinstance(element, ast.Constant) and target_string in element.s:
                                    return (element, node)
            return None, None

        # the following code determines the progressive scope exposure to the LLM
        # to help it determine the binding
        expr_instantiation_message = User(Content())
        if isinstance(expr, str) and find_string_instantiation(expr, self.original_code):
            node, parent = find_string_instantiation(expr, self.original_code)
            if parent:
                expr_instantiation_message.message = Content(
                    f"The data in the next message was instantiated via this Starlark code: {astunparse.unparse(parent)}"
                )
            elif node:
                expr_instantiation_message.message = Content(
                    f"The data in the next message was instantiated via this Starlark code: {astunparse.unparse(node.value)}"
                )

        messages.append(System(Content(
            '''You are a Starlark compiler and code generator. You generate parsable Starlark code.'''
        )))

        # get the binder prompt message
        messages.append(self.__bind_helper(
            func=func,
        ))
        # start with just the expression binding
        messages.extend(self.controller.statement_to_message(expr))
        # instantiation
        if str(expr_instantiation_message.message):
            messages.append(expr_instantiation_message)
        # goal
        messages.append(User(Content(
            f"""The overall goal of the Starlark program is to: {self.original_query}."""
        )))
        messages.append(User(Content(
            f"""The Starlark code that is currently being executed is: {self.original_code}"""
        )))

        # program scope
        def expand_str(value):
            if isinstance(value, str):
                return value
            if hasattr(value, 'get_str'):
                return value.get_str()
            return str(value)

        scope = '\n'.join(['{} = "{}"'.format(key, expand_str(value)) for key, value in self.scope_dict.items()])
        messages.append(User(Content(
            f"""The Starlark program's running global scope for all variables is:

            {scope}

            You might find data you need to bind function arguments in the values of these variables.
            """
        )))

        counter = 5  # expr, overall goal, starlark code being executed
        # counter = 6  # program scope, including all variables
        assistant_counter = 0

        while global_counter < 2:
            # try and bind the callsite without executing
            while not bound and counter < 8:

                llm_bind_result = self.controller.execute_llm_call(
                    llm_call=LLMCall(
                        user_message=User(Content()),  # we can pass an empty message here and the context_messages contain everything  # noqa:E501
                        context_messages=messages[:counter + assistant_counter][::-1],  # reversing the list using list slicing
                        executor=self.controller.get_executor(),
                        model=self.controller.get_executor().get_default_model(),
                        temperature=0.0,
                        max_prompt_len=self.controller.get_executor().max_input_tokens(),
                        completion_tokens_len=self.controller.get_executor().max_output_tokens(),
                        prompt_name=''
                    ),
                    query=self.original_query,
                    original_query=self.original_query,
                )

                bindable = str(llm_bind_result.message)

                # the LLM can get confused and generate a function definition instead of a callsite
                # or enclose the result in ```python ... ``` code blocks.
                if 'def ' in bindable:
                    bindable = bindable.replace('def ', '')

                if '```python' in bindable:
                    match = re.search(r'```python([\s\S]*?)```', bindable)
                    if match:
                        bindable = match.group(1).replace('python', '').strip()

                if '```starlark' in bindable:
                    match = re.search(r'```starlark([\s\S]*?)```', bindable)
                    if match:
                        bindable = match.group(1).replace('starlark', '').strip()

                # get function definition
                parser = Parser()
                parser.agents = self.agents
                function_call = parser.get_callsite(bindable)

                if 'None' in str(bindable):
                    # move forward a stage and add the latest assistant response
                    # as the assistant response will have a # based question in it
                    # which will help bind the unbindable arguments.
                    counter += 1
                    assistant_counter += 1

                    if '#' in bindable:
                        question = bindable.split('#')[1].strip()
                        prompt = f'''
                        Using the data found in previous messages, answer the question "{question}", and then bind the callsite
                        using the same reply rules as in previous messages. Reply with only Starlark code.
                        '''
                        messages.insert(0, User(Content(prompt)))
                    else:
                        # todo figure this out
                        messages.insert(0, Assistant(message=Content(bindable)))
                    if counter > len(messages) - assistant_counter:
                        # we've run out of messages, so we'll just use the original code
                        break

                elif 'None' not in str(bindable) and function_call:
                    break
                else:
                    # no function_call result, so bump the counter
                    messages.insert(0, Assistant(message=Content(bindable)))
                    messages.insert(0, User(message=Content(
                        """Please try harder to bind the callsite.
                        Look thoroughly through the previous messages for data and then reply with your best guess at the bounded
                        callsite. Reply only with Starlark code that can be parsed by the Starlark compiler.
                        Do not apologize. Do not explain yourself. If you have previously replied with natural language,
                        it's likely I could not compile it. Please reply with only Starlark code.
                        """
                    )))
                    assistant_counter += 1
                    counter += 1

            # Using the previous messages, What is the company name associated with Steve Baxter?
            # If you've answered the question above, can you rebind the callsite?
            # def search_linkedin_profile(first_name: str, last_name: str, company_name: str) -> str
            # # Searches for the LinkedIn profile of a given first name and last name and optional
            # company name and returns the LinkedIn profile. If you use this method you do not need
            # to call get_linkedin_profile.

            if not function_call:
                logging.error(f'could not bind function call for func: {func}, expr: {expr}')
                self._result = f'could not bind and or execute the function: {func} expr: {expr}'
                yield self

            # todo need to properly parse this.
            if ' = ' not in bindable:
                # no identifier, so we'll create one to capture the result
                identifier = 'result_{}'.format(str(time.time()).replace('.', ''))
                bindable = '{} = {}'.format(identifier, bindable)
            else:
                identifier = bindable.split(' = ')[0].strip()

            # execute the function
            # todo: figure out what to do when you get back None, or ''
            starlark_code = bindable
            locals_result = {}

            try:
                locals_result = StarlarkRuntime(
                    controller=self.controller,
                    agents=self.agents,
                    vector_search=self.starlark_runtime.vector_search,
                ).run(starlark_code, '')

                self._result = locals_result[identifier]
                yield self

                # if we're here, it's because we've been next'ed() and it was the wrong binding
                # reset the binding parameters and try again.
                counter = 0
                bound = False

            except Exception as ex:
                logging.error('Error executing function call: {}'.format(ex))
                counter += 1
                starlark_code = self.starlark_runtime.rewrite_starlark_error_correction(
                    query=self.original_query,
                    starlark_code=starlark_code,
                    error=str(ex),
                    locals_dictionary=self.scope_dict,
                )

        # we should probably return uncertain_or_error here.
        # raise ValueError('could not bind and or execute the function: {} expr: {}'.format(func, expr))
        self._result = f'could not bind and or execute the function: {func} expr: {expr}'
        yield self

    def bind(
        self,
        expr,
        func,
    ) -> 'FunctionBindable':
        for bindable in self.binder(expr, func):
            return bindable

        raise ValueError('could not bind and or execute the function: {} expr: {}'.format(func, expr))
