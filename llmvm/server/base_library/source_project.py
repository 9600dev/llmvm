from typing import List

from llmvm.common.helpers import Helpers, write_client_stream
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import Content, LLMCall, User, bcl
from llmvm.server.base_library.source import Source
from llmvm.server.starlark_execution_controller import ExecutionController

logging = setup_logging()


@bcl('llmvm.server.prompts.starlark.source_project.prompt')
class SourceProject:
    def __init__(
        self,
        controller: ExecutionController
    ):
        # todo: this is weirdly circular - Controller -> Runtime -> BCL -> Controller. Weird.. Fix.
        self.controller = controller
        self.sources: List[Source] = []
        self.other_files: List[str] = []

    def set_files(self, files):
        for source_path in files:
            source = Source(source_path)
            if source.tree:
                self.sources.append(source)
            else:
                self.other_files.append(source_path)

    @bcl
    def get_source_structure(self) -> str:
        """
        gets all class names, method names, and docstrings for all classes and methods in all files
        listed in "Files:". This method does not return any source code.
        """
        structure = ''
        for source in self.sources:
            structure += f'File: {source.file_path}\n'
            for class_def in source.get_classes():
                structure += f'class {class_def.name}:\n'
                structure += f'    """{class_def.docstring}"""\n'
                structure += '\n'
                for method_def in source.get_methods(class_def.name):
                    structure += f'    def {method_def.name}:\n'
                    structure += f'        """{method_def.docstring}"""\n'
                    structure += '\n'
            structure += '\n\n'

        return structure

    @bcl
    def get_source_summary(self, file_path: str) -> str:
        """
        gets all class names, method names and natural language descriptions of class and method names
        for a given source file. The file_name must be in the "Files:" list. It does not return any source code.
        """
        def _summary_helper(source: Source) -> str:
            write_client_stream(Content(f"Asking LLM to summarize file: {source.file_path}\n\n"))
            summary = ''
            for class_def in source.get_classes():
                summary += f'class {class_def.name}:\n'
                summary += f'    """{class_def.docstring}"""\n'
                summary += '\n'

                method_definition = ''
                for method_def in source.get_methods(class_def.name):
                    method_definition += f'    def {method_def.name}({", ".join([param[0] for param in method_def.params])})\n'
                    method_definition += f'        """{method_def.docstring}"""\n'
                    method_definition += '\n'
                    method_definition += f'        {source.get_method_source(method_def.name)}\n\n'
                    method_definition += f'Summary of method {method_def.name}:\n'

                    # get the natural language definition
                    assistant = self.controller.execute_llm_call(
                        llm_call=LLMCall(
                            user_message=Helpers.prompt_message(
                                prompt_name='code_method_definition.prompt',
                                template={},
                                user_token=self.controller.get_executor().user_token(),
                                assistant_token=self.controller.get_executor().assistant_token(),
                                append_token=self.controller.get_executor().append_token(),
                            ),
                            context_messages=[User(Content(method_definition))],
                            executor=self.controller.get_executor(),
                            model=self.controller.get_executor().get_default_model(),
                            temperature=0.0,
                            max_prompt_len=self.controller.get_executor().max_input_tokens(),
                            completion_tokens_len=self.controller.get_executor().max_output_tokens(),
                            prompt_name='code_method_definition.prompt',
                        ),
                        query='',
                        original_query='',
                    )

                    method_definition += str(assistant.message) + '\n\n'
                    summary += method_definition
                    write_client_stream(Content(method_definition))

                summary += '\n\n'
            return summary

        for source in self.sources:
            if source.file_path == file_path:
                return _summary_helper(source)
        raise ValueError(f"Source file not found: {file_path}")

    @bcl
    def get_files(self):
        return self.sources

    @bcl
    def get_source(self, file_path):
        for source in self.sources:
            if source.file_path == file_path:
                return source.source_code

        for source in self.other_files:
            if source == file_path:
                with open(source, 'r') as file:
                    return file.read()

        raise ValueError(f"Source file not found: {file_path}")

    @bcl
    def get_methods(self, class_name) -> List['Source.Symbol']:
        methods = []
        for source in self.sources:
            methods.extend(source.get_methods(class_name))
        return methods

    @bcl
    def get_classes(self) -> List['Source.Symbol']:
        classes = []
        for source in self.sources:
            classes.extend(source.get_classes())
        return classes

    @bcl
    def get_references(self, method_name) -> List['Source.Callsite']:
        references = []
        for source in self.sources:
            references.extend(Source.get_references(source.get_tree(), method_name))
        return references
