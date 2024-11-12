    @staticmethod
    def from_dict(message: dict[str, Any]) -> 'Message':
        role = message['role']
        message_content = message['content']

        # this can be from a MessageModel, which has a url and content_type
        # or from the LLM, which doesn't.
        url = message['url'] if 'url' in message else ''
        content_type = message['content_type'] if 'content_type' in message else ''

        # when converting from MessageModel, there can be an embedded image
        # in the content parameter that needs to be converted back to bytes
        if (
            isinstance(message_content, list)
            and len(message_content) > 0
            and 'type' in message_content[0]
            and message_content[0]['type'] == 'image_url'
            and 'image_url' in message_content[0]
            and 'url' in message_content[0]['image_url']
        ):
            byte_content = base64.b64decode(message_content[0]['image_url']['url'].split(',')[1])
            content = ImageContent(byte_content, message_content[0]['image_url']['url'])

        elif (
            isinstance(message_content, list)
            and len(message_content) > 0
            and 'type' in message_content[0]
            and message_content[0]['type'] == 'image'
        ):
            byte_content = base64.b64decode(message_content[0]['source']['data'])
            content = ImageContent(byte_content, message_content[0]['source']['data'])

        elif content_type == 'pdf':
            if url and not message_content:
                with open(url, 'rb') as f:
                    content = PdfContent(f.read(), url)
            else:
                content = PdfContent(FileContent.decode(str(message_content)), url)
        elif content_type == 'file':
            # if there's a url here, but no content, then it's a file local to the server
            if url and not message_content:
                with open(url, 'r') as f:
                    content = FileContent(f.read().encode('utf-8'), url)
            # else, it's been transferred from the client to server via b64
            else:
                content = FileContent(FileContent.decode(str(message_content)), url)
        elif content_type == 'markdown':
            if url and not message_content:
                with open(url, 'r') as f:
                    content = MarkdownContent(f.read(), url)
            else:
                content = MarkdownContent(MarkdownContent.decode(str(message_content)).decode('utf-8'), url)
        else:
            content = TextContent(sequence=str(message_content), url=url)

        if role == 'user':
            return User(content)
        elif role == 'system':
            return System(content)  # type: ignore
        elif role == 'assistant':
            return Assistant(content)
        raise ValueError(f'role not found or not supported: {message}')

    def __getitem__(self, key):
        return {'role': self.role(), 'content': self.message}

    @staticmethod
    def to_dict(message: 'Message', server_serialization: bool = False) -> dict[str, Any]:
        def file_wrap(message: FileContent | PdfContent | MarkdownContent):
            return f'The following data/content is from this url: {message.url}\n\n{message.get_str()}'

        # primarily to pass to Anthropic or OpenAI api
        if isinstance(message, User) and isinstance(message.message, ImageContent):
            return {
                'role': message.role(),
                'content': [{
                    'type': 'image_url',
                    'image_url': {
                        'url': f"data:image/jpeg;base64,{base64.b64encode(message.message.sequence).decode('utf-8')}",
                        'detail': 'high'
                    }
                }],
                **({'url': message.message.url} if server_serialization else {}),
                **({'content_type': 'image'} if server_serialization else {})
            }
        elif isinstance(message, User) and isinstance(message.message, PdfContent):
            return {
                'role': message.role(),
                'content': message.message.b64encode() if server_serialization else file_wrap(message.message),
                **({'url': message.message.url} if server_serialization else {}),
                **({'content_type': 'pdf'} if server_serialization else {})
            }
        elif isinstance(message, User) and isinstance(message.message, FileContent):
            return {
                'role': message.role(),
                'content': message.message.b64encode() if server_serialization else file_wrap(message.message),
                **({'url': message.message.url} if server_serialization else {}),
                **({'content_type': 'file'} if server_serialization else {})
            }
        elif isinstance(message, User) and isinstance(message.message, MarkdownContent):
            return {
                'role': message.role(),
                'content': message.message.b64encode() if server_serialization else file_wrap(message.message),
                **({'url': message.message.url} if server_serialization else {}),
                **({'content_type': 'markdown'} if server_serialization else {})
            }
        else:
            return {
                'role': message.role(),
                'content': str(message.message),
                **({'url': message.message.url} if server_serialization else {}),
                **({'content_type': ''} if server_serialization else {})
            }


