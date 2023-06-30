class LangChainExecutor(Executor):
    def __init__(
            self,
            openai_key: str,
            temperature: float = 0.6,
            verbose: bool = True,
    ):
        self.openai_key = openai_key
        self.temperature = temperature
        self.verbose = verbose
        self.gpt = langchain_OpenAI(temperature=temperature)  # type: ignore
        # Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
        self.tools = load_tools(
            ["serpapi", "llm-math", "news-api"],
            llm=self.gpt,
            news_api_key='ecdb70595c9c4464ac70d338610c9390'
        )
        self.chat = False

        self.context: Any = None

    def name(self) -> str:
        return 'langchain'

    def chat_context(self, chat: bool):
        self.chat = chat

    def parse_action(self, query: str) -> Tuple[str, Callable]:
        if '.pdf' in query:
            from langchain.vectorstores import FAISS

            url = Helpers.extract_token(query, '.pdf')
            from urllib.parse import urlparse

            from langchain.document_loaders import PyPDFLoader

            documents = []

            result = urlparse(url)
            if result.scheme == '' or result.scheme == 'file':
                logging.debug('LangChainExecutor.parse_action loading and splitting {}'.format(result))
                loader: BaseLoader = PyPDFLoader(result.path)
                documents = loader.load_and_split()

                if len(documents) == 0:
                    # use tesseract to do the parsing instead
                    import pdf2image
                    import pytesseract
                    from langchain.document_loaders import TextLoader
                    from pytesseract import Output, TesseractError

                    text: List[str] = []
                    images = pdf2image.convert_from_path(result.path)  # type: ignore
                    for pil_im in images:
                        ocr_dict = pytesseract.image_to_data(pil_im, output_type=Output.DICT)
                        text.append(' '.join(ocr_dict['text']))

                    with tempfile.NamedTemporaryFile('w') as temp:
                        temp.write('\n'.join(text))
                        temp.flush()

                        loader = TextLoader(temp.name)
                        documents = loader.load()

            elif result.scheme == 'https' or result.scheme == 'http':
                import io
                response = requests.get(url=url, timeout=20)
                with tempfile.NamedTemporaryFile(suffix='.pdf', mode='wb', delete=True) as temp_file:
                    temp_file.write(response.content)
                    loader = PyPDFLoader(temp_file.name)
                    documents = loader.load_and_split()

            text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings(
                model='text-embedding-ada-002',
                openai_api_key=self.openai_key,
            )  # type: ignore

            logging.debug('LangChainExecutor.parse_action FAISS.from_documents {}'.format(result))
            docsearch = FAISS.from_documents(texts, embeddings)
            llm = ChatOpenAI(
                openai_api_key=self.openai_key,
                model_name='gpt-3.5-turbo',  # type: ignore
                temperature=self.temperature,
            )  # type: ignore

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=docsearch.as_retriever()
            )

            self.context = qa
            return (query, qa.run)
        elif 'edgar(' in query:
            symbol = Helpers.in_between(query, 'edgar(', ')')
            logging.debug('loading firefox and getting the latest 10Q for {}'.format(symbol))
            report_text = EdgarHelpers.get_latest_form_text(symbol, EdgarHelpers.FormType.TENQ)
            with open('edgar.text', 'w') as f:
                f.write(report_text)

            query = Helpers.strip_between(query, 'edgar(', ')')

            documents = []

            from langchain.document_loaders.text import TextLoader

            with tempfile.NamedTemporaryFile(suffix='.txt', mode='w', delete=True) as t:
                t.write(report_text)
                t.seek(0)
                logging.debug('parsing in BeautifulSoup')
                html_loader = TextLoader(t.name)
                data = html_loader.load()
                documents = html_loader.load_and_split()

            logging.debug('token splitting')
            text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings(
                model='text-embedding-ada-002',
                openai_api_key=self.openai_key,
            )  # type: ignore

            logging.debug('LangChainExecutor.parse_action FAISS.from_documents {}'.format(symbol))
            docsearch = FAISS.from_documents(texts, embeddings)
            llm = ChatOpenAI(
                openai_api_key=self.openai_key,
                model_name='gpt-3.5-turbo',  # type: ignore
                temperature=self.temperature,
            )  # type: ignore

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=docsearch.as_retriever()
            )

            self.context = qa
            return (query, qa.run)
        else:
            # generic langchain
            agent = initialize_agent(self.tools, self.gpt, agent='zero-shot-react-description', verbose=self.verbose)  # type: ignore
            self.context = agent

            def execute_agent(query: str):
                result = agent({'input': query})
                return result['output']
            return (query, execute_agent)

    def execute(self, query: Union[str, List[Dict]], data: str) -> Assistant:
        logging.debug('LangChainExecutor.execute_query query={}'.format(query))

        # todo check to see if we've got repl context
        prompt, action = self.parse_action(str(query))
        return Assistant(
            message=Content(Text(action(prompt))),
            error=False,
            llm_call_context=None,
        )

    def can_execute(self, query: Union[str, List[Dict]], data: str) -> bool:
        return True


