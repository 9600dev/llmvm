import asyncio
import os
import tempfile
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import httpx
import aiofiles

from bs4 import BeautifulSoup
import playwright
import playwright.async_api

from llmvm.common.helpers import Helpers
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import Content, DownloadParams, FileContent, LLMCall, MarkdownContent, PdfContent, User
from llmvm.server.python_execution_controller import ExecutionController
from llmvm.server.tools.webhelpers import ChromeHelpers, WebHelpers
from llmvm.common.helpers import write_client_stream

logging = setup_logging()


class WebAndContentDriver():
    def __init__(
        self,
        cookies: List[Dict] = [],
    ):
        self.cookies = cookies

    async def __requests_download(self, url, filename: str):
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=True)
            async with aiofiles.open(filename, 'wb') as file:
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    await file.write(chunk)

    def download(self, download: DownloadParams) -> Content:
        logging.debug('WebAndContentDriver.download: {}'.format(download['url']))

        # the client can often send through urls with quotes around them
        if download['url'].startswith('"') and download['url'].endswith('"'):
            download['url'] = download['url'][1:-1]

        # deal with files
        result = urlparse(download['url'])
        if result.scheme == '' or result.scheme == 'file':
            if '.pdf' in result.path:
                return PdfContent(sequence=b'', url=str(result.path))
            if '.htm' in result.path or '.html' in result.path:
                return WebHelpers.convert_html_to_markdown(open(result.path, 'r').read(), url=download['url'])

        # deal with pdfs
        elif (result.scheme == 'http' or result.scheme == 'https') and '.pdf' in result.path:
            chrome_helper = ChromeHelpers(cookies=self.cookies)
            loop = asyncio.get_event_loop()
            # downloads the pdf and gets a local file url
            task = loop.create_task(chrome_helper.pdf_url(download['url']))

            pdf_filename = loop.run_until_complete(task)
            _ = loop.run_until_complete(loop.create_task(chrome_helper.close()))
            return PdfContent(sequence=b'', url=pdf_filename)

        # deal with csv files
        elif (result.scheme == 'http' or result.scheme == 'https') and '.csv' in result.path:
            chrome_helper = ChromeHelpers(cookies=self.cookies)
            loop = asyncio.get_event_loop()
            task = loop.create_task(chrome_helper.download(download['url']))
            csv_filename = loop.run_until_complete(task)
            _ = loop.run_until_complete(loop.create_task(chrome_helper.close()))
            return FileContent(sequence=b'', url=csv_filename)

        # deal with websites
        elif result.scheme == 'http' or result.scheme == 'https':
            # special case for arxiv.org because in headless mode things get weird
            if 'arxiv.org' in download['url']:
                download_url = download['url'].replace('/abs/', '/pdf/')
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    asyncio.run(self.__requests_download(download_url, temp_file.name))
                    return PdfContent(sequence=b'', url=temp_file.name)

            chrome_helper = ChromeHelpers(cookies=self.cookies)
            loop = asyncio.get_event_loop()
            try:
                task = loop.create_task(chrome_helper.get_url(download['url']))
                result = loop.run_until_complete(task)
                _ = loop.run_until_complete(loop.create_task(chrome_helper.close()))
                # sometimes results can be a downloaded file (embedded pdf in the chrome browser)
                # so we have to deal with that.
                if os.path.exists(result) and Helpers.is_pdf(open(result, 'rb')):
                    return PdfContent(sequence=b'', url=result)
                elif os.path.exists(result):
                    return FileContent(sequence=b'', url=result)
                else:
                    return WebHelpers.convert_html_to_markdown(result, url=download['url'])
            except Exception as e:
                logging.debug(f'WebAndContentDriver.download() exception: {e}')
                # see if the browser is trying to download a file
                chrome_helper = ChromeHelpers(cookies=self.cookies)
                loop = asyncio.get_event_loop()
                task = loop.create_task(chrome_helper.download(download['url']))
                result = loop.run_until_complete(task)
                _ = loop.run_until_complete(loop.create_task(chrome_helper.close()))
                if os.path.exists(result) and Helpers.is_pdf(open(result, 'rb')):
                    return PdfContent(sequence=b'', url=result)
                elif os.path.exists(result):
                    return FileContent(sequence=b'', url=result)
                else:
                    return WebHelpers.convert_html_to_markdown(result, url=download['url'])

        # else, nothing
        return Content(f'WebAndContentDriver.download: nothing found for {download["url"]}')

    def download_with_goal(
            self,
            download: DownloadParams,
            controller: ExecutionController,
        ) -> Content:
        logging.debug(
            'WebAndContentDriver.download_with_goal: url={} goal={} search_term={}'
            .format(download['url'], download['goal'], download['search_term'])
        )
        # here we're going to go to the url and see if it's the correct content or not, based on the goal
        result = urlparse(download['url'])

        if (
            '.pdf' in result.path
            or '.csv' in result.path
        ):
            return self.download(download)

        # special case for arxiv.org because in headless mode things get weird
        if 'arxiv.org' in download['url']:
            download_url = download['url'].replace('/abs/', '/pdf/')
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                asyncio.run(self.__requests_download(download_url, temp_file.name))
                return PdfContent(sequence=b'', url=temp_file.name)

        chrome_helper = ChromeHelpers(cookies=self.cookies)
        loop = asyncio.get_event_loop()
        task = loop.create_task(chrome_helper.get_url(download['url']))
        result = loop.run_until_complete(task)

        if os.path.exists(result) and Helpers.is_pdf(open(result, 'rb')):
            return PdfContent(sequence=b'', url=result)
        elif os.path.exists(result):
            return FileContent(sequence=b'', url=result)

        markdown_content = MarkdownContent(sequence=WebHelpers.convert_html_to_markdown(result, url=download['url']).get_str(), url=download['url'])

        write_client_stream(f'Checking the content of {download["url"]} against the original user query of \"{download["goal"]}\".\n')

        next_action = controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_message(
                    prompt_name='download_and_validate.prompt',
                    template={
                        'url': download['url'],
                        'user_goal': download['goal'],
                        'referring_search_term': download['search_term'] or '',
                    },
                    user_token=controller.get_executor().user_token(),
                    assistant_token=controller.get_executor().assistant_token(),
                    append_token=controller.get_executor().append_token(),
                ),
                context_messages=[User(markdown_content)],
                executor=controller.get_executor(),
                model=controller.get_executor().get_default_model(),
                temperature=0.0,
                max_prompt_len=controller.get_executor().max_input_tokens(),
                completion_tokens_len=controller.get_executor().max_output_tokens(),
                prompt_name='download_and_validate.prompt',
            ),
            query=download['search_term'] or '',
            original_query=download['goal'],
        )

        next_action_str = next_action.message.get_str()
        loop.run_until_complete(chrome_helper.close())
        logging.debug(f'WebAndContentDriver.download_with_goal decision: {next_action_str}')


        if 'yes' in next_action_str.lower():
            write_client_stream(f'Yes, the content looks good. "{next_action_str}"\n')
            return markdown_content
        else:
            write_client_stream(f'Decided to proceed to {next_action_str}.\n')

            download_params = DownloadParams({
                'url': Helpers.get_full_url(download['url'], next_action_str),
                'goal': download['goal'],
                'search_term': download['search_term'],
            })
            return self.download(download_params)

