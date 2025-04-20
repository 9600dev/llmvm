import asyncio
import os
import tempfile
from typing import Dict, Optional, Tuple, cast
from urllib.parse import urlparse
import httpx
import aiofiles

from bs4 import BeautifulSoup
import playwright
import playwright.async_api

from llmvm.common.helpers import Helpers
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import Assistant, ContainerContent, Content, Message, TextContent, DownloadParams, FileContent, LLMCall, MarkdownContent, PdfContent, User
from llmvm.server.python_execution_controller import ExecutionController
from llmvm.server.tools.webhelpers import ChromeHelpers, WebHelpers
from llmvm.common.helpers import write_client_stream

logging = setup_logging()


class WebAndContentDriver():
    def __init__(
        self,
        cookies: list[Dict] = [],
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

        download['url'] = download['url'].replace(';fileType=text%2Fxml', '')

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

        # deal with xml files
        elif (result.scheme == 'http' or result.scheme == 'https') and '.xml' in result.path:
            chrome_helper = ChromeHelpers(cookies=self.cookies)
            loop = asyncio.get_event_loop()
            task = loop.create_task(chrome_helper.download(download['url']))
            xml_filename = loop.run_until_complete(task)
            _ = loop.run_until_complete(loop.create_task(chrome_helper.close()))

            with open(xml_filename, 'r') as f:
                xml_file_content = f.read()
                xml_str = WebHelpers.convert_xml_to_text(xml_file_content)
                return TextContent(xml_str)

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
                    return cast(MarkdownContent, WebHelpers.convert_html_to_markdown(result, url=download['url']))

        # else, nothing
        return TextContent(f'WebAndContentDriver.download: nothing found for {download["url"]}')

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

        markdown_content = WebHelpers.convert_html_to_markdown(result, url=download['url'])

        write_client_stream(f'Checking the content of {download["url"]} against the original user query of \"{download["goal"]}\".\n')

        next_action: Assistant = controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_user(
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
                context_messages=cast(list[Message], [User(markdown_content)]),
                executor=controller.get_executor(),
                model=controller.get_executor().default_model,
                temperature=0.0,
                max_prompt_len=controller.get_executor().max_input_tokens(),
                completion_tokens_len=controller.get_executor().max_output_tokens(),
                prompt_name='download_and_validate.prompt',
            ),
            query=download['search_term'] or '',
            original_query=download['goal'],
        )

        next_action_str = next_action.get_str()
        loop.run_until_complete(chrome_helper.close())
        logging.debug(f'WebAndContentDriver.download_with_goal decision: {next_action_str}')


        if 'yes' in next_action_str.lower():
            write_client_stream(f'Yes, the content looks good. "{next_action_str}"\n')
            return markdown_content
        elif 'no' in next_action_str.lower():
            write_client_stream('No, the content does not look good, and does not provide a path forward.\n')
            return TextContent('No, the content does not look good, and does not provide a path forward.')
        else:
            write_client_stream(f'Decided to proceed to {next_action_str}.\n')

            download_params = DownloadParams({
                'url': Helpers.get_full_url(download['url'], next_action_str),
                'goal': download['goal'],
                'search_term': download['search_term'],
            })
            return self.download(download_params)

    async def download_multiple_async(
        self, downloads: list[DownloadParams],
        max_concurrent: int = 5
    ) -> list[Tuple[DownloadParams, Content]]:
        """
        Download multiple URLs concurrently.

        Args:
            downloads: list of DownloadParams objects
            max_concurrent: Maximum number of concurrent downloads

        Returns:
            list of tuples containing (original_download_params, content)
        """

        if not isinstance(downloads, list):
            downloads = [downloads]

        if isinstance(downloads, list):
            downloads = Helpers.flatten(downloads)

        if not all('url' in d for d in downloads):
            raise ValueError('All downloads must be DownloadParams objects.')

        # Clean up URLs
        for download in downloads:
            if download['url'].startswith('"') and download['url'].endswith('"'):
                download['url'] = download['url'][1:-1]
            download['url'] = download['url'].replace(';fileType=text%2Fxml', '')

        # Group URLs by type for efficient processing
        pdf_urls = []
        csv_urls = []
        xml_urls = []
        web_urls = []
        file_urls = []
        arxiv_urls = []

        for download in downloads:
            result = urlparse(download['url'])
            if result.scheme == '' or result.scheme == 'file':
                file_urls.append(download)
            elif (result.scheme == 'http' or result.scheme == 'https'):
                if '.pdf' in result.path:
                    pdf_urls.append(download)
                elif '.csv' in result.path:
                    csv_urls.append(download)
                elif '.xml' in result.path:
                    xml_urls.append(download)
                elif 'arxiv.org' in download['url']:
                    arxiv_urls.append(download)
                else:
                    web_urls.append(download)

        results = []

        # Process file URLs (these are local and don't need parallelism)
        for download in file_urls:
            content = await self._process_file_url(download)
            results.append((download, content))

        # Process arxiv URLs (special case)
        arxiv_results = await self._process_arxiv_urls(arxiv_urls)
        results.extend(arxiv_results)

        # Process web URLs in parallel (the main benefit)
        if web_urls:
            chrome_helper = ChromeHelpers(cookies=self.cookies)

            # Extract just the URLs for parallel processing
            urls = [download['url'] for download in web_urls]
            url_to_download = {download['url']: download for download in web_urls}

            # Process URLs concurrently
            parallel_results = await chrome_helper.concurrent_process_urls(urls, max_concurrent)
            await chrome_helper.close()

            # Convert results back to Content objects
            for url, html_content, _ in parallel_results:
                download = url_to_download[url]

                # Check if the result is a file path (for downloaded content)
                if os.path.exists(html_content) and Helpers.is_pdf(open(html_content, 'rb')):
                    content = PdfContent(sequence=b'', url=html_content)
                elif os.path.exists(html_content):
                    content = FileContent(sequence=b'', url=html_content)
                else:
                    content = WebHelpers.convert_html_to_markdown(html_content, url=download['url'])

                results.append((download, content))

        # Process PDF, CSV, and XML URLs (these could also be parallelized)
        # For simplicity, I'll process these in sequence for now
        for download in pdf_urls + csv_urls + xml_urls:
            content = await self._process_special_url(download)
            results.append((download, content))

        return results

    async def _process_file_url(self, download: DownloadParams) -> Content:
        """Process a local file URL."""
        result = urlparse(download['url'])
        if '.pdf' in result.path:
            return PdfContent(sequence=b'', url=str(result.path))
        if '.htm' in result.path or '.html' in result.path:
            return WebHelpers.convert_html_to_markdown(open(result.path, 'r').read(), url=download['url'])
        return TextContent(f'WebAndContentDriver: nothing found for {download["url"]}')

    async def _process_arxiv_urls(self, downloads: list[DownloadParams]) -> list[Tuple[DownloadParams, PdfContent]]:
        """Process arxiv.org URLs in parallel."""
        results = []

        async def download_arxiv(download: DownloadParams):
            download_url = download['url'].replace('/abs/', '/pdf/')
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                await self.__requests_download(download_url, temp_file.name)
                return download, PdfContent(sequence=b'', url=temp_file.name)

        tasks = [download_arxiv(download) for download in downloads]
        if tasks:
            results = await asyncio.gather(*tasks)

        return results

    async def _process_special_url(self, download: DownloadParams) -> Content:
        """Process PDF, CSV, or XML URLs."""
        result = urlparse(download['url'])
        chrome_helper = ChromeHelpers(cookies=self.cookies)

        try:
            if '.pdf' in result.path:
                pdf_filename = await chrome_helper.pdf_url(download['url'])
                await chrome_helper.close()
                return cast(Content, PdfContent(sequence=b'', url=pdf_filename))
            elif '.csv' in result.path:
                csv_filename = await chrome_helper.download(download['url'])
                await chrome_helper.close()
                return cast(Content, FileContent(sequence=b'', url=csv_filename))
            elif '.xml' in result.path:
                xml_filename = await chrome_helper.download(download['url'])
                await chrome_helper.close()

                with open(xml_filename, 'r') as f:
                    xml_file_content = f.read()
                    xml_str = WebHelpers.convert_xml_to_text(xml_file_content)
                    return cast(Content, TextContent(xml_str))
            else:
                raise ValueError(f"Unknown content type: {result.path}")
        except Exception as e:
            logging.debug(f'_process_special_url exception: {e}')
            await chrome_helper.close()
            return cast(Content, TextContent(f'Error processing {download["url"]}: {str(e)}'))