import sys
import os
import rich
import click
import subprocess

from bs4 import BeautifulSoup
from rich.markdown import Markdown
from typing import Optional, cast
from anthropic import Anthropic


MODEL_DEFAULT='claude-sonnet-4-20250514'


class InternalSearcher():
    def __init__(
        self,
        controller,
        runtime,
    ):
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def _log(self, message: str, prepend: str = '[DEBUG stderr]: ', end: str = '\n'):
        sys.stderr.write(f"{prepend}{message}{end}")
        sys.stderr.flush()

    def _llm_call(self, messages: list[str], model: str = MODEL_DEFAULT) -> str:
        messages_dict = []

        for i, msg in enumerate(messages):
            if i % 2:
                messages_dict.append({'role': 'assistant', 'content': 'I am ready for your next message.'})
                messages_dict.append({'role': 'user', 'content': msg})
            else:
                messages_dict.append({'role': 'user', 'content': msg})

        completion = self.client.messages.stream(
            messages=messages_dict,
            model=model,
            max_tokens=4096,
        )

        result = ''

        with completion as stream:
            for text in stream.text_stream:
                self._log(text, prepend='', end='')
                result += text
        return result

    def _run_applescript(self, script):
        """Run an AppleScript command and return its output."""
        process = subprocess.Popen(['osascript', '-e', script],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        out, _ = process.communicate()
        return out.decode('utf-8').strip()

    def _get_page_html(self, url) -> str:
        self._log(f"InternalSearcher.get_page_html: {url}")
        """Navigate to a URL in Chrome and get its HTML content."""
        script = f'''
        tell application "Google Chrome"
            open location "{url}"
            delay 3
            set html_content to execute front window's active tab javascript "document.documentElement.outerHTML"
            return html_content
        end tell
        '''
        return self._run_applescript(script)

    def _get_chrome_tabs_url_and_title(self) -> str:
        applescript = '''
        tell application "Google Chrome"
            set tabList to {}
            set windowList to every window
            repeat with theWindow in windowList
                set tabList to tabList & (every tab of theWindow whose URL is not "")
            end repeat

            set output to ""
            repeat with theTab in tabList
                set theURL to URL of theTab
                set theTitle to title of theTab
                set output to output & theURL & "," & theTitle & linefeed
            end repeat

            return output
        end tell
        '''
        return self._run_applescript(applescript)

    def _search_outline(self, query):
        self._log(f"InternalSearcher.search_outline: {query}")
        encoded_query = query
        url = f"https://outline.ant.dev/search/{encoded_query}"
        html = self._get_page_html(url)

        soup = BeautifulSoup(html, 'html.parser')
        results = soup.find_all('a', role='menuitem')

        for a_tag in results:
            for tag in a_tag.find_all(['button', 'path', 'svg']):
                tag.decompose()

        result = 'These are the search results from Outline. The full URL for the outline wiki site is https://outline.ant.dev.\n\n'
        for i, r in enumerate(results):
            result += f'Result: {str(results[i].contents)}\n\n\n'
        result += 'End of search results from Outline.\n\n'
        return result

    def _search_google(self, query):
        self._log(f"InternalSearcher.search_google: {query}")
        encoded_query = query
        url = f"https://cloudsearch.google.com/cloudsearch/search?authuser=0&q={encoded_query}"
        html = self._get_page_html(url)

        soup = BeautifulSoup(html, 'html.parser')
        results = soup.select("div > div > h3")

        result = 'These are the search results from Google Cloud Search.\n\n'
        for i, r in enumerate(results):
            result += f'Result: {str(results[i].contents)}\n\n\n'
        result += 'End of search results from Google Cloud Search.\n\n'
        return result

    def _search_google_docs(self, query):
        self._log(f"InternalSearcher.search_google_docs: {query}")
        encoded_query = query
        url = f"https://docs.google.com/document/u/0/?tgif=d&q={encoded_query}"
        html = self._get_page_html(url)

        import re

        titles = re.findall(r'docs-homescreen-list-item-title-value">(.*?)</div>', html)
        owners = re.findall(r'aria-label="Owned by (.*?)"', html)
        dates = re.findall(r'aria-label="Last opened by me (.*?)"', html)

        result = 'These are the search results from Google Docs Search. Unfortunately I dont have URLs for these results\n'
        result += 'so for the clickable URL you can use the Google Doc Search URL with the title: https://docs.google.com/document/u/0/?tgif=d&q=title_of_the_document\n\n'

        zip_results = list(zip(titles, owners, dates))

        for r in zip_results:
            result += f"Title: {r[0]}\nOwner: {r[1]}\nDate: {r[2]}\n\n"
        result += 'End of search results from Google Docs Search.\n\n'
        return result

    def search(
            self,
            query: str,
            instructions: Optional[str] = None,
            model: str = MODEL_DEFAULT,
            context: Optional[str] = None,
            hardcore: bool = False,
            tabs: bool = False,
        ) -> str:
        """
        Search for a query and return the results from Outline, Google Docs, and Google Cloud Search.
        This will rank the search results from each source in order of relevance.

        :param query: The search query to search for.
        :type query: str
        :return: The search results from Outline, Google Docs, and Google Cloud Search.
        :rtype: str
        """
        list_of_queries = [query]
        if hardcore:
            additional_instructions = f'The user also added additional instructions that you should take into consideration: {instructions}' or ''

            QUERY_GENERATOR = f"""
            I want you to take a user query, and deconstruct the query in to two search queries that you think
            will best satisfy the users original query/intent.

            Only generate two search queries. You should put them into a Python string list and return just that list.
            Do not add or use the original User query in the list.

            Example:
            User: "I want to know about the history of the internet"
            ['history of the internet',  'original construction of the internet']

            The User query is: {query}. {additional_instructions}
            """
            query_results = self._llm_call([QUERY_GENERATOR])
            query_list = query_results.strip()
            self._log(f"InternalSearcher.search: hardcore query_list: {query_list}")
            list_of_queries = list_of_queries + cast(list, eval(query_list))

        outline_results = []
        google_results = []
        google_docs_results = []

        self._run_applescript('tell application "Google Chrome" to activate')

        for query in list_of_queries:
            encoded_query = query
            outline_results.append(self._search_outline(encoded_query))
            google_results.append(self._search_google(encoded_query))
            google_docs_results.append(self._search_google_docs(encoded_query))

        # add ranking instructions
        ranking_instructions = 'Please rank the search results from each source in order of relevance.'
        if instructions:
            ranking_instructions = f'Please rank the search results from each source using these instructions: {instructions}'

        context_message = ''
        if context:
            context_message = f"""
                There is additional context that might be useful for your ranking task from the user.
                That context is in the first message I sent you. You can use that to help you rank the search results.
            """

        tabs_message = ''
        if tabs:
            open_tabs = self._get_chrome_tabs_url_and_title()
            tabs_message += f"""
                I have also added the URLs and titles of all the Users open Chrome tabs as helpful context for you.
                You can think of this as what the user is currently working on, and are informative for you to rank the search results.
                You can only use these URLs and titles as context, you do not need to rank them.

                User Open Chrome Tabs:

                {open_tabs}
            """

        PROMPT = f"""
        I need you to rank the search results from three different search engine sources.

        I have searched the query \"{query}\" on the users behalf from Outline, Google Cloud Search, and Google Docs Search.
        These search results are provided to you in the previous message.

        {ranking_instructions}

        {context_message}

        {tabs_message}

        You should emit Markdown, and it should at least have the following:
            * Title of the search result which should be a clickable markdown link to the full url.
            * Author of the search result (if applicable)
            * Date of the search result (if applicable)
            * The snippet or summary of the search result (must have)
            * Your commentary on what you think this result is about and why you think it is relevant.
            * The URL of the search result

        Example:

        1. [Title of the search result](https://www.example.com)
            * Summary or snippet of search result. Might be a few lines
            * Date of search result
            * Author of search result
            * Your commentary on what you think this result is about and why you think it is relevant.
            * URL of search result
        2. [Title of the search result](https://www.example.com)
            ...

        Just rank the search results and generate the markdown, don't add extra commentary.
        Do not add headings or categories, or cluster the search results. Just rank according to the instructions.
        You can emit up to 15 search results. If you think there is a really good set of 5 ranked results, you
        that match the query very well, you can stop there.
        """

        messages_stack_str = ['\n'.join(outline_results), '\n'.join(google_results), '\n'.join(google_docs_results), PROMPT]
        if context:
            if os.path.exists(os.path.expanduser(context)):
                with open(context, 'r') as f:
                    context = f.read()
            messages_stack_str.insert(0, context)

        return self._llm_call(messages_stack_str, model=model)

@click.command()
@click.argument("query", type=str)
@click.option("--instructions", '-i', default=None, help="Instructions for ranking the search results.")
@click.option('--context', '-c', default=None, help="Extra context for Claude. String or file path.")
@click.option("--model", '-m', default=MODEL_DEFAULT, help="model to use for ranking")
@click.option("--hardcore", '-h', is_flag=True, help="Enable hardcore mode. Claude will generate extra search queries.")
@click.option("--tabs", '-t', is_flag=True, help="Add all my open Chrome tabs and titles as context for Claude")
def main(
    query: str,
    instructions: Optional[str],
    context: Optional[str],
    model: str,
    hardcore: bool,
    tabs: bool,
):
    search = InternalSearcher(None, None)
    result = search.search(query, instructions, model, context, hardcore, tabs)
    rich.console.Console().print(Markdown(result))

if __name__ == "__main__":
    result = main()
    sys.exit(0)
