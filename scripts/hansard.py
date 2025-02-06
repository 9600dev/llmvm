import re
import requests
import click
import os
from bs4 import BeautifulSoup

def extract_pdf_links(html_content):
    # Find all PDF links using regex
    pdf_pattern = r'https://parlinfo\.aph\.gov\.au/[^"\']*?\.pdf'
    pdf_links = re.findall(pdf_pattern, html_content)
    return pdf_links


def get_last_page(html):
    pattern = r'<li class="last">\s*<a href="\?page=(\d+)'
    match = re.search(pattern, html)
    if match:
        last_page = match.group(1)
        return last_page
    else:
        raise ValueError("No last page found in the HTML content")


def get_url(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

    response = requests.get(url, headers=headers)
    html_content = response.content.decode('utf-8')
    return html_content


def get_pdf(url) -> bytes:
    print(f'getting pdf from {url}')
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    response = requests.get(url, headers=headers)
    pdf_content = response.content
    return pdf_content

def paginate(url) -> str:
    pattern = r"page=(\d+)&"
    match = re.search(pattern, url)
    current_page = 0
    if match:
        current_page = int(match.group(1))

    new_page = current_page + 1
    new_url = re.sub(r'page=\d+', f'page={new_page}', url)
    return new_url

def crawl(starting_url, output_dir):
    if 'page=' not in starting_url:
        raise ValueError("Starting URL must contain 'page=' parameter")

    start_page = get_url(starting_url)
    last_page_number = int(get_last_page(start_page))

    url = starting_url

    for page_number in range(1, last_page_number + 1):
        print('paginating to page', page_number)
        url = paginate(url)
        print('extracting links from page ', page_number)
        links = extract_pdf_links(get_url(url))
        counter = 0

        for link in links:
            counter += 1
            pdf_content = get_pdf(link)
            filename = os.path.join(output_dir, f'page_{page_number}_counter_{counter}.pdf')
            print('writing to ', filename)
            with open(filename, 'wb') as f:
                f.write(pdf_content)

@click.command()
@click.argument('url', type=str, required=True)
@click.option('--output_dir', '-o', default='output', required=False, help='Output directory for the PDF files')
def main(url, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    crawl(url, output_dir)

if __name__ == '__main__':
    main()
