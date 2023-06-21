    @staticmethod
    def walk_tree_and_convert_markdown(directory: str, output_directory: str):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.html') or file.endswith('.htm'):
                    with open(os.path.join(root, file), 'r') as f:
                        text = f.read()
                        hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                        if os.path.exists(os.path.join(output_directory, hash + '.md')):
                            continue

                        markdown = WebHelpers.html_to_markdown(text)
                        with open(os.path.join(output_directory, hash + '.md'), 'w') as f:
                            logging.debug('writing file: {} as {}'.format(os.path.join(root, file), os.path.join(output_directory, hash + '.md')))
                            f.write(markdown)

