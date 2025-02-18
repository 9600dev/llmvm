import os
import sys
import subprocess
import click

@click.command()
@click.argument('file_or_glob', required=True)
def cli(
    file_or_glob: str,
):
    if '*' not in file_or_glob:
        file_or_glob = os.path.expanduser(file_or_glob)

    # Set environment variables so the Streamlit app can access them
    os.environ["FILE_OR_GLOB"] = file_or_glob

    if '*' in file_or_glob:
        import glob
        files = sorted(glob.glob(file_or_glob))
        click.echo(f"Found {len(files)} tests to show comparisons for.")
    else:
        click.echo(f"Showing comparison for {file_or_glob}")

    script_dir = os.path.dirname(os.path.realpath(__file__))
    subprocess.run(["streamlit", "run", f"{script_dir}/compare_new.py"])


if __name__ == '__main__':
    cli()