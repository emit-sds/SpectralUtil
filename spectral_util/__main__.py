import click


@click.group(no_args_is_help=True)
def cli(**kwargs):
    """\
    Spectral Utilities

    \b
    Repository: https://github.com/emit-sds/SpectralUtil
    Report an issue: https://github.com/emit-sds/SpectralUtil/issues
    """
    pass


if __name__ == "__main__":
    cli()
