from morphpiece import __version__

def test_installation():
    assert isinstance(__version__, str), "The version of the application is not set"
