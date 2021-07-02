"""Utilities for unit test suite for galaxy.files."""
import os
import tempfile

from galaxy.files import (
    ConfiguredFileSources,
    DictFileSourcesUserContext,
)

TEST_USERNAME = "alice"
TEST_EMAIL = "alice@galaxyproject.org"


def serialize_and_recover(file_sources_o, user_context=None):
    as_dict = file_sources_o.to_dict(for_serialization=True, user_context=user_context)
    file_sources = ConfiguredFileSources.from_dict(as_dict)
    return file_sources


def find_file_a(dir_list):
    return find(dir_list, class_="File", name="a")


def find(dir_list, class_=None, name=None):
    for ent in dir_list:
        if class_ is not None and ent["class"] != class_:
            continue
        if name is not None and ent["name"] == name:
            return ent

    return None


def list_root(file_sources, uri, recursive, user_context=None):
    file_source_pair = file_sources.get_file_source_path(uri)
    file_source = file_source_pair.file_source
    res = file_source.list("/", recursive=recursive, user_context=user_context)
    return res


def list_dir(file_sources, uri, recursive, user_context=None):
    file_source_pair = file_sources.get_file_source_path(uri)
    file_source = file_source_pair.file_source
    print(file_source_pair.path)
    print(uri)
    res = file_source.list(file_source_pair.path, recursive=recursive, user_context=user_context)
    return res


def user_context_fixture(user_ftp_dir=None, role_names=None, group_names=None, is_admin=False):
    user_context = DictFileSourcesUserContext(
        username=TEST_USERNAME,
        email=TEST_EMAIL,
        user_ftp_dir=user_ftp_dir,
        preferences={
            'webdav|password': 'secret1234',
            'dropbox|access_token': os.environ.get('GALAXY_TEST_DROPBOX_ACCESS_TOKEN'),
        },
        role_names=role_names or set(),
        group_names=group_names or set(),
        is_admin=is_admin,
    )
    return user_context


def assert_realizes_as(file_sources, uri, expected, user_context=None):
    file_source_path = file_sources.get_file_source_path(uri)
    with tempfile.NamedTemporaryFile(mode='r') as temp:
        file_source_path.file_source.realize_to(file_source_path.path, temp.name, user_context=user_context)
        with open(temp.name, "r") as f:
            realized_contents = f.read()
        if realized_contents != expected:
            message = "Expected to realize contents at [{}] as [{}], instead found [{}]".format(
                uri,
                expected,
                realized_contents,
            )
            raise AssertionError(message)


def assert_realizes_contains(file_sources, uri, expected, user_context=None):
    file_source_path = file_sources.get_file_source_path(uri)
    with tempfile.NamedTemporaryFile(mode='r') as temp:
        file_source_path.file_source.realize_to(file_source_path.path, temp.name, user_context=user_context)
        realized_contents = temp.read()
        if expected not in realized_contents:
            message = "Expected to realize contents at [{}] to contain [{}], instead found [{}]".format(
                uri,
                expected,
                realized_contents,
            )
            raise AssertionError(message)


def write_from(file_sources, uri, content, user_context=None):
    file_source_path = file_sources.get_file_source_path(uri)
    with tempfile.NamedTemporaryFile(mode='w') as f:
        f.write(content)
        f.flush()
        file_source_path.file_source.write_from(file_source_path.path, f.name, user_context=user_context)
