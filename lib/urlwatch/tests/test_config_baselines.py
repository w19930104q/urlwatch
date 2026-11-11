# -*- coding: utf-8 -*-
#
# Regression tests for per-filter-configuration baselines.
#
# A local HTTP server (stdlib only, no network access) serves stateful
# responses with ETag/Last-Modified based conditional requests, so that the
# cross-process A/B/A baseline switching, HTTP 304 handling and failure
# recovery behavior can be exercised end-to-end.

import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from urlwatch.filters import FilterBase
from urlwatch.handler import JobState
from urlwatch.jobs import ShellJob, UrlJob
from urlwatch.storage import CacheMiniDBStorage, get_job_config_id


class StatefulHttpHandler(BaseHTTPRequestHandler):
    # Shared, mutable server state (set on the server instance by tests)
    body = 'alpha\n'
    etag = '"etag-1"'
    fail_requests = 0
    requests = []

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        type(self).requests.append({
            'if_none_match': self.headers.get('If-None-Match'),
            'if_modified_since': self.headers.get('If-Modified-Since'),
        })

        if type(self).fail_requests > 0:
            type(self).fail_requests -= 1
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'injected failure')
            return

        if self.headers.get('If-None-Match') == type(self).etag:
            self.send_response(304)
            self.end_headers()
            return

        payload = type(self).body.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('ETag', type(self).etag)
        self.send_header('Last-Modified', 'Thu, 01 Jan 1970 00:16:40 GMT')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def http_server(monkeypatch):
    for var in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
                'ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy'):
        monkeypatch.delenv(var, raising=False)

    StatefulHttpHandler.body = 'alpha\n'
    StatefulHttpHandler.etag = '"etag-1"'
    StatefulHttpHandler.fail_requests = 0
    StatefulHttpHandler.requests = []

    server = ThreadingHTTPServer(('127.0.0.1', 0), StatefulHttpHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def cache_storage():
    fd, filename = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    os.remove(filename)
    storage = CacheMiniDBStorage(filename)
    try:
        yield storage
    finally:
        storage.close()
        if os.path.exists(filename):
            os.remove(filename)


def make_job(server, filter, **kwargs):
    kwargs.setdefault('filter_config_baselines', True)
    return UrlJob(url='http://127.0.0.1:%d/page' % server.server_address[1],
                  filter=filter, **kwargs)


def process(cache_storage, job):
    with JobState(cache_storage, job) as job_state:
        job_state.process()
        # Mirror worker.run_jobs(): successful runs reset the retry counter,
        # and every processed run persists its state (failures keep the old
        # body while updating the retry counter).
        if job_state.exception is None:
            job_state.tries = 0
        job_state.save()
        return job_state


# ---------------------------------------------------------------------------
# Fingerprint / grouping rules
# ---------------------------------------------------------------------------

def test_fingerprint_equivalent_spellings_and_grouping():
    url = 'http://example.com/'

    def job(spec, **kwargs):
        return UrlJob(url=url, filter=spec, filter_config_baselines=True, **kwargs)

    # Legacy string form vs. list/dict form, default subfilter mapping,
    # and subfilter parameter ordering must all map to the same identifier.
    string_form = job('grep:foo,html2text')
    list_form = job([{'grep': {'re': 'foo'}}, 'html2text'])
    none_subfilter = job([{'grep': {'re': 'foo'}}, {'html2text': None}])
    reordered_params = job([{'html2text': {'parser': 'html5lib', 'method': 'bs4'}}])
    ordered_params = job([{'html2text': {'method': 'bs4', 'parser': 'html5lib'}}])

    assert get_job_config_id(string_form) == get_job_config_id(list_form)
    assert get_job_config_id(list_form) == get_job_config_id(none_subfilter)
    assert get_job_config_id(reordered_params) == get_job_config_id(ordered_params)

    # The filter chain order is significant.
    swapped = job(['html2text', {'grep': {'re': 'foo'}}])
    assert get_job_config_id(swapped) != get_job_config_id(list_form)

    # Different filters are different configurations.
    assert get_job_config_id(job('strip')) != get_job_config_id(job('reverse'))

    # Encoding participates in the identifier.
    assert get_job_config_id(job('strip', encoding='utf-8')) != \
        get_job_config_id(job('strip', encoding='latin-1'))
    assert get_job_config_id(job('strip', encoding='utf-8')) != \
        get_job_config_id(job('strip'))

    # Name and tags do not participate.
    assert get_job_config_id(job('strip', name='A', tags=['one'])) == \
        get_job_config_id(job('strip', name='B', tags=['two', 'three']))


def test_fingerprint_disabled_and_unsupported_jobs():
    # Disabled by default for URL jobs.
    assert get_job_config_id(UrlJob(url='http://example.com/', filter='strip')) is None

    # Shell jobs are never scoped, even if the key is present.
    assert get_job_config_id(ShellJob(command='date', filter_config_baselines=True)) is None


def test_canonical_chain_rejects_unknown_filter():
    with pytest.raises(ValueError):
        FilterBase.canonical_filter_chain(['this-filter-does-not-exist'])


def test_canonical_chain_handles_nested_and_scalar_values():
    # Nested dict/list subfilter values and scalars are normalized stably.
    first = FilterBase.canonical_filter_chain([{'html2text': {'method': 'bs4', 'width': 80}}])
    reordered = FilterBase.canonical_filter_chain([{'html2text': {'width': 80, 'method': 'bs4'}}])
    assert first == reordered

    # An empty filter chain is valid and distinct from a non-empty one.
    assert FilterBase.canonical_filter_chain(None) == ()
    assert FilterBase.canonical_filter_chain('strip') != ()


def test_job_state_scoping_requires_minidb(cache_storage):
    job = UrlJob(url='http://example.com/', filter='strip', filter_config_baselines=True)
    state = JobState(cache_storage, job)
    assert state.config_id == get_job_config_id(job)
    assert state.config_id is not None

    # Backends without config baseline support always get the legacy scope.
    class DummyStorage:
        supports_config_baselines = False

    legacy_state = JobState(DummyStorage(), job)
    assert legacy_state.config_id is None


# ---------------------------------------------------------------------------
# End-to-end behavior against a local HTTP server
# ---------------------------------------------------------------------------

def test_first_run_sends_no_conditional_headers(http_server, cache_storage):
    job = make_job(http_server, 'grep:alpha')
    state = process(cache_storage, job)

    assert state.exception is None
    assert state.verb is None  # report.*() is called by the worker, not process()
    first_request = StatefulHttpHandler.requests[0]
    assert first_request['if_none_match'] is None
    assert first_request['if_modified_since'] is None
    # The successful first retrieval is stored as a "new" baseline.
    data, _, tries, etag = cache_storage.load(job, job.get_guid(),
                                              config_id=state.config_id)
    assert data == state.new_data
    assert tries == 0
    assert etag == '"etag-1"'


def test_cross_process_aba_restores_full_baseline(http_server, cache_storage):
    """Simulate separate urlwatch processes by reopening the cache DB."""
    cache_path = cache_storage.filename
    guid = None
    config_a = config_b = None

    def reopen():
        storage = CacheMiniDBStorage(cache_path)
        return storage

    # --- Process 1: configuration A builds its baseline ----------------
    storage = reopen()
    # "strip" passes the body through unchanged, so the stored baseline
    # equals the served content and can be compared directly.
    job_a = make_job(http_server, 'strip')
    guid = job_a.get_guid()
    state = process(storage, job_a)
    config_a = state.config_id
    baseline_a = state.new_data
    assert state.exception is None
    assert baseline_a == 'alpha'
    storage.close()

    # The server content changes; configuration B must not compare against A.
    StatefulHttpHandler.body = 'beta\n'
    StatefulHttpHandler.etag = '"etag-2"'

    # --- Process 2: switch to configuration B, fresh baseline ----------
    storage = reopen()
    job_b = make_job(http_server, [{'re.sub': {'pattern': 'beta', 'repl': 'beta'}}])
    config_b = JobState(storage, job_b).config_id
    assert config_b != config_a
    state = process(storage, job_b)
    assert state.exception is None
    assert state.old_data is None  # no baseline for this configuration
    assert state.new_data == 'beta\n'
    # No conditional headers on first use of this configuration.
    second_run_request = StatefulHttpHandler.requests[-1]
    assert second_run_request['if_none_match'] is None
    assert second_run_request['if_modified_since'] is None
    storage.close()

    # Both baselines coexist under the same GUID.
    storage = reopen()
    data_a, ts_a, tries_a, etag_a = storage.load(None, guid, config_id=config_a)
    data_b, ts_b, tries_b, etag_b = storage.load(None, guid, config_id=config_b)
    assert data_a == baseline_a
    assert data_b == 'beta\n'
    # ETags come from the server response and are not part of the fingerprint
    assert etag_a == '"etag-1"'
    assert etag_b == '"etag-2"'

    # The server content goes back to what A recorded.
    StatefulHttpHandler.body = 'alpha\n'
    StatefulHttpHandler.etag = '"etag-3"'

    # --- Process 3: switch back to A; its baseline is fully restored ---
    job_a_again = make_job(http_server, 'strip')
    state = process(storage, job_a_again)
    assert state.config_id == config_a
    assert state.old_data == baseline_a
    # A's own validator (etag) and timestamp were restored from its baseline
    # before the request, so its conditional headers are A's and not B's.
    assert StatefulHttpHandler.requests[-1]['if_none_match'] == '"etag-1"'
    assert state.timestamp == ts_a
    assert state.exception is None
    # The retrieved content matches A's restored baseline, so it is unchanged.
    assert state.new_data == baseline_a
    storage.close()

    storage = reopen()
    restored_a, _, restored_tries, _ = storage.load(
        None, guid, config_id=config_a)
    restored_b, _, _, _ = storage.load(None, guid, config_id=config_b)
    assert restored_a == baseline_a
    assert restored_tries == 0
    assert restored_b == 'beta\n'  # B is untouched by running A
    storage.close()


def test_304_only_reuses_current_configuration(http_server, cache_storage):
    job = make_job(http_server, 'grep:alpha')

    # Baseline for the active configuration.
    state = process(cache_storage, job)
    config_id = state.config_id
    assert state.exception is None

    # Second run: server answers 304 because the stored ETag is sent.
    state = process(cache_storage, job)
    assert type(state.exception).__name__ == 'NotModifiedError'
    latest_request = StatefulHttpHandler.requests[-1]
    assert latest_request['if_none_match'] == '"etag-1"'

    # A second configuration exists but has no baseline; even if the server
    # would 304, no conditional headers are sent and its own baseline is built.
    other_job = make_job(http_server, 'grep:beta')
    other_state = process(cache_storage, other_job)
    assert other_state.config_id != config_id
    assert other_state.old_data is None
    assert other_state.exception is None
    other_request = StatefulHttpHandler.requests[-1]
    assert other_request['if_none_match'] is None
    assert other_request['if_modified_since'] is None

    # The 304 run must not have written additional rows for the first
    # configuration: history still contains exactly the original snapshot.
    history = cache_storage.get_history_data(job.get_guid(), 10,
                                             config_id=config_id)
    assert list(history) == ['alpha']

    # Both configurations have their own single baseline under the same GUID.
    guids = set(cache_storage.get_guids())
    assert guids == {job.get_guid()}


def test_failure_does_not_overwrite_baseline_or_other_configs(http_server, cache_storage):
    job_a = make_job(http_server, 'grep:alpha')
    state_a = process(cache_storage, job_a)
    config_a = state_a.config_id
    baseline_a = state_a.new_data

    job_b = make_job(http_server, 'grep:beta')
    state_b = process(cache_storage, job_b)
    config_b = state_b.config_id
    baseline_b = state_b.new_data

    # The next two requests for A fail with HTTP 500.
    StatefulHttpHandler.fail_requests = 2
    failed_state = process(cache_storage, job_a)
    assert failed_state.exception is not None
    assert failed_state.tries == 1

    # A's successful baseline (body/etag) is still readable; the failure row
    # records the retry count alongside the preserved body.
    data_a, _, tries_a, etag_a = cache_storage.load(
        None, job_a.get_guid(), config_id=config_a)
    assert data_a == baseline_a
    assert etag_a == '"etag-1"'
    assert tries_a == 1

    # B is completely untouched by A's failure.
    data_b, _, tries_b, etag_b = cache_storage.load(
        None, job_b.get_guid(), config_id=config_b)
    assert data_b == baseline_b
    assert tries_b == 0
    assert etag_b == StatefulHttpHandler.etag

    # A subsequent successful run resets the retry counter and keeps A's scope.
    StatefulHttpHandler.fail_requests = 0
    recovered = process(cache_storage, job_a)
    assert recovered.exception is None
    assert recovered.tries == 0
    data_a, _, tries_a, _ = cache_storage.load(
        None, job_a.get_guid(), config_id=config_a)
    assert data_a == baseline_a
    assert tries_a == 0


def test_history_dedup_and_compared_versions_scoped(http_server, cache_storage):
    job = make_job(http_server, 'grep:alpha', compared_versions=5)

    # First snapshot.
    process(cache_storage, job)
    config_id = get_job_config_id(job)

    # Same body again -> the worker-level unchanged path would not save a new
    # row, but even if saved, get_history_data deduplicates identical data.
    cache_storage.save(job, job.get_guid(), 'alpha', 2000, 0, '"etag-1"',
                       config_id=config_id)
    cache_storage.save(job, job.get_guid(), 'alpha', 3000, 0, '"etag-1"',
                       config_id=config_id)

    history = cache_storage.get_history_data(job.get_guid(), 5,
                                             config_id=config_id)
    assert list(history.keys()) == ['alpha']  # deduplicated

    # A different configuration does not see A's history.
    other = make_job(http_server, 'grep:beta', compared_versions=5)
    process(cache_storage, other)
    other_history = cache_storage.get_history_data(other.get_guid(), 5,
                                                   config_id=get_job_config_id(other))
    assert 'alpha' not in other_history


def test_equivalent_filter_spellings_share_baseline(http_server, cache_storage):
    job_string = make_job(http_server, 'grep:alpha')
    state = process(cache_storage, job_string)
    config_id = state.config_id
    baseline = state.new_data

    job_dict = make_job(http_server, [{'grep': {'re': 'alpha'}}])
    assert get_job_config_id(job_dict) == config_id

    again = process(cache_storage, job_dict)
    assert again.config_id == config_id
    assert again.old_data == baseline  # same baseline, treated as known state


def test_legacy_entries_kept_and_feature_disabled_behavior(http_server, cache_storage):
    guid = None

    # Write a legacy entry as older urlwatch versions would (config_id NULL).
    legacy_job = UrlJob(url='http://127.0.0.1:%d/page' % http_server.server_address[1],
                        filter='grep:alpha')
    cache_storage.save(legacy_job, legacy_job.get_guid(), 'legacy body\n', 100,
                       0, '"etag-legacy"')
    guid = legacy_job.get_guid()

    # Legacy rows remain readable through the unscoped load path.
    data, _, tries, etag = cache_storage.load(None, guid)
    assert data == 'legacy body\n'
    assert etag == '"etag-legacy"'

    # Feature disabled: the job keeps using the single legacy baseline.
    disabled = UrlJob(url=legacy_job.url, filter='grep:alpha')
    state = JobState(cache_storage, disabled)
    assert state.config_id is None
    state.load()
    assert state.old_data == 'legacy body\n'
    assert state.etag == '"etag-legacy"'

    # Enabling the feature starts a fresh baseline; legacy data is not deleted.
    enabled = UrlJob(url=legacy_job.url, filter='grep:alpha',
                     filter_config_baselines=True)
    state = process(cache_storage, enabled)
    assert state.config_id is not None
    assert state.old_data is None  # legacy row is not reused as new baseline
    assert cache_storage.load(None, guid)[0] == 'legacy body\n'
    scoped = cache_storage.load(None, guid, config_id=state.config_id)
    assert scoped[0] == state.new_data


def test_treat_new_as_changed_still_applies(http_server, cache_storage):
    job = make_job(http_server, 'grep:alpha', treat_new_as_changed=True)
    state = process(cache_storage, job)
    # JobState.process() applies the flag before retrieval.
    assert state.old_data == ''
    assert state.timestamp is None
    assert state.new_data is not None


def test_gc_keeps_baselines_of_all_configurations(http_server, cache_storage):
    job_a = make_job(http_server, 'grep:alpha')
    job_b = make_job(http_server, 'grep:beta')
    config_a = process(cache_storage, job_a).config_id
    config_b = process(cache_storage, job_b).config_id

    # Add extra snapshots per configuration (timestamps well after the
    # baseline rows written by process(), which use time.time()).
    base_ts = 2_000_000_000
    for offset in (1, 2):
        cache_storage.save(job_a, job_a.get_guid(), 'alpha%s' % offset,
                           base_ts + offset, 0, None, config_id=config_a)
    for offset in (1, 2):
        cache_storage.save(job_b, job_b.get_guid(), 'beta%s' % offset,
                           base_ts + offset, 0, None, config_id=config_b)

    # Cleaning a GUID trims every configuration partition but keeps the
    # latest snapshot of each one, so switching back stays possible.
    removed = cache_storage.clean(job_a.get_guid(), retain_limit=1)
    assert removed == 4

    # The latest A snapshot remains, and B's latest snapshot is kept as well.
    assert cache_storage.load(None, job_a.get_guid(),
                              config_id=config_a)[0] == 'alpha2'
    assert cache_storage.load(None, job_b.get_guid(),
                              config_id=config_b)[0] == 'beta2'

    # Scoped cleaning only touches the given configuration partition.
    for offset in (3, 4):
        cache_storage.save(job_a, job_a.get_guid(), 'alpha%s' % offset,
                           base_ts + offset, 0, None, config_id=config_a)
    assert cache_storage.clean(job_a.get_guid(), 1, config_id=config_a) == 2
    assert cache_storage.load(None, job_a.get_guid(),
                              config_id=config_a)[0] == 'alpha4'
    assert cache_storage.load(None, job_b.get_guid(),
                              config_id=config_b)[0] == 'beta2'

    # Whole-GUID garbage collection (job removed) still deletes everything.
    cache_storage.gc([], retain_limit=1)
    assert list(cache_storage.get_guids()) == []
