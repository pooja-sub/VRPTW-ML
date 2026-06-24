import sys
import threading
from datetime import datetime


class _StreamToFileAndStd:
    def __init__(self, filename, stream, lock):
        self._file = open(filename, 'a', encoding='utf-8')
        self._stream = stream
        self._lock = lock

    def write(self, data):
        if not data:
            return
        with self._lock:
            try:
                self._file.write(data)
                self._file.flush()
            except Exception:
                pass
            try:
                self._stream.write(data)
            except Exception:
                pass

    def flush(self):
        with self._lock:
            try:
                self._file.flush()
            except Exception:
                pass
            try:
                self._stream.flush()
            except Exception:
                pass

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass


_orig_stdout = None
_orig_stderr = None
_stdout_wrapper = None
_stderr_wrapper = None
_lock = threading.Lock()


def start_logging(logfile_path=None, tee=True):
    """Redirect stdout/stderr to a log file. If tee=True, also keep printing to console.

    logfile_path: path to log file. If None, creates `run_YYYYmmdd_HHMMSS.log` in CWD.
    """
    global _orig_stdout, _orig_stderr, _stdout_wrapper, _stderr_wrapper

    if logfile_path is None:
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        logfile_path = f"run_{now}.log"

    if _orig_stdout is not None:
        # already started
        return logfile_path

    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr

    if tee:
        _stdout_wrapper = _StreamToFileAndStd(logfile_path, _orig_stdout, _lock)
        _stderr_wrapper = _StreamToFileAndStd(logfile_path, _orig_stderr, _lock)
        sys.stdout = _stdout_wrapper
        sys.stderr = _stderr_wrapper
    else:
        # Only file (no console echo)
        class _FileOnly:
            def __init__(self, f):
                self._f = open(f, 'a', encoding='utf-8')

            def write(self, data):
                try:
                    self._f.write(data)
                    self._f.flush()
                except Exception:
                    pass

            def flush(self):
                try:
                    self._f.flush()
                except Exception:
                    pass

            def close(self):
                try:
                    self._f.close()
                except Exception:
                    pass

        sys.stdout = _FileOnly(logfile_path)
        sys.stderr = _FileOnly(logfile_path)

    print(f"[Logging started] Logfile: {logfile_path}")
    return logfile_path


def stop_logging():
    global _orig_stdout, _orig_stderr, _stdout_wrapper, _stderr_wrapper
    if _orig_stdout is None:
        return
    try:
        if _stdout_wrapper:
            _stdout_wrapper.close()
        if _stderr_wrapper:
            _stderr_wrapper.close()
    except Exception:
        pass
    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr
    _orig_stdout = None
    _orig_stderr = None
    _stdout_wrapper = None
    _stderr_wrapper = None
