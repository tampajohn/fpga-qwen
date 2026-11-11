#!/usr/bin/env python3
"""Local-only chat server; full behavioral SystemVerilog inference via Verilator."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit

try:
    from .rtl_chat import RTLChatEngine
    from .chat_backends import BackendRouter
    from .rtl_model import FAST_BINARY
except ImportError:
    from rtl_chat import RTLChatEngine
    from chat_backends import BackendRouter
    from rtl_model import FAST_BINARY

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'web'


class ChatService:
    def __init__(self, engine):
        self.engine = engine
        self.generation_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.active_id = None
        self.cancel = None


class ChatHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    @property
    def service(self):
        return self.server.service

    def log_message(self, format, *args):
        # Access metadata only; never log prompt bodies or model output.
        super().log_message(format, *args)

    def reply(self, status, data, content_type='application/json; charset=utf-8'):
        if not isinstance(data, bytes):
            data = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        if self.close_connection:
            self.send_header('Connection', 'close')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/api/status':
            with self.service.state_lock:
                busy = self.service.active_id is not None
            self.reply(200, dict(ready=True, busy=busy, **self.service.engine.status()))
            return
        files = {'/': ('index.html', 'text/html; charset=utf-8'),
                 '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                 '/style.css': ('style.css', 'text/css; charset=utf-8')}
        if path not in files:
            self.reply(404, {'error': 'Not found'})
            return
        name, kind = files[path]
        self.reply(200, (WEB / name).read_bytes(), kind)

    def body(self):
        origin = self.headers.get('Origin')
        if origin and urlsplit(origin).netloc != self.headers.get('Host'):
            raise PermissionError('Cross-origin requests are not allowed')
        if self.headers.get_content_type() != 'application/json':
            raise ValueError('Use application/json')
        length = int(self.headers.get('Content-Length', '0'))
        if not 0 < length <= 65536:
            raise ValueError('Request body must be 1–65,536 bytes')
        body = json.loads(self.rfile.read(length))
        if not isinstance(body, dict):
            raise ValueError('Expected a JSON object')
        return body

    def do_POST(self):
        try:
            payload = self.body()
        except PermissionError as e:
            self.close_connection = True
            self.reply(403, {'error': str(e)})
            return
        except (ValueError, UnicodeError) as e:
            self.close_connection = True
            self.reply(400, {'error': str(e)})
            return
        path = urlsplit(self.path).path
        if path == '/api/stop':
            with self.service.state_lock:
                matches = payload.get('request_id') == self.service.active_id and self.service.cancel is not None
                if matches:
                    self.service.cancel.set()
            self.reply(200, {'stopping': bool(matches)})
            return
        if path != '/api/chat':
            self.reply(404, {'error': 'Not found'})
            return
        request_id = payload.get('request_id')
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 100:
            self.reply(400, {'error': 'Missing or invalid request identifier'})
            return
        if not self.service.generation_lock.acquire(blocking=False):
            self.reply(409, {'error': 'Another reply is running. Stop it or wait for it to finish.'})
            return
        stream = None
        started = False
        cancel = threading.Event()
        try:
            with self.service.state_lock:
                self.service.active_id = request_id
                self.service.cancel = cancel
            request = self.service.engine.prepare(payload)
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Accel-Buffering', 'no')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.close_connection = True
            started = True
            stream = self.service.engine.stream(request, cancel)
            for event in stream:
                self.wfile.write((json.dumps(event)+'\n').encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            cancel.set()
        except Exception as e:
            if started:
                try:
                    self.wfile.write((json.dumps({'type': 'error', 'error': str(e)})+'\n').encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.reply(400 if isinstance(e, ValueError) else 500, {'error': str(e)})
        finally:
            cancel.set()
            if stream:
                stream.close()
            with self.service.state_lock:
                self.service.active_id = None
                self.service.cancel = None
            self.service.generation_lock.release()


def make_server(engine, port=8765):
    server = ThreadingHTTPServer(('127.0.0.1', port), ChatHandler)
    server.daemon_threads = True
    server.service = ChatService(engine)
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model')
    parser.add_argument('--simulator', choices=['fast', 'reference'], default='fast')
    parser.add_argument('--block-size', type=int, choices=[1, 4], default=None,
                        help='Default: 4 for the fast profile, 1 for reference or a custom binary')
    parser.add_argument('--prefill-size', type=int, choices=[1, 4, 8], default=None,
                        help='Default: 8 for fast profile; otherwise verification block size. Custom B8 requires --block-size 4')
    parser.add_argument('--sv-binary', type=Path, help='Trusted local simulator executable override')
    parser.add_argument('--workers', type=int, choices=[0, 1, 2, 4, 8], default=8)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError('Invalid port')
    binary = args.sv_binary or (FAST_BINARY if args.simulator == 'fast' else ROOT / 'build/full/Vqwen_behavioral')
    if not binary.is_file():
        raise RuntimeError('Build the simulator first: make chat (fast) or make chat-reference')
    block_size = args.block_size if args.block_size is not None else (4 if args.simulator == 'fast' and args.sv_binary is None else 1)
    prefill_size = args.prefill_size if args.prefill_size is not None else (
        8 if args.simulator == 'fast' and args.sv_binary is None and block_size == 4 else block_size)
    print('Preparing full SystemVerilog chat; no PyTorch or GPU inference...', flush=True)
    engine = BackendRouter(RTLChatEngine(args.model, workers=args.workers, binary=binary,
                                         lanes=8 if args.simulator == 'fast' else 1, block_size=block_size,
                                         prefill_size=prefill_size))
    server = make_server(engine, args.port)
    print(f'Local chat ready: http://127.0.0.1:{server.server_port}', flush=True)
    print('Default: full behavioral SystemVerilog simulation; not synthesizable or cycle-accurate.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with server.service.state_lock:
            if server.service.cancel:
                server.service.cancel.set()
        server.server_close()
        # Let an active handler observe cancellation before destroying its engine.
        with server.service.generation_lock:
            engine.close()


if __name__ == '__main__':
    main()
