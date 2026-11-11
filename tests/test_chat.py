import http.client
import json
import threading
import unittest

from tools.chat_server import make_server


class SlowEngine:
    """Deterministic HTTP concurrency/cancellation fixture, not a model substitute."""
    def status(self):
        return dict(device='cpu', gpu_inference=False)

    def prepare(self, payload):
        return payload

    def stream(self, request, cancel):
        yield {'type': 'start'}
        if not cancel.wait(timeout=5):
            raise AssertionError('Test failed to cancel the request')
        yield {'type': 'done', 'finish_reason': 'stopped', 'text': ''}


class ChatHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(SlowEngine(), port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def connect(self):
        return http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=10)

    def post(self, path, payload, **headers):
        conn = self.connect()
        conn.request('POST', path, json.dumps(payload), {'Content-Type': 'application/json', **headers})
        response = conn.getresponse()
        status, body = response.status, json.loads(response.read())
        conn.close()
        return status, body

    def test_status_and_static_routes(self):
        conn = self.connect()
        conn.request('GET', '/api/status')
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertFalse(json.loads(response.read())['gpu_inference'])
        for path in ['/', '/app.js', '/style.css']:
            conn.request('GET', path)
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            self.assertGreater(len(response.read()), 100)
        conn.request('GET', '/../pyproject.toml')
        response = conn.getresponse()
        self.assertEqual(response.status, 404)
        response.read()
        conn.close()

    def test_cross_origin_rejected(self):
        status, body = self.post('/api/stop', {'request_id': 'test'}, Origin='https://example.org')
        self.assertEqual(status, 403)

    def test_busy_stop_and_stream_completion(self):
        conn = self.connect()
        conn.request('POST', '/api/chat', json.dumps({'request_id': 'active'}),
                     {'Content-Type': 'application/json'})
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.readline())['type'], 'start')
        status, _ = self.post('/api/chat', {'request_id': 'second'})
        self.assertEqual(status, 409)
        status, body = self.post('/api/stop', {'request_id': 'wrong'})
        self.assertFalse(body['stopping'])
        status, body = self.post('/api/stop', {'request_id': 'active'})
        self.assertTrue(body['stopping'])
        self.assertEqual(json.loads(response.readline())['finish_reason'], 'stopped')
        self.assertEqual(response.read(), b'')
        conn.close()


if __name__ == '__main__':
    unittest.main()
