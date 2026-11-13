"""Isolated real chat server for native browser checks, without stable promotion.

Launch only inside the main benchmark guard. The test server uses a separate
lock directory so tests can exercise chat and guard visibility while the live
browser remains blocked by the outer guard. No build/export/download occurs.
"""
import argparse
from pathlib import Path
import signal
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.benchmark_guard import BenchmarkGate
from tools.chat_backends import BackendRouter
from tools.chat_server import make_server
from tools.rtl_chat import RTLChatEngine
from tools.rtl_model import FAST_BINARY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=ROOT/'build/native-cpu/weights.bin')
    parser.add_argument('--model')
    parser.add_argument('--guard-state-dir', type=Path, default=ROOT/'build/native-cpu/browser-guard')
    args = parser.parse_args()
    main_guard = ROOT/'build/benchmark-status'
    if args.guard_state_dir.resolve()==main_guard.resolve():
        parser.error('Isolated server must not use the live server guard directory')
    if not BenchmarkGate(main_guard).status()['running']:
        parser.error('Launch under the main benchmark guard to exclude live browser generation')
    if not FAST_BINARY.is_file() or not args.binary.is_file():
        parser.error('Existing fast simulator metadata asset and explicit native binary are required')
    # Construction prepares tokenizer/layout metadata only. RTL execution starts
    # lazily if someone explicitly requests an RTL backend; this suite never does.
    simulation = RTLChatEngine(args.model,workers=8,binary=FAST_BINARY,
                               lanes=8,block_size=4,prefill_size=8)
    router = BackendRouter(simulation)
    router.native_cpu.binary = args.binary.resolve()
    router.native_cpu.weights = args.weights.resolve()
    if not router.native_cpu.available():
        router.close()
        parser.error('Explicit native binary and complete weight pack must already exist')
    if 'torch' in sys.modules:
        router.close()
        raise RuntimeError('PyTorch unexpectedly loaded into the browser server')
    server = make_server(router,port=8766,benchmark_directory=args.guard_state_dir)
    def stop(signum, frame):
        # HTTPServer.shutdown must execute on a different thread than serve_forever.
        threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    print('Isolated native browser server: http://127.0.0.1:8766',flush=True)
    print(f'Explicit binary: {router.native_cpu.binary}; stable path unchanged',flush=True)
    try:
        server.serve_forever()
    finally:
        with server.service.state_lock:
            if server.service.cancel:
                server.service.cancel.set()
        server.server_close()
        with server.service.generation_lock:
            router.close()


if __name__=='__main__':main()
