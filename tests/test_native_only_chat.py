"""Lightweight optional-backend contracts; no model loading or inference."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.chat_backends import BackendRouter
from tools.native_cpu_chat import NativeCPUChat


class NativeOnlyChatTests(unittest.TestCase):
    def test_server_import_does_not_import_simulator_or_torch(self):
        root=Path(__file__).resolve().parents[1]
        result=subprocess.run([sys.executable,'-c',
            "import sys; import tools.chat_server; assert not any(name in sys.modules for name in ('tools.rtl_model','rtl_model','tools.rtl_chat','rtl_chat','torch','transformers'))"],
            cwd=root,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_native_without_simulator_has_status_and_ignores_rtl_workers(self):
        protocol=SimpleNamespace(tokenizer=None)
        request=SimpleNamespace(max_draft=0)
        with patch.object(NativeCPUChat,'available',return_value=True), patch.object(Path,'is_file',return_value=False):
            router=BackendRouter(protocol=protocol)
            self.assertEqual(router.default_backend(),'native-cpu-f16')
            self.assertEqual(router.simulations,{})
            status=router.status()
            self.assertEqual(status['device'],'cpu')
            self.assertFalse(status['prefix_reuse_enabled'])
            self.assertEqual(status['cached_tokens'],0)
            self.assertFalse(next(x for x in status['backends'] if x['id']=='rtl')['available'])
            with patch('tools.chat_backends.ChatProtocol.prepare',return_value=request) as prepare:
                backend,actual=router.prepare(dict(simulation_workers=999))
                self.assertEqual(backend,'native-cpu-f16');self.assertIs(actual,request)
                self.assertIs(prepare.call_args.args[0],protocol)
            with self.assertRaisesRegex(ValueError,'Unknown or unavailable'):
                router.prepare(dict(backend='rtl'))
            with patch('tools.chat_backends.ChatProtocol.prepare',return_value=SimpleNamespace(max_draft=1)):
                with self.assertRaisesRegex(ValueError,'Suffix drafting'):
                    router.prepare(dict(backend='native-cpu-f16'))
            router.close()

    def test_no_available_backend_has_explicit_error(self):
        with patch.object(NativeCPUChat,'available',return_value=False), patch.object(Path,'is_file',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'No available CPU backend'):
                BackendRouter(protocol=SimpleNamespace(tokenizer=None))


if __name__=='__main__':unittest.main()
