#!/usr/bin/env python3
"""Sambungkan kembali sesi Colab yang catatan lokalnya di-"prune" oleh CLI (kegagalan exec sesaat) padahal VM masih terdaftar
di server (`colab sessions` menampilkan "[?]"). Membangun ulang SessionState dari daftar assignment server + menjalankan ulang
daemon keep-alive. Pemakaian:  ~/.local/share/uv/tools/google-colab-cli/bin/python colab/readopt.py [nama=rsna]"""
import sys
sys.path.insert(0, '/home/pasha/.local/share/uv/tools/google-colab-cli/lib/python3.14/site-packages')
from colab_cli.common import state
from colab_cli.auth import AuthProvider
from colab_cli.state import SessionState
from colab_cli.commands.session import spawn_keep_alive
name = sys.argv[1] if len(sys.argv) > 1 else 'rsna'
state.auth_provider = AuthProvider.OAUTH2
asg = state.client.list_assignments()
if not asg: sys.exit('tidak ada assignment di server (VM sudah dilepas)')
a = asg[0]
s = SessionState(name=name, token=a.runtime_proxy_info.token, url=a.runtime_proxy_info.url, endpoint=a.endpoint, variant='GPU', accelerator=a.accelerator.value)
state.store.add(s)
s.keep_alive_pid = spawn_keep_alive(a.endpoint, name, auth_provider=AuthProvider.OAUTH2, config_path=None)
state.store.add(s); print('readopted', a.endpoint, 'keep-alive', s.keep_alive_pid)
