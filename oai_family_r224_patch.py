"""Tambalan skrip inferensi nartaa (sel 4 notebook Apex) agar mendukung arm 'sibling r224':
backbone img_size per arm, hash per berkas ckpt, crop tengah (307 px) -> ukuran jaringan (224) sebelum normalisasi, 1 tampilan.
Semua parameter opsional; arm tanpa parameter ini berperilaku PERSIS seperti skrip asli."""


def _sub1(t, o, n):
    assert t.count(o) == 1, f"pola tak unik/tak ada: {o[:70]!r} x{t.count(o)}"
    return t.replace(o, n)


def patch_r224(c4):
    c4 = _sub1(c4, "def build_backbone(arch, pretrained=False):", "def build_backbone(arch, pretrained=False, img_size=None):")
    c4 = _sub1(c4, "kw = dict(pretrained=pretrained, num_classes=0, in_chans=3, img_size=INPUT_RES)",
               "kw = dict(pretrained=pretrained, num_classes=0, in_chans=3, img_size=(img_size or INPUT_RES))")
    c4 = _sub1(c4, "def load_model(pt_path, arch_default, res_default, device, ngpu=1):",
               "def load_model(pt_path, arch_default, res_default, device, ngpu=1, img=None):")
    old_sha = '''    if os.path.basename(pt_path).startswith("raptor_ft_alldata"):   # induk: hash terikat; anak kita: tanpa hash tetap
        assert observed_sha == '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef', "Bound checkpoint bytes differ"'''
    new_sha = '''    _KNOWN = {"raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt": '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef',
              "raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt": '3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6'}
    if os.path.basename(pt_path) in _KNOWN:   # ckpt publik nartaa: hash terikat; anak kita: tanpa hash tetap
        assert observed_sha == _KNOWN[os.path.basename(pt_path)], "Bound checkpoint bytes differ"'''
    c4 = _sub1(c4, old_sha, new_sha)
    c4 = _sub1(c4, 'assert int(ck.get("res", res_default)) == 384, "Expected original 384 checkpoint"',
               'assert int(ck.get("res", res_default)) in (384, 224), "Unexpected checkpoint resolution"')
    c4 = _sub1(c4, "    ck_res = INPUT_RES\n    bb = build_backbone(arch, pretrained=False)\n",
               "    ck_res = (img or INPUT_RES)\n    bb = build_backbone(arch, pretrained=False, img_size=img)\n")
    c4 = _sub1(c4, "def gpu_windows(vol, mask, k, res, dev, norm=NORM, mirror=False):",
               "def gpu_windows(vol, mask, k, res, dev, norm=NORM, mirror=False, crop=None):")
    c4 = _sub1(c4, "    assert tri.shape[-2:] == (IMG, IMG) and res == INPUT_RES\n    lo = (IMG - res) // 2\n    tri = tri[:, :, lo:lo + res, lo:lo + res]",
               "    _c = crop or res\n    assert tri.shape[-2:] == (IMG, IMG) and (crop is not None or res == INPUT_RES)\n    lo = (IMG - _c) // 2\n    tri = tri[:, :, lo:lo + _c, lo:lo + _c]")
    c4 = _sub1(c4, 'model, res = load_model(find_weight_file(arm["file"]), arm["arch"], arm["res"], d)',
               'model, res = load_model(find_weight_file(arm["file"]), arm["arch"], arm["res"], d, img=arm.get("img"))')
    old_loop = '''                    xw = eval_windows(vol, mask, k=K_EVAL, res=res, norm=NORM) if xla is not None else gpu_windows(vol, mask, K_EVAL, res, d)
                    assert xla is None, "Anatomical L/R TTA requires CUDA/fp16"
                    xm = gpu_windows(vol, mask, K_EVAL, res, d, norm=NORM, mirror=True)
                    arm_probs[a][i] = infer_pair_probs(model, xw, xm, d)
                    del xm'''
    new_loop = '''                    _arm = ARMS[a]
                    if _arm.get("single"):    # sibling r224: 1 tampilan, crop tengah -> ukuran jaringan
                        xw = gpu_windows(vol, mask, K_EVAL, res, d, crop=_arm.get("crop"))
                        arm_probs[a][i] = infer_probs(model, xw, d)
                        continue
                    xw = eval_windows(vol, mask, k=K_EVAL, res=res, norm=NORM) if xla is not None else gpu_windows(vol, mask, K_EVAL, res, d)
                    assert xla is None, "Anatomical L/R TTA requires CUDA/fp16"
                    xm = gpu_windows(vol, mask, K_EVAL, res, d, norm=NORM, mirror=True)
                    arm_probs[a][i] = infer_pair_probs(model, xw, xm, d)
                    del xm'''
    c4 = _sub1(c4, old_loop, new_loop)
    compile(c4, "cell4-r224", "exec")
    return c4
