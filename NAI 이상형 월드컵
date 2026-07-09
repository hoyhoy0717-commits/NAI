"""
NAI 그림체 이상형 월드컵 v1.0
===============================
작가 태그 조합을 무작위 / 유전 알고리즘으로 생성하고, 조합이 만들어지는
즉시 NovelAI API로 실제 이미지를 생성합니다. 생성된 이미지로 Elo 기반
이상형 월드컵을 진행해 내 취향에 맞는 그림체 조합을 찾아냅니다.


"""

import io
import json
import os
import random
import re
import ssl
import subprocess
import sys
import http.client
import threading
import time
import uuid
import zipfile
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from datetime import datetime
from queue import Queue, Empty
from fractions import Fraction

# ═══════════════════════════════════════════════
#  NAI API  (nai_generator.py의 검증된 호출 로직 재사용)
# ═══════════════════════════════════════════════

NAI_GEN_HOST = "image.novelai.net"
NAI_GEN_PATH = "/ai/generate-image"

MODELS = [
    "nai-diffusion-4-5-full",
    "nai-diffusion-4-5-curated",
    "nai-diffusion-4-full",
]

SIZE_PRESETS = {
    "세로 832x1216": (832, 1216),
    "정방형 1024x1024": (1024, 1024),
    "가로 1216x832": (1216, 832),
}

SAMPLERS = [
    "k_euler_ancestral",
    "k_euler",
    "k_dpmpp_2s_ancestral",
    "k_dpmpp_2m",
    "k_dpmpp_sde",
]

DEFAULT_ADVANCED_PARAMS = {
    "cfg_rescale": 0.5,
    "noise_schedule": "karras",
    "ucPreset": 0,
    "qualityToggle": True,
    "dynamic_thresholding": False,
    "dynamic_thresholding_percentile": 0.999,
    "dynamic_thresholding_mimic_scale": 10,
    "legacy": False,
    "legacy_v3_extend": False,
    "sm": False,
    "sm_dyn": False,
    "skip_cfg_above_sigma": 58,
    "skip_cfg_below_sigma": 0,
    "deliberate_euler_ancestral_bug": False,
    "prefer_brownian": True,
    "cfg_sched_eligibility": "enable_for_post_summer_samplers",
    "explike_fine_detail": False,
    "minimize_sigma_inf": False,
    "uncond_per_vibe": True,
    "wonky_vibe_correlation": True,
}

DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, "
    "extra digit, fewer digits, cropped, worst quality, low quality, "
    "normal quality, jpeg artifacts, signature, watermark, blurry, "
    "artistic error, bad proportions, logo, artist logo, comic, manga panel, "
    "panel, speech bubble, cover art, magazine cover, collage, text focus, "
    "english text, japanese text, multiple views"
)

DEFAULT_SUBJECT_PROMPT = "1girl, solo, standing, simple background, cowboy shot, nude, tojo nozomi"
DEFAULT_SUBJECT_WEIGHT = 2.0


def nai_headers(api_key: str) -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": "https://novelai.net/",
        "Origin": "https://novelai.net",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }


def extract_png_from_zip(data: bytes) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for name in archive.namelist():
                if name.lower().endswith(".png"):
                    return archive.read(name)
    except zipfile.BadZipFile:
        pass
    if data[:4] == b"\x89PNG":
        return data
    raise ValueError("응답에서 PNG를 추출하지 못했습니다.")


def generate_style_image(
    api_key: str,
    subject_prompt: str,
    style_str: str,
    negative_prompt: str,
    model: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    sampler: str,
    seed: int,
    subject_weight: float = DEFAULT_SUBJECT_WEIGHT,
) -> bytes:
    """작가 조합(style_str)과 테스트 주제(subject_prompt)를 합쳐 이미지 1장을 생성합니다.

    작가 태그는 개별적으로 1.5~2.0까지 가중치를 받는 경우가 많아서, 가중치가 없는
    주제/의상 프롬프트가 상대적으로 묻혀 캐릭터가 지정한 옷을 안 입거나 특정 작가의
    시그니처 요소(로고, 텍스트, 만화 컷 등)가 그대로 튀어나오는 경우가 있습니다.
    이를 막기 위해 주제 프롬프트에도 자체 가중치를 줘서 작가 태그와 균형을 맞춥니다.
    """
    subject = subject_prompt.strip()
    style = style_str.strip()
    if subject:
        subject_part = f"{subject_weight:.1f}::{subject}::"
    else:
        subject_part = ""
    base_caption = ", ".join(p for p in [subject_part, style] if p)
    if not base_caption:
        raise ValueError("프롬프트가 비어 있습니다.")

    parameters = {
        "params_version": 3,
        "width": width,
        "height": height,
        "scale": cfg,
        "steps": steps,
        "seed": seed,
        "n_samples": 1,
        "add_original_image": False,
        "characterPrompts": [],
        "sampler": sampler,
        "v4_prompt": {
            "caption": {"base_caption": base_caption, "char_captions": []},
            "use_coords": False,
            "use_order": True,
        },
        "v4_negative_prompt": {
            "caption": {"base_caption": negative_prompt, "char_captions": []}
        },
        "negative_prompt": negative_prompt,
    }
    parameters.update(DEFAULT_ADVANCED_PARAMS)
    parameters["sampler"] = sampler  # DEFAULT_ADVANCED_PARAMS 이후 다시 확정

    payload = {
        "input": base_caption,
        "model": model,
        "action": "generate",
        "parameters": parameters,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    ctx = ssl.create_default_context()
    conn = http.client.HTTPSConnection(NAI_GEN_HOST, context=ctx, timeout=120)
    try:
        conn.request("POST", NAI_GEN_PATH, body=body, headers=nai_headers(api_key))
        resp = conn.getresponse()
        data = resp.read()
        status, reason = resp.status, resp.reason
    finally:
        conn.close()

    if status != 200:
        err_body = data[:500].decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {status} {reason}: {err_body}")

    return extract_png_from_zip(data)


# ═══════════════════════════════════════════════
#  조합 / Elo / 유전 알고리즘 유틸리티
# ═══════════════════════════════════════════════

ARTIST_TAG_RE = re.compile(r"artist:[^:]+")

DEFAULT_ARTIST_TAGS = [
    "artist:for-u", "artist:mikan03 26", "artist:hwansang",
    "artist:momoko (momopoco)", "artist:ningen mame", "artist:omutatsu",
    "artist:rurudo", "artist:necomi", "artist:chamooi", "artist:cura",
    "artist:shnva", "artist:shigure ui", "artist:zain", "artist:ke-ta",
    "artist:onineko", "artist:wanke", "artist:muchi maro", "artist:mignon",
    "artist:quasarcake", "artist:supernew", "artist:channel (caststation)",
    "artist:healthyman", "artist:torino aqua",
]


def default_artists():
    return [
        {"tag": t, "elo": 1000, "count": 0, "arena_matches": 0, "arena_wins": 0}
        for t in DEFAULT_ARTIST_TAGS
    ]


def sanitize_tag(raw: str):
    clean = raw.strip()
    if not clean:
        return None
    return clean if clean.startswith("artist:") else f"artist:{clean}"


def _looks_like_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def extract_artist_names(raw: str) -> list:
    """대량 등록 텍스트에서 작가 이름만 뽑아냅니다.
    'rurudo' 처럼 한 줄에 하나씩 적은 단순 목록도, 남이 쓰던
    '0.8::artist_name::, 1.2::other::' 같은 NAI 가중치 배합 문자열을
    그대로 복사해 붙여넣은 경우도 동일하게 처리합니다. 가중치 숫자,
    빈 조각, 군더더기 콜론은 자동으로 걸러집니다."""
    names = []
    for line in raw.split("\n"):
        for chunk in line.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            # "0.8::artist:xxx::" / "artist:xxx" / "::artist:xxx::" 등을 모두 지원
            for token in chunk.split("::"):
                token = token.strip()
                if not token or _looks_like_number(token):
                    continue
                names.append(token)
    return names


def artists_from_style(style_str: str) -> list:
    return [t.strip() for t in ARTIST_TAG_RE.findall(style_str)]


def calc_elo(rating_a, rating_b, a_won, k=32):
    ea = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    eb = 1 / (1 + 10 ** ((rating_a - rating_b) / 400))
    new_a = round(rating_a + k * ((1 if a_won else 0) - ea))
    new_b = round(rating_b + k * ((0 if a_won else 1) - eb))
    return new_a, new_b


def parse_style_combo(style_str: str) -> list:
    """'0.9::artist:x::, 1.1::artist:y::' -> [(weight, tag), ...]"""
    items = []
    for chunk in style_str.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split("::")
        if len(parts) >= 2:
            try:
                w = float(parts[0])
            except ValueError:
                w = 1.0
            tag = parts[1].strip()
            if tag:
                items.append((w, tag))
    return items


def build_style_str(tags: list) -> str:
    parts = []
    for tag in tags:
        weight = round(random.randint(8, 18) / 10, 1)
        space = " " if tag[-1:].isdigit() else ""
        parts.append(f"{weight:.1f}::{tag}{space}::")
    return ", ".join(parts)


def crossover_and_mutate(style_a, style_b, all_tags, mutation_rate=0.05, min_tags=5, max_tags=10):
    combo_a = parse_style_combo(style_a)
    combo_b = parse_style_combo(style_b)
    pool_tags = list({tag for _, tag in combo_a + combo_b})
    random.shuffle(pool_tags)

    target = random.randint(min_tags, max_tags)
    child_tags = pool_tags[:target]

    if len(child_tags) < target and all_tags:
        remaining = [t for t in all_tags if t not in child_tags]
        random.shuffle(remaining)
        child_tags += remaining[: target - len(child_tags)]

    final_tags = []
    for tag in child_tags:
        if all_tags and random.random() < mutation_rate:
            tag = random.choice(all_tags)
        final_tags.append(tag)

    final_tags = list(dict.fromkeys(final_tags))[:max_tags]
    return build_style_str(final_tags)


def mutate_style_pairs(seed_pairs, all_tags, weight_jitter=0.3, retag_rate=0.15,
                        add_chance=0.3, remove_chance=0.2, min_tags=5, max_tags=10):
    """기존 (weight, tag) 조합 하나를 바탕으로 배합비를 흔들고 작가를 교체/추가/제거해
    '유사 그림체' 변형 하나를 만듭니다."""
    pairs = list(seed_pairs)
    existing_tags = {t for _, t in pairs}

    mutated = []
    for w, tag in pairs:
        new_w = min(2.0, max(0.5, round(w + random.uniform(-weight_jitter, weight_jitter), 1)))
        new_tag = tag
        if all_tags and random.random() < retag_rate:
            replacement_pool = [t for t in all_tags if t not in existing_tags]
            if replacement_pool:
                new_tag = random.choice(replacement_pool)
                existing_tags.discard(tag)
                existing_tags.add(new_tag)
        mutated.append((new_w, new_tag))

    if len(mutated) > min_tags and random.random() < remove_chance:
        idx = random.randrange(len(mutated))
        removed_tag = mutated.pop(idx)[1]
        existing_tags.discard(removed_tag)

    if len(mutated) < max_tags and all_tags and random.random() < add_chance:
        candidates = [t for t in all_tags if t not in existing_tags]
        if candidates:
            new_tag = random.choice(candidates)
            mutated.append((round(random.uniform(0.8, 1.3), 1), new_tag))
            existing_tags.add(new_tag)

    return mutated


# ═══════════════════════════════════════════════
#  로컬 저장소 (JSON + 이미지 폴더)
# ═══════════════════════════════════════════════

DATA_DIR = Path(__file__).resolve().parent / "nai_style_arena_data"
IMG_DIR = DATA_DIR / "images"
STATE_FILE = DATA_DIR / "state.json"


def ensure_dirs():
    IMG_DIR.mkdir(parents=True, exist_ok=True)


def load_state():
    ensure_dirs()
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            data.setdefault("artists", default_artists())
            data.setdefault("combinations", [])
            data.setdefault("history", [])
            return data
        except Exception:
            pass
    return {"artists": default_artists(), "combinations": [], "history": []}


def open_in_viewer(path: Path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # noqa
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


def load_thumbnail(path: Path, max_w=560):
    """이미지를 최대 max_w 너비로 표시합니다. Tk PhotoImage는 정수 배율만 지원하므로
    zoom(확대)과 subsample(축소)을 조합해 원하는 크기에 가깝게 맞춥니다."""
    img = tk.PhotoImage(file=str(path))
    w = img.width()
    if w <= max_w:
        return img
    ratio = Fraction(max_w, w).limit_denominator(4)
    zoom_f, sub_f = ratio.numerator, ratio.denominator
    if zoom_f > 1:
        img = img.zoom(zoom_f, zoom_f)
    if sub_f > 1:
        img = img.subsample(sub_f, sub_f)
    return img


# ═══════════════════════════════════════════════
#  메인 애플리케이션
# ═══════════════════════════════════════════════

class StyleArenaApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("NAI 그림체 이상형 월드컵 v1.0")
        root.geometry("1360x920")

        self.C = dict(
            BG="#1e1e2e", BG2="#181825", BG3="#313244",
            ACCENT="#cba6f7", FG="#cdd6f4", FG2="#a6adc8",
            GREEN="#a6e3a1", YELLOW="#f9e2af", RED="#f38ba8",
            BLUE="#89b4fa", DIV="#45475a",
        )
        root.configure(bg=self.C["BG"])
        self._setup_style()

        state = load_state()
        self.artists = state["artists"]
        self.combinations = state["combinations"]
        self.history = state["history"]
        self.current_match = None
        self.last_batch_id = None
        self._image_cache = {}
        self._running = False
        self._stop_event = threading.Event()
        self._log_queue = Queue()

        self.api_key_var = tk.StringVar()
        self.model_var = tk.StringVar(value=MODELS[0])
        self.size_var = tk.StringVar(value=list(SIZE_PRESETS.keys())[0])
        self.sampler_var = tk.StringVar(value=SAMPLERS[0])
        self.steps_var = tk.IntVar(value=28)
        self.cfg_var = tk.DoubleVar(value=8.0)
        self.delay_var = tk.DoubleVar(value=1.0)
        self.subject_var = tk.StringVar(value=DEFAULT_SUBJECT_PROMPT)
        self.negative_var = tk.StringVar(value=DEFAULT_NEGATIVE)
        self.subject_weight_var = tk.DoubleVar(value=DEFAULT_SUBJECT_WEIGHT)

        self._build_layout()
        self._poll_log_queue()

    # ── 스타일 ─────────────────────────────────
    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        C = self.C
        style.configure("TNotebook", background=C["BG"], borderwidth=0)
        style.configure("TNotebook.Tab", background=C["BG2"], foreground=C["FG2"],
                         padding=(14, 8))
        style.map("TNotebook.Tab", background=[("selected", C["ACCENT"])],
                  foreground=[("selected", C["BG"])])
        style.configure("Treeview", background=C["BG2"], fieldbackground=C["BG2"],
                         foreground=C["FG"], rowheight=26, borderwidth=0)
        style.configure("Treeview.Heading", background=C["BG3"], foreground=C["FG"])
        style.map("Treeview", background=[("selected", C["ACCENT"])],
                  foreground=[("selected", C["BG"])])

    # ── 저장 ───────────────────────────────────
    def _save(self):
        ensure_dirs()
        data = {"artists": self.artists, "combinations": self.combinations,
                "history": self.history}
        STATE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 레이아웃 ───────────────────────────────
    def _build_layout(self):
        C = self.C
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 0))
        self.notebook.bind("<<NotebookTabChanged>>", lambda e: self._on_tab_changed())

        self.tab_settings = tk.Frame(self.notebook, bg=C["BG"])
        self.tab_pool = tk.Frame(self.notebook, bg=C["BG"])
        self.tab_generate = tk.Frame(self.notebook, bg=C["BG"])
        self.tab_arena = tk.Frame(self.notebook, bg=C["BG"])
        self.tab_favorites = tk.Frame(self.notebook, bg=C["BG"])
        self.tab_improve = tk.Frame(self.notebook, bg=C["BG"])
        self.tab_evolution = tk.Frame(self.notebook, bg=C["BG"])
        self.tab_dashboard = tk.Frame(self.notebook, bg=C["BG"])

        self.notebook.add(self.tab_settings, text="⚙ 설정")
        self.notebook.add(self.tab_pool, text="🧑‍🎨 작가 명단")
        self.notebook.add(self.tab_generate, text="🎲 조합 생성")
        self.notebook.add(self.tab_arena, text="⚔ 이상형 월드컵")
        self.notebook.add(self.tab_favorites, text="🌟 즐겨찾기")
        self.notebook.add(self.tab_improve, text="🔧 그림체 개선")
        self.notebook.add(self.tab_evolution, text="🧬 조합 진화")
        self.notebook.add(self.tab_dashboard, text="📊 통계")

        self._build_settings_tab()
        self._build_pool_tab()
        self._build_generate_tab()
        self._build_arena_tab()
        self._build_favorites_tab()
        self._build_improve_tab()
        self._build_evolution_tab()
        self._build_dashboard_tab()

        # 하단 공통 로그 / 진행률 영역
        bottom = tk.Frame(self.root, bg=C["BG"])
        bottom.pack(fill=tk.X, padx=8, pady=8)

        self.status_var = tk.StringVar(value="대기 중")
        tk.Label(bottom, textvariable=self.status_var, bg=C["BG"], fg=C["FG2"],
                  font=("Consolas", 9)).pack(anchor="w")

        prog_row = tk.Frame(bottom, bg=C["BG"])
        prog_row.pack(fill=tk.X, pady=(4, 4))
        self.progress = ttk.Progressbar(prog_row, mode="determinate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.stop_btn = tk.Button(prog_row, text="⛔ 중단", command=self._stop_bg,
                                    state=tk.DISABLED, bg=C["RED"], fg=C["BG"],
                                    relief=tk.FLAT, padx=10)
        self.stop_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.log_widget = tk.Text(bottom, height=8, bg=C["BG2"], fg=C["FG"],
                                    insertbackground=C["FG"], font=("Consolas", 9))
        self.log_widget.pack(fill=tk.X, pady=(4, 0))
        self.log_widget.tag_config("err", foreground=C["RED"])
        self.log_widget.tag_config("ok", foreground=C["GREEN"])
        self.log_widget.tag_config("warn", foreground=C["YELLOW"])
        self.log_widget.tag_config("dim", foreground=C["FG2"])

        self._refresh_pool()
        self._refresh_dashboard()

    # ── 로그 / 진행률 / 스레드 유틸 ─────────────
    def _log(self, msg, tag=None):
        self._log_queue.put((msg, tag))

    def _poll_log_queue(self):
        try:
            while True:
                msg, tag = self._log_queue.get_nowait()
                self.log_widget.insert(tk.END, msg, tag or "")
                self.log_widget.see(tk.END)
        except Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    def _set_status(self, text):
        self.root.after(0, lambda: self.status_var.set(text))

    def _set_progress(self, pct):
        self.root.after(0, lambda: self.progress.configure(value=pct))

    def _stop_bg(self):
        self._stop_event.set()

    def _run_bg(self, fn, *args):
        if self._running:
            messagebox.showwarning("알림", "이미 다른 작업이 진행 중입니다.")
            return
        if not self.api_key_var.get().strip():
            messagebox.showwarning("알림", "설정 탭에서 NAI API 키를 먼저 입력해 주세요.")
            return
        self._running = True
        self._stop_event.clear()
        self.stop_btn.configure(state=tk.NORMAL)
        threading.Thread(target=self._bg_wrapper, args=(fn, args), daemon=True).start()

    def _bg_wrapper(self, fn, args):
        try:
            fn(*args)
        except Exception as e:
            self._log(f"오류: {e}\n", "err")
        finally:
            self._running = False
            self.root.after(0, lambda: self.stop_btn.configure(state=tk.DISABLED))

    def _on_tab_changed(self):
        current = self.notebook.select()
        widget = self.root.nametowidget(current)
        if widget is self.tab_arena:
            self._refresh_arena()
        elif widget is self.tab_favorites:
            self._refresh_favorites()
        elif widget is self.tab_improve:
            self._refresh_improve_combo_list()
        elif widget is self.tab_dashboard:
            self._refresh_dashboard()
        elif widget is self.tab_pool:
            self._refresh_pool()
        elif widget is self.tab_evolution:
            self._refresh_evolution_history()

    # ═══════════════════════════════════════
    #  탭 1: 설정
    # ═══════════════════════════════════════
    def _build_settings_tab(self):
        C = self.C
        f = tk.Frame(self.tab_settings, bg=C["BG"], padx=16, pady=16)
        f.pack(fill=tk.BOTH, expand=True)

        def row(label, factory):
            r = tk.Frame(f, bg=C["BG"])
            r.pack(fill=tk.X, pady=5)
            tk.Label(r, text=label, bg=C["BG"], fg=C["FG2"], width=18, anchor="w",
                     font=("Consolas", 10)).pack(side=tk.LEFT)
            widget = factory(r)
            widget.pack(side=tk.LEFT, fill=tk.X, expand=True)
            return widget

        row("NAI API 키", lambda p: tk.Entry(p, textvariable=self.api_key_var, show="*",
                                               bg=C["BG2"], fg=C["FG"], insertbackground=C["FG"]))
        tk.Label(f, text="※ 보안을 위해 API 키는 저장하지 않습니다. 매 실행마다 다시 입력해 주세요.",
                  bg=C["BG"], fg=C["FG2"], font=("Consolas", 8)).pack(anchor="w", padx=(148, 0))

        row("모델", lambda p: ttk.Combobox(p, textvariable=self.model_var, values=MODELS, state="readonly"))
        row("이미지 크기", lambda p: ttk.Combobox(p, textvariable=self.size_var,
                                              values=list(SIZE_PRESETS.keys()), state="readonly"))
        row("샘플러", lambda p: ttk.Combobox(p, textvariable=self.sampler_var, values=SAMPLERS, state="readonly"))
        row("Steps", lambda p: tk.Spinbox(p, from_=1, to=50, textvariable=self.steps_var, width=8,
                                            bg=C["BG2"], fg=C["FG"]))
        row("CFG(Scale)", lambda p: tk.Spinbox(p, from_=1, to=15, increment=0.5, textvariable=self.cfg_var,
                                                  width=8, bg=C["BG2"], fg=C["FG"]))
        row("생성 간 딜레이(초)", lambda p: tk.Spinbox(p, from_=0, to=30, increment=0.5, textvariable=self.delay_var,
                                                    width=8, bg=C["BG2"], fg=C["FG"]))

        tk.Label(f, text="테스트 기본 프롬프트 (모든 조합에 공통으로 붙는 인물/구도/의상 등)",
                  bg=C["BG"], fg=C["FG2"], font=("Consolas", 9)).pack(anchor="w", pady=(12, 2))
        self.subject_box = tk.Text(f, height=3, bg=C["BG2"], fg=C["FG"], insertbackground=C["FG"])
        self.subject_box.insert("1.0", DEFAULT_SUBJECT_PROMPT)
        self.subject_box.pack(fill=tk.X)

        weight_row = tk.Frame(f, bg=C["BG"])
        weight_row.pack(fill=tk.X, pady=(6, 0))
        tk.Label(weight_row, text="주제 프롬프트 강조 가중치", bg=C["BG"], fg=C["FG2"], width=18, anchor="w",
                  font=("Consolas", 10)).pack(side=tk.LEFT)
        tk.Spinbox(weight_row, from_=1.0, to=2.5, increment=0.1, textvariable=self.subject_weight_var,
                    width=8, bg=C["BG2"], fg=C["FG"]).pack(side=tk.LEFT)
        tk.Label(f, text=("작가 태그는 개별적으로 1.5~2.0까지 가중치를 받는 경우가 많아서, 가중치가 없는 "
                            "의상/포즈 지시가 묻혀 캐릭터가 지정한 옷을 안 입거나 특정 작가의 로고·텍스트·만화 컷이 "
                            "그대로 튀어나올 수 있습니다. 이 값을 올리면 위 테스트 프롬프트가 작가 태그에 밀리지 않도록 "
                            "강조됩니다 (기본 1.4, 문제가 계속되면 1.6~1.8까지 올려보세요)."),
                  bg=C["BG"], fg=C["FG2"], font=("Consolas", 8), wraplength=1050, justify="left"
                  ).pack(anchor="w", pady=(2, 0))

        tk.Label(f, text="네거티브 프롬프트", bg=C["BG"], fg=C["FG2"],
                  font=("Consolas", 9)).pack(anchor="w", pady=(12, 2))
        self.negative_box = tk.Text(f, height=3, bg=C["BG2"], fg=C["FG"], insertbackground=C["FG"])
        self.negative_box.insert("1.0", DEFAULT_NEGATIVE)
        self.negative_box.pack(fill=tk.X)

        tk.Label(f, text=f"이미지/데이터 저장 위치: {DATA_DIR}", bg=C["BG"], fg=C["FG2"],
                  font=("Consolas", 8)).pack(anchor="w", pady=(14, 0))

    def _current_gen_settings(self):
        w, h = SIZE_PRESETS[self.size_var.get()]
        return dict(
            api_key=self.api_key_var.get().strip(),
            model=self.model_var.get(),
            width=w, height=h,
            steps=self.steps_var.get(),
            cfg=self.cfg_var.get(),
            sampler=self.sampler_var.get(),
            delay=self.delay_var.get(),
            subject=self.subject_box.get("1.0", tk.END).strip(),
            negative=self.negative_box.get("1.0", tk.END).strip(),
            subject_weight=self.subject_weight_var.get(),
        )

    # ═══════════════════════════════════════
    #  탭 2: 작가 명단
    # ═══════════════════════════════════════
    def _build_pool_tab(self):
        C = self.C
        f = tk.Frame(self.tab_pool, bg=C["BG"], padx=16, pady=16)
        f.pack(fill=tk.BOTH, expand=True)

        top = tk.Frame(f, bg=C["BG"])
        top.pack(fill=tk.X)
        tk.Label(top, text="대량 등록 (줄바꿈/쉼표 목록은 물론, '0.8::artist:xxx::, 1.2::...' 같은 NAI 배합 프롬프트를 그대로 붙여넣어도 작가 이름만 자동으로 추출됩니다)",
                  bg=C["BG"], fg=C["FG2"], wraplength=1000, justify="left").pack(anchor="w")
        self.bulk_box = tk.Text(top, height=4, bg=C["BG2"], fg=C["FG"], insertbackground=C["FG"])
        self.bulk_box.pack(fill=tk.X, pady=(2, 4))
        tk.Button(top, text="🚀 명단 등록", command=self._add_bulk_artists,
                   bg=C["ACCENT"], fg=C["BG"], relief=tk.FLAT).pack(anchor="e")

        single_row = tk.Frame(f, bg=C["BG"])
        single_row.pack(fill=tk.X, pady=8)
        self.single_artist_var = tk.StringVar()
        tk.Entry(single_row, textvariable=self.single_artist_var, bg=C["BG2"], fg=C["FG"],
                  insertbackground=C["FG"]).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(single_row, text="➕ 추가", command=self._add_single_artist,
                   bg=C["ACCENT"], fg=C["BG"], relief=tk.FLAT).pack(side=tk.LEFT, padx=6)
        tk.Button(single_row, text="🚨 전체 초기화", command=self._full_reset,
                   bg=C["RED"], fg=C["BG"], relief=tk.FLAT).pack(side=tk.RIGHT)

        self.pool_tree = ttk.Treeview(f, columns=("elo", "count", "winrate"), show="tree headings", height=16)
        self.pool_tree.heading("#0", text="작가 태그")
        self.pool_tree.heading("elo", text="선호도")
        self.pool_tree.heading("count", text="검출수")
        self.pool_tree.heading("winrate", text="승률")
        self.pool_tree.column("#0", width=320)
        self.pool_tree.column("elo", width=90, anchor="center")
        self.pool_tree.column("count", width=90, anchor="center")
        self.pool_tree.column("winrate", width=110, anchor="center")
        self.pool_tree.pack(fill=tk.BOTH, expand=True, pady=(8, 4))

        tk.Button(f, text="선택 작가 제거", command=self._remove_selected_artist,
                   bg=C["RED"], fg=C["BG"], relief=tk.FLAT).pack(anchor="e")

    def _add_bulk_artists(self):
        raw = self.bulk_box.get("1.0", tk.END)
        names = extract_artist_names(raw)
        existing = {a["tag"] for a in self.artists}
        added = 0
        skipped_dupes = 0
        for name in names:
            tag = sanitize_tag(name)
            if not tag:
                continue
            if tag in existing:
                skipped_dupes += 1
                continue
            self.artists.append({"tag": tag, "elo": 1000, "count": 0,
                                   "arena_matches": 0, "arena_wins": 0})
            existing.add(tag)
            added += 1
        self.bulk_box.delete("1.0", tk.END)
        self._save()
        self._refresh_pool()
        msg = f"{added}명의 작가가 등록되었습니다."
        if skipped_dupes:
            msg += f" (이미 있던 {skipped_dupes}명은 건너뜀)"
        messagebox.showinfo("완료", msg)

    def _add_single_artist(self):
        tag = sanitize_tag(self.single_artist_var.get())
        if not tag:
            return
        if any(a["tag"] == tag for a in self.artists):
            messagebox.showinfo("알림", "이미 등록된 작가입니다.")
            return
        self.artists.append({"tag": tag, "elo": 1000, "count": 0,
                               "arena_matches": 0, "arena_wins": 0})
        self.single_artist_var.set("")
        self._save()
        self._refresh_pool()

    def _remove_selected_artist(self):
        sel = self.pool_tree.selection()
        if not sel:
            return
        tag = sel[0]
        self.artists = [a for a in self.artists if a["tag"] != tag]
        self._save()
        self._refresh_pool()

    def _full_reset(self):
        if not messagebox.askyesno("경고", "모든 작가/조합/기록을 초기화하시겠습니까?\n(생성된 이미지 파일은 삭제되지 않습니다)"):
            return
        self.artists = default_artists()
        self.combinations = []
        self.history = []
        self.current_match = None
        self.last_batch_id = None
        self._save()
        self._refresh_pool()
        self._refresh_dashboard()
        messagebox.showinfo("완료", "초기화되었습니다.")

    def _refresh_pool(self):
        self.pool_tree.delete(*self.pool_tree.get_children())
        for a in sorted(self.artists, key=lambda x: -x["elo"]):
            wr = (a["arena_wins"] / a["arena_matches"] * 100) if a["arena_matches"] else 0
            self.pool_tree.insert("", "end", iid=a["tag"], text=a["tag"],
                                    values=(a["elo"], a["count"], f"{wr:.1f}%"))

    # ═══════════════════════════════════════
    #  탭 3: 랜덤 조합 생성 (즉시 이미지 생성)
    # ═══════════════════════════════════════
    def _build_generate_tab(self):
        C = self.C
        f = tk.Frame(self.tab_generate, bg=C["BG"], padx=16, pady=16)
        f.pack(fill=tk.BOTH, expand=True)

        row = tk.Frame(f, bg=C["BG"])
        row.pack(fill=tk.X)

        self.gen_count_var = tk.IntVar(value=5)
        self.gen_min_var = tk.IntVar(value=5)
        self.gen_max_var = tk.IntVar(value=10)

        def field(label, var, width=8):
            tk.Label(row, text=label, bg=C["BG"], fg=C["FG2"]).pack(side=tk.LEFT, padx=(0, 4))
            tk.Spinbox(row, from_=1, to=999, textvariable=var, width=width,
                        bg=C["BG2"], fg=C["FG"]).pack(side=tk.LEFT, padx=(0, 14))

        field("생성 개수", self.gen_count_var)
        field("조합당 작가 수 최소", self.gen_min_var)
        field("최대", self.gen_max_var)

        tk.Button(row, text="🎲 랜덤 조합 생성 + NAI 이미지 생성 시작",
                   command=self._start_random_generation,
                   bg=C["ACCENT"], fg=C["BG"], relief=tk.FLAT, padx=10).pack(side=tk.LEFT)

        tk.Label(f, text=("각 조합이 만들어지는 즉시 설정 탭의 프롬프트/모델 설정으로 NAI에 요청을 보내 "
                            "이미지 1장을 생성하고 저장합니다. API 사용량(Anlas)이 소모되니 개수에 유의하세요."),
                  bg=C["BG"], fg=C["FG2"], font=("Consolas", 9), wraplength=1000, justify="left"
                  ).pack(anchor="w", pady=(10, 0))

        self.gen_count_label = tk.Label(f, text="", bg=C["BG"], fg=C["FG2"])
        self.gen_count_label.pack(anchor="w", pady=(10, 0))
        self._update_gen_count_label()

    def _update_gen_count_label(self):
        self.gen_count_label.configure(
            text=f"현재 저장된 조합 수: {len(self.combinations)}개")

    def _start_random_generation(self):
        count = self.gen_count_var.get()
        min_t, max_t = self.gen_min_var.get(), self.gen_max_var.get()
        if min_t > max_t:
            messagebox.showwarning("알림", "최소 작가 수가 최대보다 클 수 없습니다.")
            return
        if len(self.artists) < min_t:
            messagebox.showwarning("알림", f"작가를 최소 {min_t}명 이상 등록해 주세요.")
            return
        settings = self._current_gen_settings()
        self._run_bg(self._do_random_generation, count, min_t, max_t, settings)

    def _do_random_generation(self, count, min_t, max_t, settings):
        seen = {c["style"] for c in self.combinations}
        made = 0
        loops = 0
        while made < count and loops < count * 20:
            if self._stop_event.is_set():
                self._log("⛔ 중단됨\n", "err")
                break
            loops += 1
            size = random.randint(min_t, min(max_t, len(self.artists)))
            tags = [a["tag"] for a in random.sample(self.artists, size)]
            style = build_style_str(tags)
            if style in seen:
                continue
            seen.add(style)

            combo_id = uuid.uuid4().hex[:10]
            seed = random.randint(0, 4_294_967_295)
            self._set_status(f"조합 {made + 1}/{count} 생성 중 (NAI 이미지 요청)...")
            self._log(f"┌ 조합 {made + 1}/{count}: {style[:80]}...\n", "dim")
            try:
                png = generate_style_image(
                    settings["api_key"], settings["subject"], style, settings["negative"],
                    settings["model"], settings["width"], settings["height"],
                    settings["steps"], settings["cfg"], settings["sampler"], seed,
                    subject_weight=settings["subject_weight"],
                )
                img_name = f"{combo_id}.png"
                (IMG_DIR / img_name).write_bytes(png)
                self.combinations.append({
                    "id": combo_id, "style": style, "elo": 1000, "matches": 0, "wins": 0,
                    "is_favorite": False, "is_locked": False, "memo": "",
                    "generation": 1, "image_file": img_name, "seed": seed,
                })
                for tag in tags:
                    for a in self.artists:
                        if a["tag"] == tag:
                            a["count"] += 1
                made += 1
                self._save()
                self._log(f"└ ✓ 생성 완료 ({img_name})\n\n", "ok")
            except Exception as e:
                self._log(f"└ ✗ 실패: {e}\n\n", "err")

            self._set_progress(made / count * 100)
            self.root.after(0, self._update_gen_count_label)
            if made < count and not self._stop_event.is_set():
                time.sleep(settings["delay"])

        self._set_status(f"완료 — 조합 {made}개 생성")
        self._set_progress(100)
        self.root.after(0, self._refresh_pool)

    # ═══════════════════════════════════════
    #  탭 4: 이상형 월드컵 (Elo Arena)
    # ═══════════════════════════════════════
    def _build_arena_tab(self):
        C = self.C
        f = tk.Frame(self.tab_arena, bg=C["BG"], padx=16, pady=16)
        f.pack(fill=tk.BOTH, expand=True)

        top = tk.Frame(f, bg=C["BG"])
        top.pack(fill=tk.X)
        self.arena_mode_var = tk.StringVar(value="전체 리그")
        ttk.Combobox(top, textvariable=self.arena_mode_var,
                      values=["전체 리그", "즐겨찾기 리그", "최근 개선 배치"], state="readonly", width=14
                      ).pack(side=tk.LEFT)
        self.arena_mode_var.trace_add("write", lambda *a: self._refresh_arena())
        self.arena_count_label = tk.Label(top, text="", bg=C["BG"], fg=C["ACCENT"])
        self.arena_count_label.pack(side=tk.RIGHT)

        self.arena_body = tk.Frame(f, bg=C["BG"])
        self.arena_body.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def _arena_pool(self):
        mode = self.arena_mode_var.get()
        if mode == "즐겨찾기 리그":
            return [c for c in self.combinations if c["is_favorite"]]
        if mode == "최근 개선 배치":
            if not getattr(self, "last_batch_id", None):
                return []
            return [c for c in self.combinations if c.get("batch_id") == self.last_batch_id]
        return list(self.combinations)

    def _refresh_arena(self):
        for w in self.arena_body.winfo_children():
            w.destroy()
        C = self.C

        matches = sum(1 for h in self.history if h.get("type") == "arena")
        self.arena_count_label.configure(text=f"누적 대결 {matches}회 | 조합 {len(self.combinations)}개")

        pool = self._arena_pool()
        if len(pool) < 2:
            msg = "대결할 조합이 2개 이상 필요합니다. 먼저 조합을 생성해 주세요."
            if self.arena_mode_var.get() == "최근 개선 배치" and not getattr(self, "last_batch_id", None):
                msg = "아직 '그림체 개선' 탭에서 생성한 배치가 없습니다."
            tk.Label(self.arena_body, text=msg, bg=C["BG"], fg=C["FG2"]).pack(pady=40)
            return

        if not self.current_match or self.current_match["a"] not in {c["id"] for c in pool} \
                or self.current_match["b"] not in {c["id"] for c in pool}:
            a, b = random.sample(pool, 2)
            self.current_match = {"a": a["id"], "b": b["id"]}

        combo_a = next(c for c in self.combinations if c["id"] == self.current_match["a"])
        combo_b = next(c for c in self.combinations if c["id"] == self.current_match["b"])

        pane = tk.Frame(self.arena_body, bg=C["BG"])
        pane.pack(fill=tk.BOTH, expand=True)
        self._build_combo_card(pane, combo_a, "A", side=tk.LEFT)
        tk.Label(pane, text="VS", bg=C["BG"], fg=C["RED"], font=("Consolas", 16, "bold")
                  ).pack(side=tk.LEFT, padx=10)
        self._build_combo_card(pane, combo_b, "B", side=tk.LEFT)

    def _build_combo_card(self, parent, combo, label, side):
        C = self.C
        card = tk.Frame(parent, bg=C["BG2"], padx=10, pady=10)
        card.pack(side=side, fill=tk.BOTH, expand=True, padx=6)

        header = tk.Frame(card, bg=C["BG2"])
        header.pack(fill=tk.X)
        tk.Label(header, text=f"조합 {label} (Gen.{combo['generation']}, {combo['elo']}점)",
                  bg=C["BG2"], fg=C["FG"], font=("Consolas", 10, "bold")).pack(side=tk.LEFT)
        fav_text = "🌟 해제" if combo["is_favorite"] else "⭐ 즐겨찾기"
        tk.Button(header, text=fav_text, command=lambda: self._toggle_favorite(combo["id"]),
                   bg=C["BG3"], fg=C["FG"], relief=tk.FLAT, font=("Consolas", 8)
                   ).pack(side=tk.RIGHT)

        img_path = IMG_DIR / combo["image_file"]
        if img_path.exists():
            try:
                photo = load_thumbnail(img_path, max_w=560)
                self._image_cache[combo["id"]] = photo
                img_label = tk.Label(card, image=photo, bg=C["BG2"], cursor="hand2")
                img_label.pack(pady=8)
                img_label.bind("<Button-1>", lambda e: self._vote(label))
                img_label.bind("<Double-Button-1>", lambda e: open_in_viewer(img_path))
            except Exception:
                tk.Label(card, text="(이미지를 불러올 수 없습니다)", bg=C["BG2"], fg=C["FG2"]).pack(pady=8)
        else:
            tk.Label(card, text="(이미지 없음)", bg=C["BG2"], fg=C["FG2"]).pack(pady=8)

        style_box = tk.Text(card, height=3, bg="#121214", fg=C["FG2"], font=("Consolas", 8),
                              wrap=tk.WORD)
        style_box.insert("1.0", combo["style"])
        style_box.configure(state=tk.DISABLED)
        style_box.pack(fill=tk.X)

        btn_row = tk.Frame(card, bg=C["BG2"])
        btn_row.pack(fill=tk.X, pady=(8, 0))
        tk.Button(btn_row, text="📋 태그 복사", command=lambda: self._copy_to_clipboard(combo["style"]),
                   bg=C["BG3"], fg=C["FG"], relief=tk.FLAT).pack(side=tk.LEFT)
        tk.Button(btn_row, text=f"이 조합({label}) 선택!", command=lambda: self._vote(label),
                   bg=C["ACCENT"], fg=C["BG"], relief=tk.FLAT).pack(side=tk.RIGHT)

    def _copy_to_clipboard(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo("복사됨", "태그가 클립보드에 복사되었습니다!")

    def _toggle_favorite(self, combo_id):
        for c in self.combinations:
            if c["id"] == combo_id:
                c["is_favorite"] = not c["is_favorite"]
        self._save()
        self._refresh_arena()

    def _vote(self, winner_label):
        if not self.current_match:
            return
        combo_a = next(c for c in self.combinations if c["id"] == self.current_match["a"])
        combo_b = next(c for c in self.combinations if c["id"] == self.current_match["b"])
        a_won = winner_label == "A"

        new_a, new_b = calc_elo(combo_a["elo"], combo_b["elo"], a_won)
        combo_a["elo"], combo_b["elo"] = new_a, new_b
        combo_a["matches"] += 1
        combo_b["matches"] += 1
        combo_a["wins"] += 1 if a_won else 0
        combo_b["wins"] += 1 if not a_won else 0

        tags_a = set(artists_from_style(combo_a["style"]))
        tags_b = set(artists_from_style(combo_b["style"]))
        for a in self.artists:
            in_a, in_b = a["tag"] in tags_a, a["tag"] in tags_b
            if in_a or in_b:
                a["arena_matches"] += 1
                if in_a:
                    a["arena_wins"] += 1 if a_won else 0
                    a["elo"] += 15 if a_won else -15
                if in_b:
                    a["arena_wins"] += 1 if not a_won else 0
                    a["elo"] += 15 if not a_won else -15

        self.history.insert(0, {
            "type": "arena",
            "winner": combo_a["style"] if a_won else combo_b["style"],
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })
        self.current_match = None
        self._save()
        self._refresh_arena()

    # ═══════════════════════════════════════
    #  탭 5: 즐겨찾기 / 정리
    # ═══════════════════════════════════════
    def _build_favorites_tab(self):
        C = self.C
        f = tk.Frame(self.tab_favorites, bg=C["BG"], padx=16, pady=16)
        f.pack(fill=tk.BOTH, expand=True)

        self.fav_list_frame = tk.Frame(f, bg=C["BG"])
        canvas = tk.Canvas(self.fav_list_frame, bg=C["BG"], highlightthickness=0, height=520)
        scrollbar = ttk.Scrollbar(self.fav_list_frame, orient="vertical", command=canvas.yview)
        self.fav_inner = tk.Frame(canvas, bg=C["BG"])
        self.fav_inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.fav_inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.fav_list_frame.pack(fill=tk.BOTH, expand=True)

        clean = tk.LabelFrame(f, text="🗑️ 저평가 조합 정리", bg=C["BG"], fg=C["RED"])
        clean.pack(fill=tk.X, pady=(12, 0))
        row = tk.Frame(clean, bg=C["BG"])
        row.pack(fill=tk.X, pady=6, padx=6)
        self.min_matches_var = tk.IntVar(value=5)
        self.min_elo_var = tk.IntVar(value=900)
        tk.Label(row, text="최소 평가 횟수", bg=C["BG"], fg=C["FG2"]).pack(side=tk.LEFT)
        tk.Spinbox(row, from_=1, to=999, textvariable=self.min_matches_var, width=6,
                    bg=C["BG2"], fg=C["FG"]).pack(side=tk.LEFT, padx=(4, 14))
        tk.Label(row, text="폐기 커트라인(Elo 미만)", bg=C["BG"], fg=C["FG2"]).pack(side=tk.LEFT)
        tk.Spinbox(row, from_=0, to=3000, textvariable=self.min_elo_var, width=8,
                    bg=C["BG2"], fg=C["FG"]).pack(side=tk.LEFT, padx=(4, 14))
        tk.Button(row, text="🧹 조건에 맞는 조합 폐기", command=self._discard_low_rated,
                   bg=C["RED"], fg=C["BG"], relief=tk.FLAT).pack(side=tk.LEFT)

    def _discard_low_rated(self):
        min_m, min_e = self.min_matches_var.get(), self.min_elo_var.get()
        before = len(self.combinations)
        self.combinations = [
            c for c in self.combinations
            if c["is_locked"] or not (c["matches"] >= min_m and c["elo"] < min_e)
        ]
        removed = before - len(self.combinations)
        self._save()
        self._refresh_favorites()
        self._refresh_dashboard()
        messagebox.showinfo("완료", f"기준 미달 조합 {removed}개가 폐기되었습니다.")

    def _refresh_favorites(self):
        for w in self.fav_inner.winfo_children():
            w.destroy()
        C = self.C
        favs = [c for c in self.combinations if c["is_favorite"]]
        if not favs:
            tk.Label(self.fav_inner, text="즐겨찾기에 등록된 조합이 없습니다. 월드컵에서 ⭐를 눌러보세요.",
                      bg=C["BG"], fg=C["FG2"]).pack(pady=20)
            return

        for combo in favs:
            row = tk.Frame(self.fav_inner, bg=C["BG2"], padx=10, pady=8)
            row.pack(fill=tk.X, pady=4)

            img_path = IMG_DIR / combo["image_file"]
            if img_path.exists():
                try:
                    photo = load_thumbnail(img_path, max_w=260)
                    self._image_cache[f"fav_{combo['id']}"] = photo
                    tk.Label(row, image=photo, bg=C["BG2"], cursor="hand2").grid(
                        row=0, column=0, rowspan=4, padx=(0, 10))
                except Exception:
                    pass

            info = tk.Frame(row, bg=C["BG2"])
            info.grid(row=0, column=1, sticky="nsew")
            row.grid_columnconfigure(1, weight=1)

            lock_mark = "🔒 " if combo["is_locked"] else ""
            tk.Label(info, text=f"{lock_mark}Gen.{combo['generation']} | {combo['elo']}점 | "
                                  f"{combo['wins']}승 {combo['matches']}전",
                      bg=C["BG2"], fg=C["GREEN"], font=("Consolas", 9)).pack(anchor="w")
            tk.Label(info, text=combo["style"][:120] + ("..." if len(combo["style"]) > 120 else ""),
                      bg=C["BG2"], fg=C["FG2"], font=("Consolas", 8), wraplength=600, justify="left"
                      ).pack(anchor="w", pady=(2, 4))

            memo_var = tk.StringVar(value=combo.get("memo", ""))
            memo_entry = tk.Entry(info, textvariable=memo_var, bg="#121214", fg=C["FG"],
                                    insertbackground=C["FG"])
            memo_entry.pack(fill=tk.X)
            memo_entry.bind("<FocusOut>", lambda e, cid=combo["id"], v=memo_var: self._update_memo(cid, v.get()))

            btns = tk.Frame(row, bg=C["BG2"])
            btns.grid(row=0, column=2, padx=(10, 0))
            lock_text = "🔓 잠금 해제" if combo["is_locked"] else "🔒 잠금"
            tk.Button(btns, text=lock_text, command=lambda cid=combo["id"]: self._toggle_lock(cid),
                       bg=C["BG3"], fg=C["FG"], relief=tk.FLAT, width=14).pack(pady=2)
            tk.Button(btns, text="📋 복사", command=lambda s=combo["style"]: self._copy_to_clipboard(s),
                       bg=C["BG3"], fg=C["FG"], relief=tk.FLAT, width=14).pack(pady=2)
            tk.Button(btns, text="🖼 원본 보기", command=lambda p=img_path: open_in_viewer(p),
                       bg=C["BG3"], fg=C["FG"], relief=tk.FLAT, width=14).pack(pady=2)
            tk.Button(btns, text="❌ 즐겨찾기 해제", command=lambda cid=combo["id"]: self._toggle_favorite(cid) or self._refresh_favorites(),
                       bg=C["RED"], fg=C["BG"], relief=tk.FLAT, width=14).pack(pady=2)

    def _update_memo(self, combo_id, memo):
        for c in self.combinations:
            if c["id"] == combo_id:
                c["memo"] = memo
        self._save()

    def _toggle_lock(self, combo_id):
        for c in self.combinations:
            if c["id"] == combo_id:
                c["is_locked"] = not c["is_locked"]
        self._save()
        self._refresh_favorites()

    # ═══════════════════════════════════════
    #  탭 6: 기존 그림체 개선 (유사 그림체 생성)
    # ═══════════════════════════════════════
    def _build_improve_tab(self):
        C = self.C
        f = tk.Frame(self.tab_improve, bg=C["BG"], padx=16, pady=16)
        f.pack(fill=tk.BOTH, expand=True)

        tk.Label(f, text=("기존 그림체 하나를 골라, 배합비(가중치)를 조금씩 흔들거나 작가를 "
                            "교체/추가/제거한 '유사 그림체' 여러 개를 만듭니다. 만들어지는 즉시 "
                            "NAI로 실제 이미지를 생성하고, 원본과 변형본을 묶어 전용 배치로 "
                            "표시해 둡니다. 그 배치만 따로 골라 이상형 월드컵을 돌릴 수 있어요."),
                  bg=C["BG"], fg=C["FG2"], wraplength=1050, justify="left").pack(anchor="w")

        picker_row = tk.Frame(f, bg=C["BG"])
        picker_row.pack(fill=tk.X, pady=(10, 4))
        tk.Label(picker_row, text="기반이 될 그림체 선택:", bg=C["BG"], fg=C["FG2"]).pack(side=tk.LEFT)
        tk.Button(picker_row, text="🔄 목록 새로고침", command=self._refresh_improve_combo_list,
                   bg=C["BG3"], fg=C["FG"], relief=tk.FLAT).pack(side=tk.RIGHT)

        self.improve_combo_tree = ttk.Treeview(f, columns=("gen", "elo", "record"),
                                                  show="tree headings", height=6)
        self.improve_combo_tree.heading("#0", text="조합")
        self.improve_combo_tree.heading("gen", text="세대")
        self.improve_combo_tree.heading("elo", text="선호도")
        self.improve_combo_tree.heading("record", text="전적")
        self.improve_combo_tree.column("#0", width=650)
        self.improve_combo_tree.pack(fill=tk.X, pady=(0, 6))
        self.improve_combo_tree.bind("<<TreeviewSelect>>", lambda e: self._load_selected_combo_for_improve())

        edit_row = tk.Frame(f, bg=C["BG"])
        edit_row.pack(fill=tk.X, pady=(6, 4))
        tk.Label(edit_row, text="기반 그림체 (직접 붙여넣기도 가능)", bg=C["BG"], fg=C["FG2"]
                  ).pack(anchor="w")
        self.improve_style_box = tk.Text(edit_row, height=3, bg=C["BG2"], fg=C["FG"],
                                            insertbackground=C["FG"])
        self.improve_style_box.pack(fill=tk.X, pady=(2, 4))

        # 변형 옵션
        opt_row = tk.Frame(f, bg=C["BG"])
        opt_row.pack(fill=tk.X, pady=(8, 4))

        self.improve_count_var = tk.IntVar(value=6)
        self.improve_jitter_var = tk.DoubleVar(value=0.3)
        self.improve_retag_var = tk.DoubleVar(value=0.15)
        self.improve_add_var = tk.DoubleVar(value=0.3)
        self.improve_remove_var = tk.DoubleVar(value=0.2)
        self.improve_min_var = tk.IntVar(value=5)
        self.improve_max_var = tk.IntVar(value=10)

        def field(parent, label, var, frm, to, inc, width=7):
            box = tk.Frame(parent, bg=C["BG"])
            box.pack(side=tk.LEFT, padx=(0, 14))
            tk.Label(box, text=label, bg=C["BG"], fg=C["FG2"], font=("Consolas", 8)).pack(anchor="w")
            tk.Spinbox(box, from_=frm, to=to, increment=inc, textvariable=var, width=width,
                        bg=C["BG2"], fg=C["FG"]).pack()

        field(opt_row, "생성 개수", self.improve_count_var, 1, 30, 1)
        field(opt_row, "가중치 변동폭", self.improve_jitter_var, 0.0, 1.0, 0.1)
        field(opt_row, "작가 교체 확률", self.improve_retag_var, 0.0, 1.0, 0.05)
        field(opt_row, "작가 추가 확률", self.improve_add_var, 0.0, 1.0, 0.05)
        field(opt_row, "작가 제거 확률", self.improve_remove_var, 0.0, 1.0, 0.05)
        field(opt_row, "작가 수 최소", self.improve_min_var, 1, 20, 1)
        field(opt_row, "최대", self.improve_max_var, 1, 20, 1)

        tk.Button(f, text="🎲 유사 그림체 생성 + NAI 이미지 생성 시작",
                   command=self._start_similar_style_generation,
                   bg=C["ACCENT"], fg=C["BG"], relief=tk.FLAT, padx=10).pack(anchor="e", pady=(6, 10))

        # 최근 배치 결과
        batch_header = tk.Frame(f, bg=C["BG"])
        batch_header.pack(fill=tk.X)
        self.improve_batch_label = tk.Label(batch_header, text="아직 생성된 배치가 없습니다.",
                                               bg=C["BG"], fg=C["ACCENT"], font=("Consolas", 9, "bold"))
        self.improve_batch_label.pack(side=tk.LEFT)
        tk.Button(batch_header, text="⚔ 이 배치로 월드컵 시작", command=self._jump_to_batch_arena,
                   bg=C["GREEN"], fg=C["BG"], relief=tk.FLAT).pack(side=tk.RIGHT)

        self.improve_batch_tree = ttk.Treeview(f, columns=("gen", "elo", "type"),
                                                  show="tree headings", height=10)
        self.improve_batch_tree.heading("#0", text="조합")
        self.improve_batch_tree.heading("gen", text="세대")
        self.improve_batch_tree.heading("elo", text="선호도")
        self.improve_batch_tree.heading("type", text="구분")
        self.improve_batch_tree.column("#0", width=650)
        self.improve_batch_tree.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.improve_batch_tree.bind("<Double-1>", lambda e: self._open_batch_selection_image())

        self.improve_original_id = None
        self.improve_original_gen = 1
        self.last_batch_id = None

    def _refresh_improve_combo_list(self):
        self.improve_combo_tree.delete(*self.improve_combo_tree.get_children())
        for c in sorted(self.combinations, key=lambda x: -x["elo"]):
            self.improve_combo_tree.insert(
                "", "end", iid=c["id"], text=c["style"][:90] + ("..." if len(c["style"]) > 90 else ""),
                values=(c["generation"], c["elo"], f"{c['wins']}승 {c['matches']}전"))

    def _load_selected_combo_for_improve(self):
        sel = self.improve_combo_tree.selection()
        if not sel:
            return
        combo_id = sel[0]
        combo = next((c for c in self.combinations if c["id"] == combo_id), None)
        if not combo:
            return
        self.improve_original_id = combo_id
        self.improve_original_gen = combo["generation"]
        self.improve_style_box.delete("1.0", tk.END)
        self.improve_style_box.insert("1.0", combo["style"])

    @staticmethod
    def _style_from_pairs(pairs):
        parts = []
        for w, tag in pairs:
            space = " " if tag[-1:].isdigit() else ""
            parts.append(f"{max(0.1, round(w, 1)):.1f}::{tag}{space}::")
        return ", ".join(parts)

    def _start_similar_style_generation(self):
        style_str = self.improve_style_box.get("1.0", tk.END).strip()
        seed_pairs = parse_style_combo(style_str)
        if not seed_pairs:
            messagebox.showwarning("알림", "기반이 될 그림체 조합을 입력하거나 목록에서 선택해 주세요.")
            return
        min_t, max_t = self.improve_min_var.get(), self.improve_max_var.get()
        if min_t > max_t:
            messagebox.showwarning("알림", "최소 작가 수가 최대보다 클 수 없습니다.")
            return

        # 현재 선택이 텍스트 박스 내용과 다르면(직접 편집한 경우) 원본 참조를 해제합니다.
        if self.improve_original_id:
            ref = next((c for c in self.combinations if c["id"] == self.improve_original_id), None)
            if not ref or ref["style"] != style_str:
                self.improve_original_id = None
                self.improve_original_gen = 1

        count = self.improve_count_var.get()
        mutation_opts = dict(
            weight_jitter=self.improve_jitter_var.get(),
            retag_rate=self.improve_retag_var.get(),
            add_chance=self.improve_add_var.get(),
            remove_chance=self.improve_remove_var.get(),
            min_tags=min_t, max_tags=max_t,
        )
        settings = self._current_gen_settings()
        self._run_bg(self._do_similar_style_generation, seed_pairs, style_str, count, mutation_opts, settings)

    def _do_similar_style_generation(self, seed_pairs, seed_style, count, mutation_opts, settings):
        batch_id = uuid.uuid4().hex[:8]
        self.last_batch_id = batch_id
        all_tags = [a["tag"] for a in self.artists]
        seed_gen = self.improve_original_gen or 1

        # 원본을 배치에 편입: 목록에서 고른 조합이면 이미지가 있으니 배치 태그만 붙이고,
        # 직접 붙여넣은 조합이면 원본 이미지를 새로 생성합니다.
        original_combo = None
        if self.improve_original_id:
            original_combo = next((c for c in self.combinations if c["id"] == self.improve_original_id), None)

        if original_combo:
            original_combo["batch_id"] = batch_id
            self._save()
        else:
            self._log("┌ 원본 그림체 이미지를 새로 생성합니다...\n", "dim")
            try:
                seed = random.randint(0, 4_294_967_295)
                png = generate_style_image(
                    settings["api_key"], settings["subject"], seed_style, settings["negative"],
                    settings["model"], settings["width"], settings["height"],
                    settings["steps"], settings["cfg"], settings["sampler"], seed,
                    subject_weight=settings["subject_weight"],
                )
                combo_id = uuid.uuid4().hex[:10]
                img_name = f"{combo_id}.png"
                (IMG_DIR / img_name).write_bytes(png)
                self.combinations.insert(0, {
                    "id": combo_id, "style": seed_style, "elo": 1000, "matches": 0, "wins": 0,
                    "is_favorite": False, "is_locked": False, "memo": "원본 (직접 입력)",
                    "generation": seed_gen, "image_file": img_name, "seed": seed,
                    "batch_id": batch_id,
                })
                self._save()
                self._log("└ ✓ 원본 이미지 생성 완료\n\n", "ok")
            except Exception as e:
                self._log(f"└ ✗ 원본 생성 실패: {e}\n\n", "err")

        seen_styles = {c["style"] for c in self.combinations}
        made = 0
        loops = 0
        while made < count and loops < count * 20:
            if self._stop_event.is_set():
                self._log("⛔ 중단됨\n", "err")
                break
            loops += 1
            variant_pairs = mutate_style_pairs(seed_pairs, all_tags, **mutation_opts)
            if not variant_pairs:
                continue
            variant_style = self._style_from_pairs(variant_pairs)
            if variant_style in seen_styles:
                continue
            seen_styles.add(variant_style)

            combo_id = uuid.uuid4().hex[:10]
            seed = random.randint(0, 4_294_967_295)
            self._set_status(f"유사 그림체 {made + 1}/{count} 생성 중 (NAI 이미지 요청)...")
            self._log(f"┌ 유사 그림체 {made + 1}/{count}: {variant_style[:80]}...\n", "dim")
            try:
                png = generate_style_image(
                    settings["api_key"], settings["subject"], variant_style, settings["negative"],
                    settings["model"], settings["width"], settings["height"],
                    settings["steps"], settings["cfg"], settings["sampler"], seed,
                    subject_weight=settings["subject_weight"],
                )
                img_name = f"{combo_id}.png"
                (IMG_DIR / img_name).write_bytes(png)
                self.combinations.insert(0, {
                    "id": combo_id, "style": variant_style, "elo": 1000, "matches": 0, "wins": 0,
                    "is_favorite": False, "is_locked": False,
                    "memo": f"유사 그림체 (기반: {self.improve_original_id or '직접 입력'})",
                    "generation": seed_gen + 1, "image_file": img_name, "seed": seed,
                    "batch_id": batch_id,
                })
                made += 1
                self._save()
                self._log(f"└ ✓ 생성 완료 ({img_name})\n\n", "ok")
            except Exception as e:
                self._log(f"└ ✗ 실패: {e}\n\n", "err")

            self._set_progress(made / count * 100)
            if made < count and not self._stop_event.is_set():
                time.sleep(settings["delay"])

        self._set_status(f"완료 — 유사 그림체 {made}개 생성 (원본 포함 배치)")
        self._set_progress(100)
        self.root.after(0, self._refresh_improve_batch_list)
        self.root.after(0, self._update_gen_count_label)

    def _refresh_improve_batch_list(self):
        self.improve_batch_tree.delete(*self.improve_batch_tree.get_children())
        if not self.last_batch_id:
            self.improve_batch_label.configure(text="아직 생성된 배치가 없습니다.")
            return
        batch_combos = [c for c in self.combinations if c.get("batch_id") == self.last_batch_id]
        self.improve_batch_label.configure(
            text=f"최근 배치: {len(batch_combos)}개 조합 (배치 ID {self.last_batch_id})")
        for c in batch_combos:
            kind = "원본" if not c["memo"].startswith("유사 그림체") else "변형"
            self.improve_batch_tree.insert(
                "", "end", iid=c["id"], text=c["style"][:90] + ("..." if len(c["style"]) > 90 else ""),
                values=(c["generation"], c["elo"], kind))

    def _open_batch_selection_image(self):
        sel = self.improve_batch_tree.selection()
        if not sel:
            return
        combo = next((c for c in self.combinations if c["id"] == sel[0]), None)
        if combo:
            img_path = IMG_DIR / combo["image_file"]
            if img_path.exists():
                open_in_viewer(img_path)

    def _jump_to_batch_arena(self):
        if not self.last_batch_id:
            messagebox.showinfo("알림", "먼저 유사 그림체 배치를 생성해 주세요.")
            return
        self.arena_mode_var.set("최근 개선 배치")
        self.notebook.select(self.tab_arena)

    # ═══════════════════════════════════════
    #  탭 7: 조합 진화 (유전 알고리즘)
    # ═══════════════════════════════════════
    def _build_evolution_tab(self):
        C = self.C
        f = tk.Frame(self.tab_evolution, bg=C["BG"], padx=16, pady=16)
        f.pack(fill=tk.BOTH, expand=True)

        tk.Label(f, text=("월드컵에서 검증된 상위 20개 조합과 잠금된 조합을 교차·변이시켜 "
                            "다음 세대 조합을 만들고, 즉시 NAI로 이미지를 생성합니다."),
                  bg=C["BG"], fg=C["FG2"], wraplength=1000, justify="left").pack(anchor="w")

        row = tk.Frame(f, bg=C["BG"])
        row.pack(fill=tk.X, pady=10)
        self.evo_count_var = tk.IntVar(value=5)
        self.evo_min_var = tk.IntVar(value=5)
        self.evo_max_var = tk.IntVar(value=10)

        def field(label, var):
            tk.Label(row, text=label, bg=C["BG"], fg=C["FG2"]).pack(side=tk.LEFT, padx=(0, 4))
            tk.Spinbox(row, from_=1, to=999, textvariable=var, width=8,
                        bg=C["BG2"], fg=C["FG"]).pack(side=tk.LEFT, padx=(0, 14))

        field("생성할 자손 수", self.evo_count_var)
        field("조합당 작가 수 최소", self.evo_min_var)
        field("최대", self.evo_max_var)
        tk.Button(row, text="🧬 진화 + NAI 이미지 생성 시작", command=self._start_evolution,
                   bg=C["ACCENT"], fg=C["BG"], relief=tk.FLAT, padx=10).pack(side=tk.LEFT)

        tk.Label(f, text="진화 기록", bg=C["BG"], fg=C["FG"], font=("Consolas", 10, "bold")
                  ).pack(anchor="w", pady=(16, 4))
        self.evo_tree = ttk.Treeview(f, columns=("time", "gen"), show="tree headings", height=12)
        self.evo_tree.heading("#0", text="탄생한 조합")
        self.evo_tree.heading("time", text="시각")
        self.evo_tree.heading("gen", text="세대")
        self.evo_tree.column("#0", width=700)
        self.evo_tree.column("time", width=100, anchor="center")
        self.evo_tree.column("gen", width=80, anchor="center")
        self.evo_tree.pack(fill=tk.BOTH, expand=True)

    def _start_evolution(self):
        count = self.evo_count_var.get()
        min_t, max_t = self.evo_min_var.get(), self.evo_max_var.get()
        if min_t > max_t:
            messagebox.showwarning("알림", "최소 작가 수가 최대보다 클 수 없습니다.")
            return
        candidates = [c for c in self.combinations if c["matches"] > 0 or c["is_locked"]]
        candidates.sort(key=lambda c: -c["elo"])
        candidates = candidates[:20]
        if len(candidates) < 2:
            messagebox.showwarning("알림", "평가된(전적 있음) 또는 잠금된 조합이 최소 2개 이상 필요합니다.")
            return
        settings = self._current_gen_settings()
        self._run_bg(self._do_evolution, count, min_t, max_t, candidates, settings)

    def _do_evolution(self, count, min_t, max_t, candidates, settings):
        all_tags = [a["tag"] for a in self.artists]
        made = 0
        for i in range(count):
            if self._stop_event.is_set():
                self._log("⛔ 중단됨\n", "err")
                break
            p_a, p_b = random.choice(candidates), random.choice(candidates)
            child_style = crossover_and_mutate(p_a["style"], p_b["style"], all_tags, 0.05, min_t, max_t)
            next_gen = max(p_a["generation"], p_b["generation"]) + 1

            combo_id = uuid.uuid4().hex[:10]
            seed = random.randint(0, 4_294_967_295)
            self._set_status(f"진화 자손 {i + 1}/{count} 생성 중 (NAI 이미지 요청)...")
            self._log(f"┌ 진화 자손 {i + 1}/{count} (Gen.{next_gen}): {child_style[:80]}...\n", "dim")
            try:
                png = generate_style_image(
                    settings["api_key"], settings["subject"], child_style, settings["negative"],
                    settings["model"], settings["width"], settings["height"],
                    settings["steps"], settings["cfg"], settings["sampler"], seed,
                    subject_weight=settings["subject_weight"],
                )
                img_name = f"{combo_id}.png"
                (IMG_DIR / img_name).write_bytes(png)
                self.combinations.insert(0, {
                    "id": combo_id, "style": child_style, "elo": 1050, "matches": 0, "wins": 0,
                    "is_favorite": False, "is_locked": False, "memo": "",
                    "generation": next_gen, "image_file": img_name, "seed": seed,
                })
                self.history.insert(0, {
                    "type": "evolution", "winner": child_style, "generation": next_gen,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                })
                made += 1
                self._save()
                self._log(f"└ ✓ 생성 완료 ({img_name})\n\n", "ok")
            except Exception as e:
                self._log(f"└ ✗ 실패: {e}\n\n", "err")

            self._set_progress((i + 1) / count * 100)
            if i < count - 1 and not self._stop_event.is_set():
                time.sleep(settings["delay"])

        self._set_status(f"완료 — 진화 자손 {made}개 생성")
        self._set_progress(100)
        self.root.after(0, self._refresh_evolution_history)
        self.root.after(0, self._update_gen_count_label)

    def _refresh_evolution_history(self):
        self.evo_tree.delete(*self.evo_tree.get_children())
        for h in self.history:
            if h.get("type") == "evolution":
                self.evo_tree.insert("", "end", text=h["winner"][:100],
                                       values=(h["timestamp"], h.get("generation", "?")))

    # ═══════════════════════════════════════
    #  탭 8: 통계 대시보드
    # ═══════════════════════════════════════
    def _build_dashboard_tab(self):
        C = self.C
        f = tk.Frame(self.tab_dashboard, bg=C["BG"], padx=16, pady=16)
        f.pack(fill=tk.BOTH, expand=True)

        top = tk.Frame(f, bg=C["BG"])
        top.pack(fill=tk.X)
        self.dash_summary = tk.Label(top, text="", bg=C["BG"], fg=C["ACCENT"], font=("Consolas", 10, "bold"))
        self.dash_summary.pack(side=tk.LEFT)
        tk.Button(top, text="💾 조합 와일드카드 내보내기 (.txt)", command=self._export_wildcard,
                   bg=C["GREEN"], fg=C["BG"], relief=tk.FLAT).pack(side=tk.RIGHT)

        split = tk.Frame(f, bg=C["BG"])
        split.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        left = tk.Frame(split, bg=C["BG"])
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        tk.Label(left, text="🏆 인기 작가 Top 10", bg=C["BG"], fg=C["FG"], font=("Consolas", 10, "bold")).pack(anchor="w")
        self.dash_artist_tree = ttk.Treeview(left, columns=("elo", "winrate", "count"), show="tree headings", height=14)
        self.dash_artist_tree.heading("#0", text="작가")
        self.dash_artist_tree.heading("elo", text="선호도")
        self.dash_artist_tree.heading("winrate", text="승률")
        self.dash_artist_tree.heading("count", text="검출수")
        self.dash_artist_tree.pack(fill=tk.BOTH, expand=True)

        right = tk.Frame(split, bg=C["BG"])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        tk.Label(right, text="🔥 상위 조합 Top 10", bg=C["BG"], fg=C["FG"], font=("Consolas", 10, "bold")).pack(anchor="w")
        self.dash_combo_tree = ttk.Treeview(right, columns=("gen", "elo", "record"), show="tree headings", height=14)
        self.dash_combo_tree.heading("#0", text="조합")
        self.dash_combo_tree.heading("gen", text="세대")
        self.dash_combo_tree.heading("elo", text="선호도")
        self.dash_combo_tree.heading("record", text="전적")
        self.dash_combo_tree.pack(fill=tk.BOTH, expand=True)

    def _export_wildcard(self):
        if not self.combinations:
            messagebox.showinfo("알림", "내보낼 조합이 없습니다.")
            return
        ensure_dirs()
        out_path = DATA_DIR / f"style_mix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        out_path.write_text("\n".join(c["style"] for c in self.combinations), encoding="utf-8")
        messagebox.showinfo("완료", f"내보내기 완료:\n{out_path}")

    def _refresh_dashboard(self):
        gen_counts = {}
        for c in self.combinations:
            gen_counts[c["generation"]] = gen_counts.get(c["generation"], 0) + 1
        gen_text = " | ".join(f"Gen.{g}: {n}개" for g, n in sorted(gen_counts.items()))
        self.dash_summary.configure(
            text=f"조합 총 {len(self.combinations)}개 | 작가 {len(self.artists)}명 | {gen_text}")

        self.dash_artist_tree.delete(*self.dash_artist_tree.get_children())
        top_artists = sorted(self.artists, key=lambda a: -a["elo"])[:10]
        for a in top_artists:
            wr = (a["arena_wins"] / a["arena_matches"] * 100) if a["arena_matches"] else 0
            self.dash_artist_tree.insert("", "end", text=a["tag"],
                                           values=(a["elo"], f"{wr:.1f}%", a["count"]))

        self.dash_combo_tree.delete(*self.dash_combo_tree.get_children())
        top_combos = sorted((c for c in self.combinations if c["matches"] > 0),
                              key=lambda c: -c["elo"])[:10]
        for c in top_combos:
            self.dash_combo_tree.insert("", "end", text=c["style"][:60] + "...",
                                          values=(c["generation"], c["elo"],
                                                  f"{c['wins']}승 {c['matches']}전"))


def main():
    root = tk.Tk()
    StyleArenaApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
