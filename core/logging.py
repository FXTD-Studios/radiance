"""Vibrant and premium terminal console logging system for Radiance.

This module intercepts and formats all standard logging calls under the 'radiance'
logger tree. It features beautifully stylized color badges, clean timestamps,
relative component channels, and native VT100 support for Windows consoles.
It dynamically falls back to ASCII layouts on consoles that do not support Unicode
or UTF-8 to prevent UnicodeEncodeErrors, and supports multiple developer themes
selectable via the 'RADIANCE_LOG_THEME' environment variable.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import time
from typing import Sequence

# Define custom logging level for Success
SUCCESS_LEVEL = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

def _logger_success(self, message, *args, **kws):
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kws)

# Bind the success method to logging.Logger
logging.Logger.success = _logger_success


def supports_color() -> bool:
    """Returns True if the terminal stdout supports ANSI color escape sequences."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        # Check standard environments that support colors even when not a raw TTY
        term_program = os.environ.get("TERM_PROGRAM")
        if term_program in ("vscode", "Apple_Terminal", "Hyper"):
            return True
        return False
    if sys.platform == "win32":
        # Always supported in modern Windows Terminal or with virtual terminal enabled
        if os.environ.get("WT_SESSION") or "ConEmuANSI" in os.environ:
            return True
    return True


def supports_unicode() -> bool:
    """Returns True if the output stream supports full unicode/UTF-8 box characters."""
    if os.environ.get("NO_UNICODE") or os.environ.get("NO_UTF8"):
        return False
    try:
        # Check standard stdout stream encoding
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        # Test if the stream's encoding can successfully encode our core theme unicode symbols
        "✦".encode(encoding)
        "┌".encode(encoding)
        return True
    except Exception:
        return False


def init_windows_ansi() -> bool:
    """Configures the Windows console host to process ANSI/VT100 escape sequences.

    Returns True if successfully initialized or unnecessary, False otherwise.
    """
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        hStdOut = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        if hStdOut and hStdOut != -1:
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(hStdOut, ctypes.byref(mode)):
                # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
                if kernel32.SetConsoleMode(hStdOut, mode.value | 0x0004):
                    return True
    except Exception:
        pass
    return False


class TextTheme:
    """Configures the console theme layout and character symbols.

    Supports five pre-defined premium developer styles:
    - 'classic': Elegant pastel violet/blue theme with soft badges.
    - 'cyberpunk': Neon magenta/cyan theme with high-voltage separators.
    - 'minimalist': Ultra-clean, low-noise monochrome slate theme.
    - 'matrix': Retro cyber terminal green theme.
    - 'compact': Space-saving layout with color gradients.
    """

    def __init__(self, theme_name: str = "minimalist", use_unicode: bool = True, use_color: bool = True):
        self.theme_name = theme_name.lower().strip()
        if self.theme_name not in ("classic", "cyberpunk", "minimalist", "matrix", "compact", "pro"):
            self.theme_name = "pro"

        self.use_unicode = use_unicode
        self.use_color = use_color
        self.columnar = False

        # Base box character sets
        if use_unicode:
            # Double box chars
            self.box_dbl_tl, self.box_dbl_tr = "╔", "╗"
            self.box_dbl_bl, self.box_dbl_br = "╚", "╝"
            self.box_dbl_h, self.box_dbl_v = "═", "║"
            self.box_dbl_ml, self.box_dbl_mr = "╠", "╣"
            self.box_dbl_tc, self.box_dbl_bc = "╦", "╩"
            self.box_dbl_mc = "╬"

            # Single box chars
            self.box_sng_tl, self.box_sng_tr = "┌", "┐"
            self.box_sng_bl, self.box_sng_br = "└", "┘"
            self.box_sng_h, self.box_sng_v = "─", "│"
            self.box_sng_ml, self.box_sng_mr = "├", "┤"
            self.box_sng_tc, self.box_sng_bc = "┬", "┴"
            self.box_sng_mc = "┼"

            self.sep_v = "│"
        else:
            # ASCII Double box fallback
            self.box_dbl_tl, self.box_dbl_tr = "+", "+"
            self.box_dbl_bl, self.box_dbl_br = "+", "+"
            self.box_dbl_h, self.box_dbl_v = "=", "|"
            self.box_dbl_ml, self.box_dbl_mr = "+", "+"
            self.box_dbl_tc, self.box_dbl_bc = "+", "+"
            self.box_dbl_mc = "+"

            # ASCII Single box fallback
            self.box_sng_tl, self.box_sng_tr = "+", "+"
            self.box_sng_bl, self.box_sng_br = "+", "+"
            self.box_sng_h, self.box_sng_v = "-", "|"
            self.box_sng_ml, self.box_sng_mr = "+", "+"
            self.box_sng_tc, self.box_sng_bc = "+", "+"
            self.box_sng_mc = "+"

            self.sep_v = "|"

        # Theme-specific branding, symbols and colors
        if not use_color:
            self.setup_plain_theme()
        elif self.theme_name == "cyberpunk":
            self.setup_cyberpunk_theme()
        elif self.theme_name == "minimalist":
            self.setup_minimalist_theme()
        elif self.theme_name == "matrix":
            self.setup_matrix_theme()
        elif self.theme_name == "compact":
            self.setup_compact_theme()
        elif self.theme_name == "pro":
            self.setup_pro_theme()
        else:  # classic
            self.setup_classic_theme()

    def setup_plain_theme(self):
        self.logo = "Radiance"
        self.arrow = "->"
        self.chan_prefix = "["
        self.chan_suffix = "]"
        self.chan_color = ""
        self.logo_color = ""
        self.sep_color = ""

        self.lbl_crit = "[CRIT]"
        self.lbl_err = "[ERROR]"
        self.lbl_warn = "[WARN]"
        self.lbl_success = "[SUCCESS]"
        self.lbl_info = "[INFO]"
        self.lbl_debug = "[DEBUG]"

        self.status_ok = "[OK] Active"
        self.status_warn = "[!] Missing"
        self.status_err = "[X] Missing"

    def setup_classic_theme(self):
        # Current layout: Pastel Lavender Logo, Sky Blue/Cyan arrows and badges
        self.logo = f"\033[1;38;5;141m{'✦' if self.use_unicode else '*'} Radiance\033[0m"
        self.arrow = f"\033[38;5;240m{'➔' if self.use_unicode else '->'}\033[0m"
        self.chan_prefix = ""
        self.chan_suffix = ""
        self.chan_color = "\033[38;5;111m"  # soft pastel blue

        # Badges
        self.lbl_crit = "\033[1;38;5;231;48;5;196m[ CRIT ]\033[0m"
        self.lbl_err = "\033[1;38;5;196m[ERROR ]\033[0m"
        self.lbl_warn = "\033[1;38;5;214m[WARN  ]\033[0m"
        self.lbl_success = "\033[1;38;5;78m[SUCCESS]\033[0m"
        self.lbl_info = "\033[1;38;5;75m[INFO  ]\033[0m"
        self.lbl_debug = "\033[38;5;244m[DEBUG ]\033[0m"

        # Table statuses
        self.status_ok = "✓ Active" if self.use_unicode else "[OK] Active"
        self.status_warn = "⚠ Missing" if self.use_unicode else "[!] Missing"
        self.status_err = "✗ Missing" if self.use_unicode else "[X] Missing"

    def setup_cyberpunk_theme(self):
        # Neon purple/magenta, yellow arrows, bright cyan channels
        self.logo = f"\033[1;38;5;201m{'⚡' if self.use_unicode else '*'} RADIANCE\033[0m"
        self.arrow = f"\033[38;5;220m{'»' if self.use_unicode else '>>'}\033[0m"
        self.chan_prefix = "#"
        self.chan_suffix = ""
        self.chan_color = "\033[1;38;5;51m"  # neon cyan

        # Badges
        self.lbl_crit = "\033[1;38;5;196;48;5;234m[☠ CRITICAL]\033[0m"
        self.lbl_err = "\033[1;38;5;196m[✖ ERROR]\033[0m"
        self.lbl_warn = "\033[1;38;5;214m[⚠ WARN ]\033[0m"
        self.lbl_success = "\033[1;38;5;46m[✔ OK   ]\033[0m"  # bright green
        self.lbl_info = "\033[1;38;5;117m[⚡ INFO ]\033[0m"
        self.lbl_debug = "\033[38;5;99m[◈ DEBUG]\033[0m"

        # Table statuses
        self.status_ok = "⚡ Active" if self.use_unicode else "[OK] Active"
        self.status_warn = "⚠ Missing" if self.use_unicode else "[!] Missing"
        self.status_err = "☠ Missing" if self.use_unicode else "[X] Missing"

    def setup_minimalist_theme(self):
        # Clean slate-grey, quiet dot separator, low contrast text
        self.logo = f"\033[38;5;244m{'▪' if self.use_unicode else '-'} radiance\033[0m"
        self.arrow = f"\033[38;5;238m{'·' if self.use_unicode else '.'}\033[0m"
        self.chan_prefix = ""
        self.chan_suffix = ""
        self.chan_color = "\033[38;5;241m"  # dim gray

        # Quiet lowercase badges
        self.lbl_crit = "\033[1;38;5;124m[crit]\033[0m"
        self.lbl_err = "\033[38;5;160m[err ]\033[0m"
        self.lbl_warn = "\033[38;5;178m[warn]\033[0m"
        self.lbl_success = "\033[38;5;65m[ok  ]\033[0m"  # muted sage green
        self.lbl_info = "\033[38;5;244m[info]\033[0m"
        self.lbl_debug = "\033[38;5;238m[dbug]\033[0m"

        # Table statuses
        self.status_ok = "ok"
        self.status_warn = "missing"
        self.status_err = "missing!"

    def setup_matrix_theme(self):
        # Monospaced green terminal, dollar separators, custom symbols
        self.logo = f"\033[1;38;5;46m[radiance]\033[0m"
        self.arrow = f"\033[38;5;28m$\033[0m"
        self.chan_prefix = "::"
        self.chan_suffix = ""
        self.chan_color = "\033[38;5;34m"  # medium green

        # Badges
        self.lbl_crit = "\033[1;38;5;196m[!!!]\033[0m"
        self.lbl_err = "\033[1;38;5;160m[ - ]\033[0m"
        self.lbl_warn = "\033[1;38;5;214m[ ! ]\033[0m"
        self.lbl_success = "\033[1;38;5;46m[ + ]\033[0m"
        self.lbl_info = "\033[38;5;48m[ i ]\033[0m"
        self.lbl_debug = "\033[38;5;22m[ d ]\033[0m"

        # Table statuses
        self.status_ok = "[ + ] Active"
        self.status_warn = "[ ! ] Missing"
        self.status_err = "[ - ] Missing"

    def setup_compact_theme(self):
        # Space-saving layout with gradient violet logo, compact letter badges
        self.logo = f"\033[1;38;5;93mR\033[38;5;75mad\033[0m"
        self.arrow = f"\033[38;5;239m/\033[0m"
        self.chan_prefix = ""
        self.chan_suffix = ""
        self.chan_color = "\033[38;5;175m"  # pastel pink/mauve

        # Compact single-letter level indicators
        self.lbl_crit = "\033[1;38;5;196m[C]\033[0m"
        self.lbl_err = "\033[1;38;5;196m[E]\033[0m"
        self.lbl_warn = "\033[1;38;5;214m[W]\033[0m"
        self.lbl_success = "\033[1;38;5;78m[S]\033[0m"
        self.lbl_info = "\033[1;38;5;75m[I]\033[0m"
        self.lbl_debug = "\033[38;5;244m[D]\033[0m"

        # Table statuses
        self.status_ok = "Active"
        self.status_warn = "Missing"
        self.status_err = "Missing"

    def setup_pro_theme(self):
        # Clean, column-aligned studio theme. The brand is printed ONCE as a
        # session header at startup, so the per-line logo is suppressed and each
        # record is a scannable  time / LEVEL / channel / message  row.
        self.columnar = True
        self.logo = ""
        self.arrow = ""
        self.chan_prefix = ""
        self.chan_suffix = ""
        self.chan_color = "\033[38;5;102m"   # muted slate channel

        # Fixed-width 5-char UPPERCASE level tags, one accent colour each
        self.lbl_crit    = "\033[1;38;5;231;48;5;124mCRIT \033[0m"
        self.lbl_err     = "\033[38;5;167mERROR\033[0m"
        self.lbl_warn    = "\033[38;5;179mWARN \033[0m"
        self.lbl_success = "\033[38;5;108mOK   \033[0m"
        self.lbl_info    = "\033[38;5;110mINFO \033[0m"
        self.lbl_debug   = "\033[38;5;240mDEBUG\033[0m"

        self.status_ok = "ok"
        self.status_warn = "missing"
        self.status_err = "missing!"


class RadianceConsoleFormatter(logging.Formatter):
    """Custom logging formatter that renders premium colored log records.

    Format: [HH:MM:SS] ✦ Radiance ➔ [channel] [LEVEL] message
    """

    def __init__(self, use_color: bool = True, use_unicode: bool = True, theme_name: str = "minimalist"):
        super().__init__()
        self.use_color = use_color
        self.encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        self.theme = TextTheme(theme_name, use_unicode, use_color)

    def format(self, record: logging.LogRecord) -> str:
        # Format the time representation
        t_str = time.strftime("%H:%M:%S", time.localtime(record.created))

        # Extract relative component channel
        orig_name = record.name
        if orig_name.startswith("radiance."):
            channel = orig_name[9:]
        elif orig_name == "radiance":
            channel = ""
        else:
            channel = orig_name

        # Choose theme-based level badge
        theme = self.theme
        if record.levelno >= logging.CRITICAL:
            level_badge = theme.lbl_crit
        elif record.levelno >= logging.ERROR:
            level_badge = theme.lbl_err
        elif record.levelno >= logging.WARNING:
            level_badge = theme.lbl_warn
        elif record.levelno == SUCCESS_LEVEL:
            level_badge = theme.lbl_success
        elif record.levelno >= logging.INFO:
            level_badge = theme.lbl_info
        else:
            level_badge = theme.lbl_debug

        if self.use_color:
            c_reset = "\033[0m"
            c_dim = "\033[90m"

            if getattr(theme, "columnar", False):
                # Columnar studio layout: time  LEVEL  channel  message
                chan_fixed = (channel or "core")[:8].ljust(8)
                time_part = f"{c_dim}{t_str}{c_reset}"
                prefix = f"{time_part}  {level_badge}  {theme.chan_color}{chan_fixed}{c_reset}"
            else:
                # Construct channel segment
                if channel:
                    chan_str = f"{theme.chan_prefix}{channel}{theme.chan_suffix}"
                    chan_part = f" {theme.arrow} {theme.chan_color}{chan_str}{c_reset}"
                else:
                    chan_part = ""

                level_part = f" {theme.arrow} {level_badge}"
                time_part = f"{c_dim}[{t_str}]{c_reset}"
                prefix = f"{time_part} {theme.logo}{chan_part}{level_part}"

            msg = record.getMessage()

            # Format multi-line logs cleanly, maintaining prefixes on each line
            if "\n" in msg:
                formatted_lines = [f"{prefix} {line}" for line in msg.splitlines()]
                msg_part = "\n".join(formatted_lines)
            else:
                msg_part = f"{prefix} {msg}"

            # Format any exception traceback
            if record.exc_info:
                exc_text = self.formatException(record.exc_info)
                formatted_exc = [f"{prefix} \033[31m{theme.sep_v}\033[0m {line}" for line in exc_text.splitlines()]
                msg_part += "\n" + "\n".join(formatted_exc)

            return msg_part.encode(self.encoding, errors="backslashreplace").decode(self.encoding)
        else:
            # Degraded plain text formatting
            chan_part = f" [{channel}]" if channel else ""
            msg = record.getMessage()
            formatted_msg = f"[{t_str}] [Radiance]{chan_part} {level_badge} {msg}"
            if record.exc_info:
                exc_text = self.formatException(record.exc_info)
                formatted_msg += "\n" + exc_text
            return formatted_msg.encode(self.encoding, errors="backslashreplace").decode(self.encoding)


class ThrottleDedupeFilter(logging.Filter):
    """Collapses identical, back-to-back log lines into a single line + rollup.

    The Radiance pipeline can emit the exact same diagnostic many times in a
    burst (e.g. an HDR-decode warning fired once per frame, or a per-step note).
    This filter lets the first occurrence through, silently suppresses identical
    repeats that arrive within ``window`` seconds, and — when a different line
    finally arrives — prints one dim "previous line repeated xN" summary so
    nothing is hidden without trace.

    It only ever compares a record to the one immediately before it, so distinct
    interleaved messages are never lost; only true consecutive duplicates fold.
    """

    def __init__(self, window: float = 3.0, use_color: bool = True, use_unicode: bool = True):
        super().__init__()
        self.window = window
        self.use_color = use_color
        self.use_unicode = use_unicode
        self._last_key = None
        self._last_time = 0.0
        self._dup = 0

    def _emit_rollup(self) -> None:
        if self._dup <= 0:
            return
        n = self._dup + 1  # first occurrence + suppressed repeats
        arrow = "↳" if self.use_unicode else "->"
        text = f"  {arrow} previous line repeated ×{n}" if self.use_unicode \
            else f"  {arrow} previous line repeated x{n}"
        try:
            if self.use_color:
                sys.stdout.write(f"\033[38;5;240m{text}\033[0m\n")
            else:
                sys.stdout.write(text + "\n")
            sys.stdout.flush()
        except Exception:
            pass
        self._dup = 0

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            key = (record.name, record.levelno, record.getMessage())
        except Exception:
            return True
        now = time.monotonic()
        if key == self._last_key and (now - self._last_time) <= self.window:
            self._dup += 1
            self._last_time = now
            return False  # suppress this duplicate
        # A new, distinct line: flush any pending rollup for the prior burst.
        self._emit_rollup()
        self._last_key = key
        self._last_time = now
        return True


def resolve_log_level(default: int = logging.INFO) -> int:
    """The level named by RADIANCE_LOG_LEVEL, or *default*.

    The startup shortfall error instructs users to "re-run with
    RADIANCE_LOG_LEVEL=DEBUG for tracebacks". Nothing read that variable, so
    the advice did nothing — the exact kind of control this package has a test
    suite for. Accepts a level name or a numeric level; anything unrecognised
    falls back rather than raising during import.
    """
    raw = os.environ.get("RADIANCE_LOG_LEVEL", "").strip()
    if not raw:
        return default

    named = logging.getLevelName(raw.upper())
    if isinstance(named, int):
        return named
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def setup_radiance_logging(level: "int | None" = None) -> logging.Logger:
    """Configures the main 'radiance' logger tree with the custom premium console formatter.

    Clears any pre-existing handlers, activates Windows VT100 console colors if needed,
    and sets propagate=False to prevent duplicate logging inside ComfyUI.

    *level* defaults to RADIANCE_LOG_LEVEL, then INFO.
    """
    if level is None:
        level = resolve_log_level()

    logger = logging.getLogger("radiance")
    logger.setLevel(level)

    # Prevent duplicates from propagating to root logger configured by ComfyUI
    logger.propagate = False

    # Remove any existing handlers
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    # Initialize terminal capability
    init_windows_ansi()
    color_enabled = supports_color()
    unicode_enabled = supports_unicode()
    theme_name = os.environ.get("RADIANCE_LOG_THEME", "pro")

    # Create stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(
        RadianceConsoleFormatter(
            use_color=color_enabled,
            use_unicode=unicode_enabled,
            theme_name=theme_name
        )
    )
    stream_handler.setLevel(level)

    # Collapse back-to-back identical lines (e.g. an HDR-decode warning fired
    # once per frame) into one line + a dim "repeated xN" rollup.
    stream_handler.addFilter(
        ThrottleDedupeFilter(
            window=3.0,
            use_color=color_enabled,
            use_unicode=unicode_enabled,
        )
    )

    logger.addHandler(stream_handler)

    # Pro theme prints the brand once as a session header (keeps lines clean).
    if theme_name == "pro" and color_enabled:
        mark = "\u25ce" if unicode_enabled else "o"
        rule = ("\u00b7" * 50) if unicode_enabled else ("." * 50)
        # decode the unicode escapes we stored as literals above
        mark = mark.encode().decode("unicode_escape") if mark.startswith("\\u") else mark
        rule = rule.encode().decode("unicode_escape") if "\\u" in rule else rule
        sess = time.strftime("%H:%M")
        sys.stdout.write(f"\033[38;5;179m{mark} radiance\033[0m \033[38;5;240m{rule}\033[0m \033[38;5;102msession {sess}\033[0m\n")
        sys.stdout.flush()

    return logger


def print_box(
    title: str,
    lines: Sequence[str],
    style: str = "double",
    level: int = logging.INFO,
    logger_name: str = "radiance",
):
    """Logs a list of text strings enclosed inside a gorgeous Unicode box.

    Supports 'double' and 'single' border styles, and gracefully falls back to ASCII on CP1252.
    """
    if not lines:
        return

    unicode_enabled = supports_unicode()
    theme_name = os.environ.get("RADIANCE_LOG_THEME", "pro")
    theme = TextTheme(theme_name, unicode_enabled, supports_color())

    # Calculate padding based on internal content length
    max_len = max(len(line) for line in lines)
    max_len = max(max_len, len(title) + 4)

    if style == "double":
        tl, tr, bl, br = theme.box_dbl_tl, theme.box_dbl_tr, theme.box_dbl_bl, theme.box_dbl_br
        h, v = theme.box_dbl_h, theme.box_dbl_v
        ml, mr = theme.box_dbl_ml, theme.box_dbl_mr
    else:
        tl, tr, bl, br = theme.box_sng_tl, theme.box_sng_tr, theme.box_sng_bl, theme.box_sng_br
        h, v = theme.box_sng_h, theme.box_sng_v
        ml, mr = theme.box_sng_ml, theme.box_sng_mr

    box_lines = []

    # Title segment
    title_padded = f" {title} ".center(max_len + 2, h)
    box_lines.append(tl + h * (max_len + 2) + tr)
    box_lines.append(v + title_padded + v)
    box_lines.append(ml + h * (max_len + 2) + mr)

    # Body lines
    for line in lines:
        padded_line = line.ljust(max_len)
        box_lines.append(f"{v}  {padded_line}  {v}")

    # Bottom border
    box_lines.append(bl + h * (max_len + 2) + br)

    # Output using designated logger
    logger = logging.getLogger(logger_name)
    logger.log(level, "\n".join(box_lines))


def print_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    col_alignments: Sequence[str] | None = None,
    level: int = logging.INFO,
    logger_name: str = "radiance",
):
    """Renders a fully-padded, gorgeous ASCII/Unicode table with borders.

    Correctly calculates column alignment widths by stripping ANSI sequences
    so that terminal coloring inside tables does not warp cells.
    """
    if not headers:
        return

    unicode_enabled = supports_unicode()
    theme_name = os.environ.get("RADIANCE_LOG_THEME", "pro")
    theme = TextTheme(theme_name, unicode_enabled, supports_color())

    num_cols = len(headers)
    col_widths = [len(h) for h in headers]

    # Convert values to string and calculate true display lengths (sans ANSI escapes)
    string_rows = []
    ansi_regex = re.compile(r"\033\[[0-9;]*m")

    for row in rows:
        str_row = []
        for i, cell in enumerate(row):
            s_cell = str(cell)
            clean_cell = ansi_regex.sub("", s_cell)
            cell_len = len(clean_cell)
            str_row.append((s_cell, cell_len))
            if i < num_cols:
                col_widths[i] = max(col_widths[i], cell_len)
        string_rows.append(str_row)

    if col_alignments is None:
        col_alignments = ["left"] * num_cols

    # Table character sets
    tl, tr, bl, br = theme.box_sng_tl, theme.box_sng_tr, theme.box_sng_bl, theme.box_sng_br
    h, v = theme.box_sng_h, theme.box_sng_v
    ml, mr = theme.box_sng_ml, theme.box_sng_mr
    tc, bc, mc = theme.box_sng_tc, theme.box_sng_bc, theme.box_sng_mc

    table_lines = []

    # Top border
    top_border = tl + tc.join(h * (w + 2) for w in col_widths) + tr
    table_lines.append(top_border)

    # Header Row
    header_cells = []
    for i, h_text in enumerate(headers):
        padded = h_text.center(col_widths[i])
        header_cells.append(f" {padded} ")
    table_lines.append(v + v.join(header_cells) + v)

    # Separator Line
    sep_row = ml + mc.join(h * (w + 2) for w in col_widths) + mr
    table_lines.append(sep_row)

    # Data Rows
    for row in string_rows:
        row_cells = []
        for i, (cell_text, clean_len) in enumerate(row):
            if i >= num_cols:
                break
            padding_len = col_widths[i] - clean_len
            align = col_alignments[i] if i < len(col_alignments) else "left"

            if align == "right":
                padded = " " * padding_len + cell_text
            elif align == "center":
                left_pad = padding_len // 2
                right_pad = padding_len - left_pad
                padded = " " * left_pad + cell_text + " " * right_pad
            else:  # left
                padded = cell_text + " " * padding_len

            row_cells.append(f" {padded} ")
        table_lines.append(v + v.join(row_cells) + v)

    # Bottom border
    bottom_border = bl + bc.join(h * (w + 2) for w in col_widths) + br
    table_lines.append(bottom_border)

    # Output using designated logger
    logger = logging.getLogger(logger_name)
    logger.log(level, "\n".join(table_lines))


def print_premium_loader_hud(
    preset: str,
    overrides: Sequence[str] | None,
    resolved_type: str,
    latent_fmt: str,
    est_vram: float,
    avail_vram: float,
    total_vram: float,
    unet_name: str,
    unet_dtype: str,
    unet_time: float,
    unet_cached: bool,
    clip_slots: Sequence[str],
    clip_dtype: str,
    clip_time: float,
    clip_cached: bool,
    vae_name: str,
    vae_time: float,
    vae_cached: bool,
    loras: Sequence[dict] | None,
    total_time_ms: float,
    caching: bool
) -> None:
    """Renders a next-generation visual terminal HUD dashboard showing model loader metrics.

    Features dynamic color VRAM usage bars, structured diagnostic grids, and auto-truncating table column mappings
    that scale perfectly inside a double-bordered container.
    """
    if os.environ.get("RADIANCE_LOG_THEME", "pro") == "pro":
        _print_loader_summary_pro(
            preset, overrides, resolved_type, latent_fmt, est_vram, avail_vram,
            total_vram, unet_name, unet_dtype, unet_time, clip_slots, clip_dtype,
            clip_time, vae_name, vae_time, loras, total_time_ms, caching)
        return

    from radiance.model.cache import _unet_cache, _clip_cache, _vae_cache
    use_color = supports_color()
    use_unicode = supports_unicode()

    # Color tokens
    C_RESET   = "\033[0m" if use_color else ""
    C_BORDER  = "\033[38;5;239m" if use_color else ""  # deep space gray
    C_HEADER  = "\033[1;38;5;231;48;5;237m" if use_color else "" # metallic silver pill
    C_TITLE   = "\033[1;38;5;141m" if use_color else "" # lavender title accent
    C_VAL     = "\033[38;5;75m" if use_color else ""   # sky blue values
    C_DIM     = "\033[90m" if use_color else ""        # dim gray
    C_SUCCESS = "\033[38;5;78m" if use_color else ""   # muted sage green
    C_WARN    = "\033[1;38;5;178m" if use_color else "" # warm amber yellow
    C_ERR     = "\033[1;38;5;160m" if use_color else "" # crimson red
    C_BOLD    = "\033[1m" if use_color else ""

    # Box-drawing elements
    if use_unicode:
        # Outer Double Border
        box_tl, box_tr = "╔", "╗"
        box_bl, box_br = "╚", "╝"
        box_h, box_v   = "═", "║"
        box_m, box_mr   = "╠", "╣"

        # Inner Table Borders
        tbl_tl, tbl_tr = "┌", "┐"
        tbl_bl, tbl_br = "└", "┘"
        tbl_h, tbl_v   = "─", "│"
        tbl_ml, tbl_mr = "├", "┤"
        tbl_tc, tbl_bc = "┬", "┴"
        tbl_mc         = "┼"

        bullet = "●"
        chk    = "✔"
        spark  = "⚡"
        warn_icon = "⚠"
        health_optimal = "🟢 OPTIMAL"
        health_tight   = "🟡 MEMORY TIGHT"
        lbl_cached     = "🟢 CACHED"
        lbl_disk       = "🟢 DISK"
    else:
        # ASCII Fallbacks
        box_tl, box_tr = "+", "+"
        box_bl, box_br = "+", "+"
        box_h, box_v   = "=", "|"
        box_m, box_mr   = "+", "+"

        tbl_tl, tbl_tr = "+", "+"
        tbl_bl, tbl_br = "+", "+"
        tbl_h, tbl_v   = "-", "|"
        tbl_ml, tbl_mr = "+", "+"
        tbl_tc, tbl_bc = "+", "+"
        tbl_mc         = "+"

        bullet = "*"
        chk    = "[x]"
        spark  = "v"
        warn_icon = "[!]"
        health_optimal = "OPTIMAL"
        health_tight   = "MEM TIGHT"
        lbl_cached     = "CACHED"
        lbl_disk       = "DISK"

    # Box dimensions
    width = 76
    lines = []
    
    # Calculate spacing (stripping ANSI codes)
    ansi_escape = re.compile(r"\033\[[0-9;]*m")
    def display_len(s: str) -> int:
        return len(ansi_escape.sub("", s))

    def wrap_line(content: str) -> str:
        d_len = display_len(content)
        padding = width - d_len - 4
        return f"{C_BORDER}{box_v}{C_RESET}  {content}" + " " * padding + f"  {C_BORDER}{box_v}{C_RESET}"

    # 1. Top border
    lines.append(f"{C_BORDER}{box_tl}{box_h * width}{box_tr}{C_RESET}")

    # 2. Header
    title_text = f" {C_BOLD} R A D I A N C E   S Y S T E M   L O A D E R{C_RESET} "
    header_padded = title_text.center(width)
    lines.append(f"{C_BORDER}{box_v}{C_RESET}{C_HEADER}{header_padded}{C_RESET}{C_BORDER}{box_v}{C_RESET}")
    
    # 3. Double separator
    lines.append(f"{C_BORDER}{box_m}{box_h * width}{box_mr}{C_RESET}")

    # 4. Engine Health & Cache status status row
    is_tight = avail_vram > 0 and est_vram > avail_vram * 0.9
    health_pill = f"{C_WARN}{health_tight}{C_RESET}" if is_tight else f"{C_SUCCESS}{health_optimal}{C_RESET}"
    cache_pill = f"{C_SUCCESS}{spark} ON (U{_unet_cache.size} C{_clip_cache.size} V{_vae_cache.size}){C_RESET}" if caching else f"{C_DIM}OFF{C_RESET}"

    row_health = f"{C_DIM}{bullet} ENGINE HEALTH :{C_RESET} {health_pill}        {C_DIM}{bullet} CACHE ENGINE :{C_RESET} {cache_pill}"
    lines.append(wrap_line(row_health))
    lines.append(f"{C_BORDER}{box_v}{C_RESET}" + " " * width + f"{C_BORDER}{box_v}{C_RESET}")

    # 5. Environment Specifications
    lines.append(wrap_line(f"{C_TITLE}[ ENVIRONMENT SPECIFICATIONS ]{C_RESET}"))
    lines.append(wrap_line(f"Preset Name   : {C_VAL}{preset}{C_RESET}" + (f" {C_DIM}(overrode: {', '.join(overrides)}){C_RESET}" if overrides else "")))
    lines.append(wrap_line(f"Architecture  : {C_VAL}{resolved_type}{C_RESET} {C_DIM}│{C_RESET} Latent Format: {C_VAL}{latent_fmt}{C_RESET}"))
    lines.append(f"{C_BORDER}{box_v}{C_RESET}" + " " * width + f"{C_BORDER}{box_v}{C_RESET}")

    # 6. GPU VRAM Specification with dynamic colored gauge
    lines.append(wrap_line(f"{C_TITLE}[ GPU VRAM SPECIFICATION ]{C_RESET}"))
    lines.append(wrap_line(f"Estimated VRAM Need  : {C_VAL}~{est_vram:.2f} GB{C_RESET}"))

    vram_percent = (avail_vram / total_vram) * 100.0 if total_vram > 0 else 0.0
    bar_width = 16
    filled = int(round((vram_percent / 100.0) * bar_width))
    bar_char = "■" if use_unicode else "#"
    empty_char = "□" if use_unicode else "-"
    
    if use_color:
        if vram_percent > 50:
            c_bar = "\033[38;5;78m"   # green
        elif vram_percent > 20:
            c_bar = "\033[38;5;178m"  # yellow
        else:
            c_bar = "\033[38;5;160m"  # red
        vram_bar = f"{C_DIM}[{C_RESET}{c_bar}{bar_char * filled}{C_RESET}{C_DIM}{empty_char * (bar_width - filled)}]{C_RESET} {C_VAL}{vram_percent:.1f}% Free{C_RESET}"
    else:
        vram_bar = f"[{bar_char * filled}{empty_char * (bar_width - filled)}] {vram_percent:.1f}% Free"

    lines.append(wrap_line(f"Available VRAM Space : {C_VAL}{avail_vram:.2f} GB{C_RESET} / {C_VAL}{total_vram:.2f} GB{C_RESET}  {vram_bar}"))
    
    if is_tight:
        lines.append(wrap_line(f"{C_WARN}{warn_icon} Memory Tight Warning : {C_WARN}VRAM tight! Consider fp8 dtype or cpu_offload.{C_RESET}"))
    lines.append(f"{C_BORDER}{box_v}{C_RESET}" + " " * width + f"{C_BORDER}{box_v}{C_RESET}")

    # 7. Pipeline Component Matrix Grid
    lines.append(wrap_line(f"{C_TITLE}[ PIPELINE COMPONENT MATRIX ]{C_RESET}"))
    
    # Table headers
    # Total width of inner columns is 68 characters to fit perfectly inside the HUD
    # Cols: Status(10), Component(28), Type(10), Dtype(10), Time(8)
    col_status = "Status".ljust(10)
    col_name = "Model Component / Path".ljust(28)
    col_type = "Category".ljust(10)
    col_dtype = "Precision".ljust(10)
    col_time = "Time".ljust(6)

    t_header = f"{C_DIM}{col_status}│ {col_name}│ {col_type}│ {col_dtype}│ {col_time}{C_RESET}"
    lines.append(wrap_line(f"{C_BORDER}{tbl_tl}{tbl_h * 11}{tbl_tc}{tbl_h * 29}{tbl_tc}{tbl_h * 11}{tbl_tc}{tbl_h * 11}{tbl_tc}{tbl_h * 7}{tbl_tr}{C_RESET}"))
    lines.append(wrap_line(f"{C_BORDER}{tbl_v}{C_RESET} {t_header} {C_BORDER}{tbl_v}{C_RESET}"))
    lines.append(wrap_line(f"{C_BORDER}{tbl_ml}{tbl_h * 11}{tbl_mc}{tbl_h * 29}{tbl_mc}{tbl_h * 11}{tbl_mc}{tbl_h * 11}{tbl_mc}{tbl_h * 7}{tbl_mr}{C_RESET}"))

    def format_row(cached: bool, filename: str, category: str, dtype: str | None, elapsed: float) -> str:
        stat = f"{C_SUCCESS}{lbl_cached}{C_RESET}" if cached else f"{C_VAL}{lbl_disk}{C_RESET}"
        
        # Truncate model name if too long
        max_filename_len = 26
        if len(filename) > max_filename_len:
            display_name = filename[:max_filename_len-3] + "..."
        else:
            display_name = filename
            
        c_stat = stat.ljust(10)
        c_name = f"{C_BOLD}{display_name}{C_RESET}".ljust(28)
        c_type = category.ljust(10)
        c_dtype = (dtype if dtype else "default").ljust(10)
        c_time = f"{elapsed:.1f}s".ljust(6) if elapsed > 0 else f"{C_DIM}cached{C_RESET}".ljust(6)
        
        return f"{C_BORDER}{tbl_v}{C_RESET} {c_stat} {C_BORDER}{tbl_v}{C_RESET} {c_name} {C_BORDER}{tbl_v}{C_RESET} {c_type} {C_BORDER}{tbl_v}{C_RESET} {c_dtype} {C_BORDER}{tbl_v}{C_RESET} {c_time} {C_BORDER}{tbl_v}{C_RESET}"

    # UNET
    lines.append(wrap_line(format_row(unet_cached, unet_name, "UNET", unet_dtype, unet_time)))
    
    # CLIP
    clip_display_name = " + ".join(clip_slots)
    lines.append(wrap_line(format_row(clip_cached, clip_display_name, "CLIP", clip_dtype, clip_time)))

    # VAE
    lines.append(wrap_line(format_row(vae_cached, vae_name, "VAE", None, vae_time)))

    # LoRAs if applied
    if loras:
        for l in loras:
            lora_dtype = f"m={l['model_str']} c={l['clip_str']}"
            lines.append(wrap_line(format_row(False, l["name"], "LoRA", lora_dtype, 0.0)))

    lines.append(wrap_line(f"{C_BORDER}{tbl_bl}{tbl_h * 11}{tbl_bc}{tbl_h * 29}{tbl_bc}{tbl_h * 11}{tbl_bc}{tbl_h * 11}{tbl_bc}{tbl_h * 7}{tbl_br}{C_RESET}"))
    lines.append(f"{C_BORDER}{box_v}{C_RESET}" + " " * width + f"{C_BORDER}{box_v}{C_RESET}")

    # 8. Success Status & Finish Banner
    lines.append(f"{C_BORDER}{box_m}{box_h * width}{box_mr}{C_RESET}")
    
    success_text = f"{C_SUCCESS}{chk} Boot completed in {total_time_ms / 1000:.2f}s — Pipeline ready for inference.{C_RESET}"
    lines.append(wrap_line(success_text))
    
    # 9. Bottom border
    lines.append(f"{C_BORDER}{box_bl}{box_h * width}{box_br}{C_RESET}")

    # Print out as a single beautiful atomic write
    sys.stdout.write("\n" + "\n".join(lines) + "\n\n")
    sys.stdout.flush()


def _print_loader_summary_pro(preset, overrides, resolved_type, latent_fmt,
                              est_vram, avail_vram, total_vram, unet_name, unet_dtype,
                              unet_time, clip_slots, clip_dtype, clip_time, vae_name,
                              vae_time, loras, total_time_ms, caching) -> None:
    """Clean, column-aligned loader summary for the 'pro' theme.

    Replaces the ornate double-bordered HUD with a calm card: aligned key/value
    rows, a single block VRAM bar, no emoji, no heavy box art.
    """
    use_color = supports_color()
    use_unicode = supports_unicode()
    R     = "\033[0m"          if use_color else ""
    DIM   = "\033[38;5;240m"   if use_color else ""
    SLATE = "\033[38;5;102m"   if use_color else ""
    GOLD  = "\033[38;5;179m"   if use_color else ""
    GREEN = "\033[38;5;108m"   if use_color else ""
    AMBER = "\033[38;5;179m"   if use_color else ""
    VAL   = "\033[38;5;110m"   if use_color else ""

    mark = "\u25ce" if use_unicode else "o"
    rule = ("\u2500" * 46) if use_unicode else ("-" * 46)

    used = max(total_vram - avail_vram, 0.0)
    pct = (used / total_vram) if total_vram > 0 else 0.0
    barw = 18
    fill = max(0, min(barw, int(round(pct * barw))))
    if use_unicode:
        bar = ("\u2588" * fill) + ("\u2591" * (barw - fill))
    else:
        bar = ("#" * fill) + ("-" * (barw - fill))
    tight = avail_vram > 0 and est_vram > avail_vram * 0.9
    bar_color = AMBER if tight else GREEN

    ovr = f" {DIM}(overrides: {', '.join(overrides)}){R}" if overrides else ""
    cache_str = f"{GREEN}on{R}" if caching else f"{DIM}off{R}"

    def row(label, name, dtype, t):
        name = (name or "")[:34]
        tcol = f"{DIM}{t:.1f}s{R}" if (t and t > 0) else f"{DIM}cached{R}"
        dt = f"{DIM}{dtype}{R}" if dtype else ""
        return f"  {label:<6} {name:<34} {dt}  {tcol}"

    out = []
    out.append(f"{GOLD}{mark} radiance loader{R}  {DIM}{resolved_type} \u00b7 {latent_fmt}{R}")
    out.append(f"{DIM}{rule}{R}")
    out.append(f"  preset   {VAL}{preset}{R}{ovr}")
    out.append(f"  cache    {cache_str:<14} vram  {bar_color}{bar}{R} {VAL}{used:.1f}{R}/{total_vram:.0f} GB")
    out.append(f"{DIM}{rule}{R}")
    out.append(row("unet", unet_name, unet_dtype, unet_time))
    out.append(row("clip", " + ".join(clip_slots) if clip_slots else "", clip_dtype, clip_time))
    out.append(row("vae", vae_name, None, vae_time))
    if loras:
        for l in loras:
            out.append(row("lora", l.get("name", "lora"), f"m={l.get('model_str','')} c={l.get('clip_str','')}", 0.0))
    out.append(f"{DIM}{rule}{R}")
    out.append(f"  {GREEN}ready{R} in {VAL}{total_time_ms / 1000:.1f}s{R}")

    sys.stdout.write("\n" + "\n".join(out) + "\n\n")
    sys.stdout.flush()


_run_counter = {"n": 0}


def print_run_banner() -> None:
    """Print a thin separator that marks the start of a new prompt run.

    Gives the console a per-run rhythm: each queued generation opens with a
    quiet rule + index instead of every node's output running together. Safe to
    call on any theme; no-ops gracefully if stdout is unavailable.
    """
    try:
        use_color = supports_color()
        use_unicode = supports_unicode()
        _run_counter["n"] += 1
        idx = _run_counter["n"]
        stamp = time.strftime("%H:%M:%S")
        rule = ("─" * 12) if use_unicode else ("-" * 12)
        mark = "◎" if use_unicode else "o"
        if use_color:
            line = (f"\n\033[38;5;179m{mark}\033[0m \033[38;5;240m{rule}\033[0m "
                    f"\033[38;5;110mrun {idx}\033[0m \033[38;5;240m{rule}\033[0m "
                    f"\033[38;5;240m{stamp}\033[0m\n")
        else:
            line = f"\n{mark} {rule} run {idx} {rule} {stamp}\n"
        sys.stdout.write(line)
        sys.stdout.flush()
    except Exception:
        pass


def register_run_grouping() -> bool:
    """Hook ComfyUI's prompt queue so each run prints a separator banner.

    Uses the stable ``PromptServer.add_on_prompt_handler`` API. Returns True if
    the handler was registered. Fails closed (returns False) on any ComfyUI
    build that does not expose the hook, so package load is never affected.
    """
    try:
        from server import PromptServer  # type: ignore

        instance = getattr(PromptServer, "instance", None)
        if instance is None or not hasattr(instance, "add_on_prompt_handler"):
            return False

        def _on_prompt(json_data):
            print_run_banner()
            return json_data

        instance.add_on_prompt_handler(_on_prompt)
        return True
    except Exception:
        return False
