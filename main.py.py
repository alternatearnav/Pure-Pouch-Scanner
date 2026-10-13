import os
import time
import statistics
import threading
import traceback
from io import BytesIO

from kivy.app import App
from kivy.base import ExceptionHandler, ExceptionManager
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.graphics import Color, Rectangle
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.utils import platform

# ---- USB libraries: MUST be imported here, on the main thread ----
USB_IMPORT_ERROR = None
usb = None
serial4a = None
if platform == "android":
    try:
        from usb4a import usb
        from usbserial4a import serial4a
    except Exception as _e:
        USB_IMPORT_ERROR = "%s: %s" % (type(_e).__name__, _e)

BAUD = 9600
DESKTOP_PORT = "COM256"      # only used when testing on a PC
MATCH_THRESHOLD = 50         # percent of points that must fall within the band
READ_TIMEOUT_SEC = 60
PERMISSION_WAIT_SEC = 40
ERR_LEN = 350

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def res(name):
    return os.path.join(BASE_DIR, name)


# "training_file" is just the file NAME; it is located automatically
PRODUCTS = {
    "500ml": {"title": "Amul Milk (500 ml)", "label": "500 ml",
              "image": res("amul500.png"), "training_file": "footprints.txt"},
    "250ml": {"title": "Amul Milk (250 ml)", "label": "250 ml",
              "image": res("amul250.png"), "training_file": "10 amulmilk.txt"},
    "can":   {"title": "Milk Can", "label": "Milk Can",
              "image": res("milkcan.png"), "training_file": "main_milk.txt"},
}

RED = (1, 0.23, 0.19, 1)
GREEN = (0, 0.63, 0, 1)
PURE_RED = (1, 0, 0, 1)
GREY = (0.56, 0.56, 0.58, 1)


# ==========================================
# Crash protection: errors are shown, never fatal
# ==========================================
class SafeHandler(ExceptionHandler):
    def handle_exception(self, inst):
        traceback.print_exc()
        try:
            App.get_running_app().show_error(repr(inst))
        except Exception:
            pass
        return ExceptionManager.PASS


ExceptionManager.add_handler(SafeHandler())


# ==========================================
# Finding the training files (READ-ONLY)
# ==========================================
def _norm(s):
    """'10 amulmilk.txt' / '10_AmulMilk.TXT' -> '10amulmilktxt'"""
    return "".join(c for c in s.lower() if c.isalnum())


def _search_roots():
    roots = [BASE_DIR]
    try:
        cwd = os.getcwd()
        if cwd not in roots:
            roots.append(cwd)
    except Exception:
        pass
    return roots


def list_txt_files():
    found = []
    for base in _search_roots():
        for root, dirs, files in os.walk(base):
            for f in files:
                if f.lower().endswith(".txt") and f not in found:
                    found.append(f)
    return found


def find_file(name):
    for base in _search_roots():
        exact = os.path.join(base, name)
        if os.path.isfile(exact):
            return exact
    want = _norm(name)
    for base in _search_roots():
        for root, dirs, files in os.walk(base):
            for f in files:
                if _norm(f) == want:
                    return os.path.join(root, f)
    return None


# ==========================================
# Scan processing (pure Python)
# ==========================================
HALF = 5
SG = [-36, 9, 44, 69, 84, 89, 84, 69, 44, 9, -36]   # Savitzky-Golay, window 11, order 3
SG_DEN = 429.0


def _pad(a):
    return [a[0]] * HALF + list(a) + [a[-1]] * HALF


def smooth_scan(values):
    a = [float(v) for v in values]
    n = len(a)
    p = _pad(a)
    med = [sorted(p[i:i + 11])[HALF] for i in range(n)]
    p2 = _pad(med)
    out = []
    for i in range(n):
        s = 0.0
        for j in range(11):
            s += SG[j] * p2[i + j]
        out.append(s / SG_DEN)
    return out


def normalize_scan(values):
    mn, mx = min(values), max(values)
    rng = (mx - mn) if mx > mn else 1.0
    return [(v - mn) * 1023.0 / rng for v in values]


def process_scan(values):
    return normalize_scan(smooth_scan(values))


def load_training(name):
    """Find the training file and read its FOOTPRINT lines.
    The file is opened READ-ONLY and is never modified."""
    path = find_file(name)
    if path is None:
        txts = list_txt_files()
        if txts:
            raise FileNotFoundError(
                "Training file '%s' not found in the app. .txt files inside the "
                "app: %s" % (name, ", ".join(sorted(txts))))
        raise FileNotFoundError(
            "No .txt files are inside the app. Put the training files next to "
            "main.py, add 'txt' to source.include_exts in buildozer.spec, rebuild.")

    footprints = []
    total_lines = 0
    first_line = ""
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            if not first_line:
                first_line = line[:60]
            if not line.upper().startswith("FOOTPRINT"):
                continue
            parts = [p.strip() for p in line.split(",")[1:]]
            try:
                nums = [int(float(p)) for p in parts if p]
            except ValueError:
                continue
            if nums:
                footprints.append(nums)

    if not footprints:
        raise ValueError(
            "'%s' has no usable FOOTPRINT lines (%d lines read; first line: '%s')"
            % (os.path.basename(path), total_lines, first_line))

    # keep only scans that have the same number of points as the first one
    points = len(footprints[0])
    footprints = [fp for fp in footprints if len(fp) == points]
    return footprints


def compute_reference(footprints):
    processed = [process_scan(fp) for fp in footprints]
    median, deviation = [], []
    for col in zip(*processed):
        median.append(statistics.median(col))
        deviation.append(statistics.pstdev(col))
    return median, deviation


def score_scan(values, median, deviation):
    processed = process_scan(values)
    good = 0
    for v, m, d in zip(processed, median, deviation):
        if m - d <= v <= m + d:
            good += 1
    return good / len(processed) * 100.0


# ==========================================
# UI helpers
# ==========================================
def paint_bg(widget, rgba):
    with widget.canvas.before:
        Color(*rgba)
        rect = Rectangle(pos=widget.pos, size=widget.size)
    widget.bind(pos=lambda w, v: setattr(rect, "pos", v),
                size=lambda w, v: setattr(rect, "size", v))


def load_texture(path, max_side=400):
    """Shrink the image with Pillow so huge PNGs can't crash the GPU."""
    try:
        from PIL import Image as PILImage
        im = PILImage.open(path).convert("RGBA")
        im.thumbnail((max_side, max_side))
        buf = BytesIO()
        im.save(buf, format="png")
        buf.seek(0)
        return CoreImage(buf, ext="png").texture
    except Exception:
        traceback.print_exc()
        return None


def make_error_label(height=110):
    lbl = Label(text="", size_hint_y=None, height=height, font_size="13sp",
                color=(0.45, 0.45, 0.47, 1), halign="center", valign="middle")
    lbl.bind(size=lambda w, s: setattr(w, "text_size", (s[0], s[1])))
    return lbl


class Card(ButtonBehavior, BoxLayout):
    def __init__(self, product_key, **kw):
        super().__init__(orientation="vertical", padding=10, spacing=5, **kw)
        self.product_key = product_key
        paint_bg(self, (1, 1, 1, 1))
        prod = PRODUCTS[product_key]
        tex = load_texture(prod["image"])
        if tex is not None:
            self.add_widget(Image(texture=tex))
        else:
            self.add_widget(Label(text="(image missing)", color=GREY))
        self.add_widget(Label(text=prod["label"], size_hint_y=None, height=50,
                              bold=True, font_size="18sp", color=(0.23, 0.23, 0.24, 1)))


class HomeScreen(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        root = BoxLayout(orientation="vertical", padding=15, spacing=15)
        paint_bg(root, (0.96, 0.96, 0.97, 1))
        root.add_widget(Label(text="Choose the Milk Packet", bold=True, font_size="24sp",
                              size_hint_y=None, height=70, color=(0.11, 0.11, 0.12, 1)))
        row = BoxLayout(spacing=12)
        for key in PRODUCTS:
            card = Card(key)
            card.bind(on_release=lambda c: App.get_running_app().start_test(c.product_key))
            row.add_widget(card)
        root.add_widget(row)
        self.msg = make_error_label(70)
        root.add_widget(self.msg)
        self.add_widget(root)


class BigStatus(Label):
    """Text scaled to fill the whole box."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.bold = True
        self.halign = "center"
        self.valign = "middle"
        self.bind(size=self.fit, text=self.fit)

    def fit(self, *a):
        w, h = self.size
        if w < 50 or h < 50:
            return
        self.text_size = (w, h)
        n = max(len(self.text), 1)
        self.font_size = int(max(20, min(h * 0.5, w * 1.6 / n)))


def describe_devices(devices):
    out = []
    for d in devices:
        try:
            out.append("%s (VID %04x PID %04x)" % (
                d.getDeviceName(), d.getVendorId(), d.getProductId()))
        except Exception:
            out.append(str(d))
    return "; ".join(out)


class ResultScreen(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.current_thread = None
        self.cancel_event = threading.Event()
        self.test_id = 0

        root = BoxLayout(orientation="vertical", padding=15, spacing=12)
        paint_bg(root, (0.96, 0.96, 0.97, 1))

        self.title_label = Label(text="Quality Test Result", bold=True, font_size="20sp",
                                 size_hint_y=None, height=60, color=(0.11, 0.11, 0.12, 1))
        root.add_widget(self.title_label)

        box = BoxLayout()
        paint_bg(box, (1, 1, 1, 1))
        self.status_label = BigStatus(text="TESTING...", color=GREY)
        box.add_widget(self.status_label)
        root.add_widget(box)

        self.detail = make_error_label(110)
        root.add_widget(self.detail)

        btns = BoxLayout(size_hint_y=None, height=60, spacing=12)
        btns.add_widget(Button(text="Restart", on_release=lambda *a: self.restart_test()))
        btns.add_widget(Button(text="Home", on_release=lambda *a: self.go_home()))
        root.add_widget(btns)
        self.add_widget(root)

    # ---------- control ----------
    def cancel_test(self):
        self.cancel_event.set()

    def run_test(self, product_key):
        """Main thread. Step 1: reset the screen, wait for the old scan to end."""
        prev = self.current_thread
        self.cancel_test()
        prod = PRODUCTS[product_key]
        self.title_label.text = "Quality Test Result - " + prod["title"]
        self.status_label.text = "TESTING..."
        self.status_label.color = GREY
        self.detail.text = ""

        self.test_id += 1
        my_id = self.test_id
        ev = threading.Event()
        self.cancel_event = ev
        self.wait_then_connect(prod, my_id, ev, prev, time.time() + 5)

    def stale(self, my_id, ev):
        return ev.is_set() or my_id != self.test_id

    def wait_then_connect(self, prod, my_id, ev, prev, until):
        if self.stale(my_id, ev):
            return
        if prev is not None and prev.is_alive() and time.time() < until:
            Clock.schedule_once(
                lambda dt: self.wait_then_connect(prod, my_id, ev, prev, until), 0.2)
            return
        self.connect(
            my_id, ev,
            on_ok=lambda port: self.start_worker(prod, my_id, ev, port),
            on_fail=lambda msg: self.update_ui(my_id, "CONNECTION ERROR", RED, msg))

    # ---------- USB connection: MAIN THREAD ONLY ----------
    def connect(self, my_id, ev, on_ok, on_fail):
        if platform != "android":
            try:
                import serial
                on_ok(serial.Serial(DESKTOP_PORT, BAUD, timeout=0.5))
            except Exception as e:
                on_fail("%s: %s" % (type(e).__name__, e))
            return

        if USB_IMPORT_ERROR:
            on_fail("USB library not loaded: " + USB_IMPORT_ERROR)
            return
        try:
            devices = list(usb.get_usb_device_list())
        except Exception as e:
            on_fail("USB list failed: %s: %s" % (type(e).__name__, e))
            return
        if not devices:
            on_fail("No USB device found. Use a data-capable OTG cable, enable OTG "
                    "in phone settings, and plug the Arduino in directly (no hub).")
            return

        state = {"queue": list(devices), "errors": [], "dev": None, "deadline": 0}

        def fail_all():
            on_fail(" | ".join(state["errors"]) + "   [found: " +
                    describe_devices(devices) + "]")

        def next_dev(*a):
            if self.stale(my_id, ev):
                return
            if not state["queue"]:
                fail_all()
                return
            dev = state["queue"].pop(0)
            state["dev"] = dev
            try:
                if usb.has_usb_permission(dev):
                    Clock.schedule_once(lambda dt: open_dev(dev), 0)
                else:
                    usb.request_usb_permission(dev)          # shows the Allow popup
                    state["deadline"] = time.time() + PERMISSION_WAIT_SEC
                    Clock.schedule_once(poll, 0.3)
            except Exception as e:
                state["errors"].append("%s: %s: %s" % (
                    dev.getDeviceName(), type(e).__name__, e))
                Clock.schedule_once(next_dev, 0)

        def poll(dt):
            if self.stale(my_id, ev):
                return
            dev = state["dev"]
            try:
                granted = usb.has_usb_permission(dev)
            except Exception as e:
                state["errors"].append("%s: %s" % (type(e).__name__, e))
                next_dev()
                return
            if granted:
                Clock.schedule_once(lambda dt: open_dev(dev), 0.5)
            elif time.time() > state["deadline"]:
                state["errors"].append("%s: permission not granted" % dev.getDeviceName())
                next_dev()
            else:
                Clock.schedule_once(poll, 0.3)

        def open_dev(dev):
            if self.stale(my_id, ev):
                return
            name = dev.getDeviceName()
            try:
                port = serial4a.get_serial_port(name, BAUD, 8, "N", 1, timeout=0.5)
                if port is not None and port.is_open:
                    on_ok(port)
                    return
                state["errors"].append("%s: no serial driver / could not open" % name)
            except Exception as e:
                state["errors"].append("%s: %s: %s" % (name, type(e).__name__, e))
            next_dev()

        next_dev()

    # ---------- reading: BACKGROUND THREAD (no Android class lookups) ----------
    def start_worker(self, prod, my_id, ev, port):
        if self.stale(my_id, ev):
            try:
                port.close()
            except Exception:
                pass
            return
        t = threading.Thread(target=self.process_test,
                             args=(prod, my_id, ev, port), daemon=True)
        self.current_thread = t
        t.start()

    def process_test(self, prod, my_id, cancel_event, ser):
        try:
            try:
                footprints = load_training(prod["training_file"])    # read-only
            except Exception as e:
                traceback.print_exc()
                self.update_ui(my_id, "FILE ERROR", RED, str(e))
                return

            points = len(footprints[0])
            median, deviation = compute_reference(footprints)

            cancel_event.wait(2)                 # Arduino resets when the port opens
            try:
                ser.reset_input_buffer()
            except Exception:
                pass

            deadline = time.time() + READ_TIMEOUT_SEC
            values = None
            while time.time() < deadline:
                if cancel_event.is_set():
                    return
                try:
                    raw = ser.readline()
                except Exception as e:
                    self.update_ui(my_id, "CONNECTION ERROR", RED,
                                   "%s: %s" % (type(e).__name__, e))
                    return
                if not raw:
                    continue
                line = raw.decode(errors="ignore").strip()
                if not line.startswith("FOOTPRINT,"):
                    continue
                parts = [p for p in line.split(",")[1:] if p]
                try:
                    cand = [int(p) for p in parts]
                except ValueError:
                    continue
                if len(cand) != points:
                    continue
                values = cand
                break

            if cancel_event.is_set():
                return
            if values is None:
                self.update_ui(my_id, "NO DATA", RED,
                               "Connected, but no FOOTPRINT scan with %d points arrived "
                               "in %d s (check baud rate %d and the sensor output)"
                               % (points, READ_TIMEOUT_SEC, BAUD))
                return

            score = score_scan(values, median, deviation)   # internal only, never shown
            if score >= MATCH_THRESHOLD:
                self.update_ui(my_id, "PURE", GREEN)
            else:
                self.update_ui(my_id, "ADULTERATED", PURE_RED)
        except Exception as e:
            traceback.print_exc()
            self.update_ui(my_id, "ERROR", RED, "%s: %s" % (type(e).__name__, e))
        finally:
            try:
                ser.close()
            except Exception:
                pass

    def update_ui(self, my_id, status, color, detail=""):
        def apply(dt):
            if my_id != self.test_id:
                return
            self.status_label.text = status
            self.status_label.color = color
            self.detail.text = detail[:ERR_LEN]
        Clock.schedule_once(apply)

    def restart_test(self):
        key = App.get_running_app().selected_key
        if key:
            self.run_test(key)

    def go_home(self):
        self.cancel_test()
        self.test_id += 1
        App.get_running_app().sm.current = "home"


class MilkApp(App):
    title = "Pure Pouch Scanner"

    def build(self):
        self.selected_key = None
        self.sm = ScreenManager()
        self.sm.add_widget(HomeScreen(name="home"))
        self.sm.add_widget(ResultScreen(name="result"))
        return self.sm

    def start_test(self, key):
        self.selected_key = key
        self.sm.get_screen("result").run_test(key)
        self.sm.current = "result"

    def show_error(self, text):
        text = text[:ERR_LEN]
        self.sm.get_screen("home").msg.text = text
        self.sm.get_screen("result").detail.text = text

    # Android pauses the app while the USB permission popup is open.
    def on_pause(self):
        return True

    def on_resume(self):
        pass

    def on_stop(self):
        self.sm.get_screen("result").cancel_test()


if __name__ == "__main__":
    MilkApp().run()
