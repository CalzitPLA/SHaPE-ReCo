"""The logo as a slowly turning token: the 3D views' background while nothing is loaded and no run goes.

The round badge (packaging/logo/make_logo.py) spins about its vertical axis like a coin: perspective faces, both
faces the right way round, a navy rim when it stands sideways, a sheen that moves with the turn and a soft shadow.
Painted with QPainter (no OpenGL); the timer runs only while the widget is visible, so a hidden token costs nothing.
`token_stack(widget)` puts a view's widget and a token into one QStackedWidget; the views switch it
(reco_gui/viewer_field.py `_sync_token`)."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

ROOT = Path(__file__).resolve().parent.parent
# the badge, sharpest first: the 1024 px render of make_logo.py --3d, then the 256 px application icon
IMAGES = (ROOT / "packaging" / "logo" / "shape-reco-3d-badge.png", ROOT / "packaging" / "shape-reco.png")


def badge_path() -> Optional[Path]:
    return next((p for p in IMAGES if p.is_file()), None)


def _hull(points):
    """Convex hull (Andrew's monotone chain) of (x, y) tuples, counter-clockwise."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


class LogoToken(QtWidgets.QWidget):
    PERIOD_S = 12.0                             # one full turn
    FPS = 24                                    # ~3 ms a frame: under a tenth of one core while it turns
    SHARE = 0.5                                 # the token's diameter / the widget's smaller side
    THICKNESS = 0.06                            # rim / diameter
    CAMERA = 3.0                                # eye distance / diameter: the perspective
    BACKGROUND = QtGui.QColor(251, 251, 250)    # the VTK views' background (0.985, 0.985, 0.98)
    RIM = QtGui.QColor("#0E1B3D")               # the badge's navy ring (make_logo.py NAVY)
    RIM_LIGHT = QtGui.QColor("#3A4F86")

    def __init__(self, parent=None, image_path: Optional[Path] = None):
        super().__init__(parent)
        self._image_path = image_path or badge_path()
        self._image: Optional[QtGui.QImage] = None           # loaded at the first paint (two views, one is never seen)
        self._scaled: Tuple[int, QtGui.QImage] = (0, QtGui.QImage())
        self._angle = 0.0
        self._t0 = time.monotonic()
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(int(1000 / self.FPS))
        self._timer.timeout.connect(self._tick)
        self.setMinimumSize(40, 40)
        self.setToolTip("SHaPe ReCo: pick a part (Settings, input) to start")

    # ---------------------------------------------------------------------------------------------- animation
    @property
    def spinning(self) -> bool:
        return self._timer.isActive()

    def showEvent(self, event):
        self._t0 = time.monotonic() - self._angle / (2 * math.pi) * self.PERIOD_S   # resumes where it stopped
        self._timer.start()
        super().showEvent(event)

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def _tick(self):
        self._angle = 2 * math.pi * (((time.monotonic() - self._t0) / self.PERIOD_S) % 1.0)
        self.update()

    def set_angle(self, angle: float):
        """The turn in radians (0: the front face square to the eye); for stills and tests."""
        self._angle = float(angle) % (2 * math.pi)
        self.update()

    # ---------------------------------------------------------------------------------------------- painting
    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        try:
            self.paint(p, self.rect(), self._angle)
        finally:
            p.end()

    def render_frame(self, width: int, height: int, angle: Optional[float] = None) -> QtGui.QImage:
        """One frame as an image (previews, tests)."""
        img = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32_Premultiplied)
        p = QtGui.QPainter(img)
        try:
            self.paint(p, QtCore.QRect(0, 0, width, height), self._angle if angle is None else angle)
        finally:
            p.end()
        return img

    def badge(self) -> QtGui.QImage:
        if self._image is None:
            self._image = QtGui.QImage(str(self._image_path)) if self._image_path else QtGui.QImage()
        return self._image

    def _face_image(self, diameter: float) -> QtGui.QImage:
        """The badge scaled once per size to about the drawn diameter: the perspective transform then samples ~1:1."""
        px = max(16, int(math.ceil(diameter * self.devicePixelRatioF())))
        if self._scaled[0] != px:
            self._scaled = (px, self.badge().scaled(px, px, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation))
        return self._scaled[1]

    def paint(self, p: QtGui.QPainter, rect: QtCore.QRect, angle: float):
        p.fillRect(rect, self.BACKGROUND)
        if self.badge().isNull():
            return
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        R = self.SHARE * min(rect.width(), rect.height()) / 2
        if R < 4:
            return
        cx, cy = rect.center().x() + 0.5, rect.center().y() + 0.5 - 0.06 * R     # a little above the middle: the shadow
        half_t, eye = self.THICKNESS * R, self.CAMERA * 2 * R
        c, s = math.cos(angle), math.sin(angle)

        def project(x, y, z):                  # turn about the vertical axis, then a pinhole eye at +z
            xr, zr = x * c + z * s, -x * s + z * c
            k = eye / (eye - zr)
            return cx + xr * k, cy + y * k

        # shadow: a soft ellipse under the token, as wide as the token looks
        ring = [(R * math.cos(2 * math.pi * i / 72), R * math.sin(2 * math.pi * i / 72)) for i in range(72)]
        rim_pts = [project(x, y, z) for z in (-half_t, half_t) for x, y in ring]
        xs = [q[0] for q in rim_pts]
        width = max(xs) - min(xs)
        shadow = QtGui.QRadialGradient(QtCore.QPointF(cx, cy + 1.16 * R), 1.0)
        shadow.setColorAt(0.0, QtGui.QColor(0, 0, 0, 55))
        shadow.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))
        p.save()
        p.translate(cx, cy + 1.16 * R)
        p.scale(max(width, 0.2 * R) * 0.55, 0.09 * R)
        p.translate(-cx, -(cy + 1.16 * R))
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QBrush(shadow))
        p.drawEllipse(QtCore.QPointF(cx, cy + 1.16 * R), 1.0, 1.0)
        p.restore()

        # rim: the hull of both faces' circles, shaded across like a cylinder
        hull = _hull(rim_pts)
        grad = QtGui.QLinearGradient(min(xs), 0, max(xs), 0)
        grad.setColorAt(0.0, self.RIM)
        grad.setColorAt(0.5 + 0.35 * s, self.RIM_LIGHT)
        grad.setColorAt(1.0, self.RIM)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QBrush(grad))
        p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in hull]))

        # the face the eye sees; the back face mirrored so that it reads the right way round as well
        if abs(c) < 0.015:
            return
        z = half_t if c > 0 else -half_t
        sx = 1.0 if c > 0 else -1.0
        img = self._face_image(2 * R)
        w, h = img.width(), img.height()
        src = QtGui.QPolygonF([QtCore.QPointF(0, 0), QtCore.QPointF(w, 0), QtCore.QPointF(w, h), QtCore.QPointF(0, h)])
        dst = QtGui.QPolygonF([QtCore.QPointF(*project(sx * u, v, z))
                               for u, v in ((-R, -R), (R, -R), (R, R), (-R, R))])
        tf = QtGui.QTransform()
        if not QtGui.QTransform.quadToQuad(src, dst, tf):
            return
        p.save()
        p.setTransform(tf, True)
        p.drawImage(0, 0, img)
        disc = QtGui.QPainterPath()
        disc.addEllipse(QtCore.QRectF(0, 0, w, h))
        p.setClipPath(disc)
        shade = int(150 * (1.0 - abs(c)) ** 1.5)                       # darker as it turns away from the eye
        if shade:
            p.fillRect(QtCore.QRectF(0, 0, w, h), QtGui.QColor(0, 0, 0, shade))
        centre = 0.5 - 0.75 * s * sx                                    # a sheen sweeping across with the turn
        sheen = QtGui.QLinearGradient(0, 0, w, h * 0.35)
        for at, alpha in ((centre - 0.22, 0), (centre, 70), (centre + 0.22, 0)):
            sheen.setColorAt(min(1.0, max(0.0, at)), QtGui.QColor(255, 255, 255, alpha))
        p.fillRect(QtCore.QRectF(0, 0, w, h), QtGui.QBrush(sheen))
        p.restore()


def token_stack(widget: QtWidgets.QWidget) -> Tuple[QtWidgets.QStackedWidget, LogoToken]:
    """`widget` (a 3D view) and a LogoToken in one stack, the token in front."""
    stack = QtWidgets.QStackedWidget()
    token = LogoToken()
    stack.addWidget(token)
    stack.addWidget(widget)
    stack.setCurrentWidget(token)
    return stack, token
