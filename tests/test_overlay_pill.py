from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QGuiApplication, QMouseEvent, QPixmap
from PySide6.QtWidgets import QApplication

from localflow.ui.overlay_pill import (
    COMPACT_WIDTH,
    HOVER_WIDTH,
    IDLE_HITBOX_HEIGHT,
    IDLE_HITBOX_WIDTH,
    PILL_HEIGHT,
    PREVIEW_GAP,
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
    FloatingPillWidget,
    PillMode,
)


def _screen_geo():
    screen = QGuiApplication.primaryScreen()
    assert screen is not None
    return screen.availableGeometry()


@pytest.fixture
def make_pill(qtbot):
    created: list[FloatingPillWidget] = []

    def _factory(initial_pos: tuple[int, int] | None = None) -> FloatingPillWidget:
        pill = FloatingPillWidget(initial_pos=initial_pos)
        qtbot.addWidget(pill)
        created.append(pill)
        return pill

    yield _factory

    for pill in created:
        pill.hide_pill()
        pill.close()


def test_live_preview_defaults_and_hidden_noop(make_pill) -> None:
    pill = make_pill()
    assert pill._live_preview_enabled is True
    assert pill._preview_text == ""
    assert pill._top_offset() == 0
    assert pill.size().width() == COMPACT_WIDTH
    assert pill.size().height() == PILL_HEIGHT

    pill.set_preview_text("abc")
    assert not pill.isVisible()
    assert pill._top_offset() == 0


def test_preview_show_hide_preserves_pill_y_with_initial_pos(make_pill) -> None:
    geo = _screen_geo()
    cx = geo.x() + geo.width() // 2 - COMPACT_WIDTH // 2
    cy = geo.y() + geo.height() // 2
    pill = make_pill(initial_pos=(cx, cy))

    pill.show_recording()
    assert pill.pos() == QPoint(cx, cy)
    assert pill.height() == PILL_HEIGHT
    assert pill.width() == COMPACT_WIDTH

    pill.set_preview_text("привет мир")
    expected_offset = PREVIEW_HEIGHT + PREVIEW_GAP
    assert pill._top_offset() == expected_offset
    assert pill.height() == PILL_HEIGHT + expected_offset
    assert pill.width() == PREVIEW_WIDTH
    assert pill.y() == cy - expected_offset
    assert pill.y() + pill._top_offset() == cy
    assert pill.x() + pill._pill_offset_x == cx

    pill.set_preview_text("")
    assert pill.pos() == QPoint(cx, cy)
    assert pill.size().width() == COMPACT_WIDTH
    assert pill.size().height() == PILL_HEIGHT


def test_set_live_preview_enabled_toggles_strip_and_preserves_pill_y(make_pill) -> None:
    geo = _screen_geo()
    cx = geo.x() + geo.width() // 2 - COMPACT_WIDTH // 2
    cy = geo.y() + geo.height() // 2
    pill = make_pill(initial_pos=(cx, cy))

    pill.show_recording()
    pill.set_preview_text("строка превью")
    assert pill.height() == PILL_HEIGHT + PREVIEW_HEIGHT + PREVIEW_GAP

    pill.set_live_preview_enabled(False)
    assert pill.size().width() == COMPACT_WIDTH
    assert pill.size().height() == PILL_HEIGHT
    assert pill.pos() == QPoint(cx, cy)

    pill.set_live_preview_enabled(True)
    assert pill.height() == PILL_HEIGHT + PREVIEW_HEIGHT + PREVIEW_GAP
    assert pill.y() + pill._top_offset() == cy
    assert pill.x() + pill._pill_offset_x == cx


def test_preview_preserves_pill_y_when_initial_pos_is_none(make_pill) -> None:
    pill = make_pill(initial_pos=None)
    pill.show_recording()
    base_x = pill.x()
    base_y = pill.y()

    pill.set_preview_text("текст превью")
    assert pill.y() + pill._top_offset() == base_y
    assert pill.x() + pill._pill_offset_x == base_x

    pill.set_preview_text("")
    assert pill.pos() == QPoint(base_x, base_y)


@pytest.mark.parametrize(
    ("transition", "expect_visible"),
    [
        ("show_recording", True),
        ("show_transcribing", True),
        ("show_polishing", True),
        ("show_notice", True),
        ("hide_pill", False),
    ],
)
def test_mode_transitions_clear_preview_and_restore_y(
    make_pill, transition: str, expect_visible: bool
) -> None:
    geo = _screen_geo()
    cx = geo.x() + geo.width() // 2 - COMPACT_WIDTH // 2
    cy = geo.y() + geo.height() // 2
    pill = make_pill(initial_pos=(cx, cy))

    pill.show_recording()
    pill.set_preview_text("слово один два")
    assert pill._top_offset() > 0

    if transition == "show_notice":
        pill.show_notice("Готово")
    else:
        getattr(pill, transition)()

    assert pill._preview_text == ""
    assert pill._top_offset() == 0
    assert pill.height() == PILL_HEIGHT
    if expect_visible:
        assert pill.isVisible()
        assert pill.y() == cy
    else:
        assert not pill.isVisible()


def test_multiline_and_whitespace_only_normalization(make_pill) -> None:
    geo = _screen_geo()
    cx = geo.x() + geo.width() // 2 - COMPACT_WIDTH // 2
    cy = geo.y() + geo.height() // 2
    pill = make_pill(initial_pos=(cx, cy))

    pill.show_recording()
    pill.set_preview_text("  hello\n\tworld   test  ")
    assert pill._preview_text == "hello world test"
    assert pill._top_offset() == PREVIEW_HEIGHT + PREVIEW_GAP

    pill.set_preview_text(" \n\t ")
    assert pill._preview_text == ""
    assert pill._top_offset() == 0


def test_long_preview_uses_elide_left_keeping_newest_words(make_pill) -> None:
    pill = make_pill()
    long_text = ("очень длинное начало текста диктовки " * 8) + "самые_последние_слова"
    pill.show_recording()
    pill.set_preview_text(long_text)

    max_w = PREVIEW_WIDTH - 22
    elided = pill._elided_preview_text(max_w)
    assert elided.startswith("\u2026")
    assert elided.endswith("самые_последние_слова")
    assert pill._preview_metrics.horizontalAdvance(elided) <= max_w


def test_drag_with_active_preview_saves_invariant_pill_pos(make_pill, qtbot) -> None:
    geo = _screen_geo()
    cx = geo.x() + geo.width() // 2 - COMPACT_WIDTH // 2
    cy = geo.y() + geo.height() // 2
    pill = make_pill(initial_pos=(cx, cy))

    pill.show_recording()
    pill.set_preview_text("идёт запись")
    assert pill._top_offset() == PREVIEW_HEIGHT + PREVIEW_GAP

    local_pt = QPointF(float(pill.width() // 2), float(pill.height() - 10))
    start_global = QPointF(pill.mapToGlobal(local_pt.toPoint()))
    moved_global = QPointF(start_global.x() + 35.0, start_global.y() - 20.0)

    emitted: list[tuple[int, int]] = []
    pill.position_changed.connect(lambda x, y: emitted.append((x, y)))

    press_ev = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        local_pt,
        start_global,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    move_ev = QMouseEvent(
        QEvent.Type.MouseMove,
        local_pt,
        moved_global,
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    release_ev = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        local_pt,
        moved_global,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )

    QApplication.sendEvent(pill, press_ev)
    QApplication.sendEvent(pill, move_ev)
    QApplication.sendEvent(pill, release_ev)

    expected_x = cx + 35
    expected_y = cy - 20
    assert emitted == [(expected_x, expected_y)]
    assert pill._saved_pos == QPoint(expected_x, expected_y)

    pill.show_transcribing()
    assert pill.pos() == QPoint(expected_x, expected_y)


def test_screen_clamping_near_left_right_and_top_edges(make_pill) -> None:
    geo = _screen_geo()
    mid_y = geo.y() + geo.height() // 2

    # Near left edge
    left_x = geo.x() + 4
    pill_left = make_pill(initial_pos=(left_x, mid_y))
    pill_left.show_recording()
    pill_left.set_preview_text("слева")
    assert pill_left.x() >= geo.x()
    assert pill_left.x() + pill_left._pill_offset_x == left_x

    # Near right edge
    right_x = geo.x() + geo.width() - COMPACT_WIDTH - 4
    pill_right = make_pill(initial_pos=(right_x, mid_y))
    pill_right.show_recording()
    pill_right.set_preview_text("справа")
    assert pill_right.x() + pill_right.width() <= geo.x() + geo.width()
    assert pill_right.x() + pill_right._pill_offset_x == right_x

    # Near top edge
    mid_x = geo.x() + geo.width() // 2 - COMPACT_WIDTH // 2
    top_y = geo.y() + 4
    pill_top = make_pill(initial_pos=(mid_x, top_y))
    pill_top.show_recording()
    pill_top.set_preview_text("сверху")
    assert pill_top.y() >= geo.y()
    pill_top.set_preview_text("")
    assert pill_top.pos() == QPoint(mid_x, top_y)


def test_paint_preview_and_pill_regions_via_grab(make_pill) -> None:
    geo = _screen_geo()
    cx = geo.x() + geo.width() // 2 - COMPACT_WIDTH // 2
    cy = geo.y() + geo.height() // 2
    pill = make_pill(initial_pos=(cx, cy))

    pill.show_recording()
    pill.set_preview_text("тест рендера")
    assert pill._mode == PillMode.RECORDING

    pix = QPixmap(pill.size())
    pix.fill(Qt.GlobalColor.transparent)
    pill.render(pix)
    img = pix.toImage()
    w = img.width()

    preview_row_alphas = [img.pixelColor(x, 11).alpha() for x in range(w)]
    gap_row_alphas = [img.pixelColor(x, PREVIEW_HEIGHT + 3).alpha() for x in range(w)]
    pill_row_alphas = [
        img.pixelColor(x, PREVIEW_HEIGHT + PREVIEW_GAP + PILL_HEIGHT // 2).alpha()
        for x in range(w)
    ]

    assert any(a > 0 for a in preview_row_alphas)
    assert all(a == 0 for a in gap_row_alphas)
    assert any(a > 0 for a in pill_row_alphas)


def test_hide_pill_mid_drag_resets_drag_state_and_restores_saved_pos(make_pill) -> None:
    geo = _screen_geo()
    cx = geo.x() + geo.width() // 2 - COMPACT_WIDTH // 2
    cy = geo.y() + geo.height() // 2
    pill = make_pill(initial_pos=(cx, cy))

    pill.show_recording()
    pill.set_preview_text("активное превью")
    assert pill.pos() != QPoint(cx, cy)

    local_pt = QPointF(float(pill.width() // 2), float(pill.height() - 10))
    start_global = QPointF(pill.mapToGlobal(local_pt.toPoint()))
    press_ev = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        local_pt,
        start_global,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(pill, press_ev)
    assert pill._drag_offset is not None
    assert pill.cursor().shape() == Qt.CursorShape.ClosedHandCursor

    pill.hide_pill()
    assert pill._drag_offset is None
    assert pill.cursor().shape() == Qt.CursorShape.OpenHandCursor
    assert not pill.isVisible()

    pill.show_recording()
    assert pill.pos() == QPoint(cx, cy)


def test_mid_drag_notice_to_compact_transition_preserves_anchor_and_drag(make_pill) -> None:
    geo = _screen_geo()
    cx = geo.x() + geo.width() // 2 - COMPACT_WIDTH // 2
    cy = geo.y() + geo.height() // 2
    pill = make_pill(initial_pos=(cx, cy))

    pill.show_notice("Очень длинное уведомление для проверки")
    notice_width = pill.width()
    assert notice_width > COMPACT_WIDTH
    assert pill.pos() == QPoint(cx + (COMPACT_WIDTH - notice_width) // 2, cy)

    local_pt = QPointF(float(pill.width() // 2), float(pill.height() // 2))
    start_global = QPointF(pill.mapToGlobal(local_pt.toPoint()))
    press_ev = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        local_pt,
        start_global,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(pill, press_ev)
    assert pill._drag_offset is not None

    # Transition mid-drag from wide NOTICE to compact RECORDING
    pill.show_recording()
    assert pill.width() == COMPACT_WIDTH
    assert pill.pos() == QPoint(cx, cy)

    # Continuing the drag and releasing should save the exact moved anchor
    moved_global = QPointF(start_global.x() + 24.0, start_global.y() + 12.0)
    move_ev = QMouseEvent(
        QEvent.Type.MouseMove,
        local_pt,
        moved_global,
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    release_ev = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        local_pt,
        moved_global,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    emitted: list[tuple[int, int]] = []
    pill.position_changed.connect(lambda x, y: emitted.append((x, y)))

    QApplication.sendEvent(pill, move_ev)
    QApplication.sendEvent(pill, release_ev)

    expected = QPoint(cx + 24, cy + 12)
    assert pill.pos() == expected
    assert pill._saved_pos == expected
    assert emitted == [(expected.x(), expected.y())]


def test_mid_drag_compact_to_notice_transition_preserves_anchor_and_drag(make_pill) -> None:
    geo = _screen_geo()
    cx = geo.x() + geo.width() // 2 - COMPACT_WIDTH // 2
    cy = geo.y() + geo.height() // 2
    pill = make_pill(initial_pos=(cx, cy))

    pill.show_recording()
    assert pill.pos() == QPoint(cx, cy)

    local_pt = QPointF(float(pill.width() // 2), float(pill.height() // 2))
    start_global = QPointF(pill.mapToGlobal(local_pt.toPoint()))
    press_ev = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        local_pt,
        start_global,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(pill, press_ev)

    pill.show_notice("Очень длинное уведомление для проверки")
    notice_width = pill.width()
    assert notice_width > COMPACT_WIDTH
    assert pill.pos() == QPoint(cx + (COMPACT_WIDTH - notice_width) // 2, cy)

    moved_global = QPointF(start_global.x() - 18.0, start_global.y() + 10.0)
    move_ev = QMouseEvent(
        QEvent.Type.MouseMove,
        local_pt,
        moved_global,
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    release_ev = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        local_pt,
        moved_global,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    emitted: list[tuple[int, int]] = []
    pill.position_changed.connect(lambda x, y: emitted.append((x, y)))

    QApplication.sendEvent(pill, move_ev)
    QApplication.sendEvent(pill, release_ev)

    expected = QPoint(cx - 18, cy + 10)
    assert pill._saved_pos == expected
    assert emitted == [(expected.x(), expected.y())]


def test_idle_bar_geometry_and_hover_expansion(make_pill) -> None:
    geo = _screen_geo()
    cx = geo.x() + geo.width() // 2 - COMPACT_WIDTH // 2
    cy = geo.y() + geo.height() // 2
    pill = make_pill(initial_pos=(cx, cy))

    pill.show_idle_bar()
    assert pill._mode == PillMode.IDLE_BAR
    assert pill.isVisible()
    assert pill.width() == IDLE_HITBOX_WIDTH
    assert pill.height() == IDLE_HITBOX_HEIGHT
    assert pill.y() == cy + PILL_HEIGHT - IDLE_HITBOX_HEIGHT

    # Simulate hover enter and advance animation frames to full expansion
    from PySide6.QtGui import QEnterEvent

    pill.enterEvent(QEnterEvent(QPointF(10.0, 5.0), QPointF(10.0, 5.0), QPointF(10.0, 5.0)))
    for _ in range(10):
        pill._on_tick()
    assert pill._hover_progress == pytest.approx(1.0)
    assert pill.width() == HOVER_WIDTH
    assert pill.height() == PILL_HEIGHT
    assert pill.y() == cy

    # Render expanded hover state
    pix = QPixmap(pill.size())
    pix.fill(Qt.GlobalColor.transparent)
    pill.render(pix)
    assert any(
        pix.toImage().pixelColor(x, PILL_HEIGHT // 2).alpha() > 0 for x in range(pill.width())
    )

    # Collapse back to thin resting bar
    pill._collapse_idle_hover()
    for _ in range(12):
        pill._on_tick()
    assert pill._hover_progress == pytest.approx(0.0)
    assert pill.width() == IDLE_HITBOX_WIDTH
    assert pill.height() == IDLE_HITBOX_HEIGHT


def test_left_click_without_drag_toggles_and_cancel_button_cancels(make_pill) -> None:
    geo = _screen_geo()
    cx = geo.x() + geo.width() // 2 - COMPACT_WIDTH // 2
    cy = geo.y() + geo.height() // 2
    pill = make_pill(initial_pos=(cx, cy))

    toggles: list[bool] = []
    cancels: list[bool] = []
    moves: list[tuple[int, int]] = []
    pill.toggle_requested.connect(lambda: toggles.append(True))
    pill.cancel_requested.connect(lambda: cancels.append(True))
    pill.position_changed.connect(lambda x, y: moves.append((x, y)))

    def _click(local_pt: QPointF) -> None:
        g = QPointF(pill.mapToGlobal(local_pt.toPoint()))
        for ev_type, btns in (
            (QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton),
            (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton),
        ):
            ev = QMouseEvent(
                ev_type, local_pt, g, Qt.MouseButton.LeftButton, btns, Qt.KeyboardModifier.NoModifier
            )
            QApplication.sendEvent(pill, ev)

    # Click in IDLE_BAR starts recording without saving position
    pill.show_idle_bar()
    _click(QPointF(pill.width() / 2.0, pill.height() / 2.0))
    assert len(toggles) == 1
    assert moves == []

    # Click on left '×' button during RECORDING emits cancel_requested
    pill.show_recording(command_mode=True)
    cancel_center = pill._cancel_button_rect().center()
    _click(cancel_center)
    assert len(cancels) == 1
    assert len(toggles) == 1

    # Click on center body during RECORDING emits toggle_requested (stop)
    _click(QPointF(pill.width() * 0.65, pill.height() / 2.0))
    assert len(toggles) == 2
    assert moves == []

    # Render command_mode recording pill
    pix = QPixmap(pill.size())
    pix.fill(Qt.GlobalColor.transparent)
    pill.render(pix)
    assert any(
        pix.toImage().pixelColor(x, PILL_HEIGHT // 2).alpha() > 0 for x in range(pill.width())
    )


def test_right_click_emits_context_menu_signal(make_pill) -> None:
    from PySide6.QtGui import QContextMenuEvent

    pill = make_pill()
    pill.show_idle_bar()
    received: list[QPoint] = []
    pill.context_menu_requested.connect(received.append)

    global_pos = QPoint(300, 400)
    ev = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, QPoint(10, 5), global_pos)
    QApplication.sendEvent(pill, ev)
    assert received == [global_pos]


def test_idle_hover_copy_mini_pill_emits_copy_last_without_toggling(make_pill) -> None:
    from PySide6.QtGui import QEnterEvent

    pill = make_pill()
    pill.set_has_last_dictation(True)
    pill.show_idle_bar()

    pill.enterEvent(QEnterEvent(QPointF(10.0, 5.0), QPointF(10.0, 5.0), QPointF(10.0, 5.0)))
    for _ in range(10):
        pill._on_tick()
    assert pill.width() == HOVER_WIDTH

    copied: list[bool] = []
    toggled: list[bool] = []
    pill.copy_last_requested.connect(lambda: copied.append(True))
    pill.toggle_requested.connect(lambda: toggled.append(True))

    def _click(local_pt: QPointF) -> None:
        g = QPointF(pill.mapToGlobal(local_pt.toPoint()))
        for ev_type, btns in (
            (QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton),
            (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton),
        ):
            ev = QMouseEvent(
                ev_type, local_pt, g, Qt.MouseButton.LeftButton, btns, Qt.KeyboardModifier.NoModifier
            )
            QApplication.sendEvent(pill, ev)

    # Hover & click inside the right-hand copy mini-pill
    copy_center = pill._copy_button_rect().center()
    move_ev = QMouseEvent(
        QEvent.Type.MouseMove,
        copy_center,
        QPointF(pill.mapToGlobal(copy_center.toPoint())),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(pill, move_ev)
    assert pill._copy_hovered is True

    _click(copy_center)
    assert len(copied) == 1
    assert len(toggled) == 0

    # Click on the left/main area of the expanded hover bar still toggles dictation
    _click(QPointF(40.0, PILL_HEIGHT / 2.0))
    assert len(copied) == 1
    assert len(toggled) == 1
