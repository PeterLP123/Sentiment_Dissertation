"""Modal dialog screens for the TUI (confirm, queue-resume, help)."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class ConfirmScreen(ModalScreen[bool]):
    DEFAULT_CSS = """
    ConfirmScreen {
        align: center middle;
    }
    #confirm-box {
        background: $surface;
        border: thick $accent;
        padding: 1 2;
        width: 70;
        height: auto;
    }
    #confirm-message {
        margin-bottom: 1;
    }
    """

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Container(id="confirm-box"):
            yield Static(self._message, id="confirm-message")
            with Horizontal():
                yield Button("Cancel", id="confirm-cancel")
                yield Button("Start run", id="confirm-ok", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-ok")


class QueueResumeScreen(ModalScreen[bool]):
    """Ask whether to resume a saved experiment queue or start fresh."""

    DEFAULT_CSS = """
    QueueResumeScreen {
        align: center middle;
    }
    #queue-resume-box {
        background: $surface;
        border: thick $accent;
        padding: 1 2;
        width: 70;
        height: auto;
    }
    #queue-resume-message {
        margin-bottom: 1;
    }
    """

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Container(id="queue-resume-box"):
            yield Static(self._message, id="queue-resume-message")
            with Horizontal():
                yield Button("Resume queue", id="queue-resume-yes", variant="primary")
                yield Button("Start new", id="queue-resume-no", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "queue-resume-yes")


class HelpScreen(ModalScreen[None]):
    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }
    #help-box {
        background: $surface;
        border: thick $accent;
        padding: 1 2;
        width: 64;
        height: auto;
    }
    #help-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #help-body {
        margin-bottom: 1;
    }
    """

    BINDINGS = [("escape", "dismiss(None)", "Close"), ("?", "dismiss(None)", "Close")]

    _HELP_LINES = [
        "1-6    Switch tabs (Dashboard, Models, Prompt, Run, Results, News)",
        "r      Refresh the dashboard",
        "s      Start the benchmark run",
        "a      Add the current config to the experiment queue",
        "g      Run the queued experiments in order",
        "c      Cancel an in-progress run or queue",
        "?      Toggle this help",
        "q      Quit",
    ]

    def compose(self) -> ComposeResult:
        with Container(id="help-box"):
            yield Static("Keyboard shortcuts", id="help-title")
            yield Static("\n".join(self._HELP_LINES), id="help-body")
            yield Button("Close", id="help-close")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "help-close":
            self.dismiss(None)
