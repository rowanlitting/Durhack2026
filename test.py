from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Static

from canvas import Canvas


class CanvasWindow(Vertical):
    def compose(self) -> ComposeResult:
        yield Canvas()
'''
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close":
            self.display = False
            event.stop()   # don't let the app see it
'''

class DrawApp(App):
    CSS = """
    CanvasWindow { border: solid $accent; height: 1fr; }
    Canvas { height: 1fr; }
    """
    BINDINGS = [("c", "toggle_canvas", "canvas"),("h", "toggle_home", "home")]

    def compose(self) -> ComposeResult:
        yield CanvasWindow()
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "menu-canvas":
            self.action_toggle_canvas()

    def action_toggle_canvas(self) -> None:
        window = self.query_one(CanvasWindow)
        window.display = not window.display


if __name__ == "__main__":
    DrawApp().run()
