from textual.app import App, ComposeResult

from canvas import Canvas

class DrawApp(App):
    def compose(self) -> ComposeResult:
        yield Canvas()


if __name__ == "__main__":
    DrawApp().run()
