from textual.app import App, ComposeResult, RenderResult
from textual.widget import Widget
from textual.strip import Strip
from textual.events import MouseUp, MouseDown, MouseMove
from rich.style import Style
from rich.segment import Segment

#-------- general canvas widget --------

class Canvas(Widget):
    def __init__(self):
        super().__init__()
        self.pixels: dict[tuple[int, int], str] = {} #empty dictionary storing point and its color
        self.drawing = False #determines whether point mouse is at will be painted 

    def render_line(self, y: int) -> Strip:
        width = self.size.width #gets width of terminal
        segments = [] #each line is pade up of segments of text, this is a list that contains the lines segment
        for x in range(width):
            color = self.pixels.get((x,y))
            style = Style(bgcolor=color) if color else Style()
            segments.append(Segment(" ", style)) #creates a segment based on what is contained within the pixels dict
        return Strip(segments)

    def on_mouse_down(self, event: MouseDown) -> None:
        self.capture_mouse()
        self.drawing = True
        self.paint(event.x, event.y)

    def on_mouse_move(self, event: MouseMove) -> None:
        if self.drawing:
            self.capture_mouse
            self.paint(event.x, event.y)

    def on_mouse_up(self, event: MouseUp) -> None:
        self.drawing = False
        self.refresh()

    def paint(self, x: int, y: int) -> None:
        self.pixels[(x,y)] = "white"
        self.refresh()


class Koos(App):
    def compose(self) -> ComposeResult:
        yield Canvas()

if __name__ == "__main__":
    app = Koos()
    app.run()
