from textual.app import App, ComposeResult
from rotation_iteration_for_disp import points_for_disp
from canvas import Canvas, Point
from fractal_view import FractalView
import jax

class DrawApp(App):
    def compose(self) -> ComposeResult:
        yield FractalView(id = "fractal")
        #yield Canvas()
    
    
    def on_mount(self) -> None:
        rows, cols = self.size.height*2, self.size.width
        p0 = (10,30)
        p1 = (50, 20)
        p2 = (10,30)
        points =  {p0: Point(count=1, prev=None), p1: Point(count=1, prev=p0), p2: Point(count=1, prev=p1)}
        last_point = p2
        grid = points_for_disp(points, last_point, cols, rows, n_iters=100)
        self.query_one(FractalView).set_grid(grid)
    

if __name__ == "__main__":
    DrawApp().run()