from textual.app import App, ComposeResult
from rotation_iteration_for_disp import points_for_disp
from canvas import Canvas, Point
from fractal_view import FractalView
import jax
from textual.events import Key
import jax.numpy as jnp

class DrawApp(App):
    def compose(self):
        yield Canvas()

    async def on_key(self, event: Key) -> None:
        if event.key == "r" and self.query(Canvas):
            canvas = self.query_one(Canvas)
            points = canvas.points
            last_point = canvas.last_point
            w, h = canvas.size.width, canvas.size.height

            await canvas.remove()

            fractal = FractalView()
            await self.mount(fractal)
            rows, cols = 2*h, w
            #p0 = (10,30)
            #p1 = (50, 20)
            #p2 = (10,30)
            #points =  {p0: Point(count=1, prev=None), p1: Point(count=1, prev=p0), p2: Point(count=1, prev=p1)}
            #last_point = p2
            grid = points_for_disp(points, last_point, cols, rows, escape_radius_squared=1.5, n_iters=150)
            self.query_one(FractalView).set_grid(grid)
    

if __name__ == "__main__":
    DrawApp().run()