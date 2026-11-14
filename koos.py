from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical



from canvas import Canvas
from fractal_view import FractalView
import jax

class DrawApp(App):
    def compose(self) -> ComposeResult:
        #yield FractalView(id = "fractal")
        yield Canvas()
    
    '''
    def on_mount(self) -> None:
        rows, cols = self.size.height, self.size.width
        key = jax.random.PRNGKey(0)
        grid = jax.random.randint(key, (2 * rows, cols), 0, 101)
        self.query_one(FractalView).set_grid(grid)
    '''

if __name__ == "__main__":
    DrawApp().run()
