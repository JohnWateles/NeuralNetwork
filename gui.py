import tkinter as tk
import numpy as np
from neuro import load_neural_network


class PixelDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Drawer")

        # Параметры
        self.canvas_size = 28
        self.cell_size = 20
        self.pen_size = 3

        # Массив пикселей
        self.pixels = np.zeros((self.canvas_size, self.canvas_size))

        self.nn = load_neural_network("MNIST_GUI")
        self.nn.set_input_value(self.pixels.flatten())

        # Создание интерфейса
        self.create_widgets()

        # Настройка событий
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)
        self.prev_x = None
        self.prev_y = None

    def create_widgets(self):
        # Основной контейнер
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Холст для рисования слева
        canvas_frame = tk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_size * self.cell_size,
            height=self.canvas_size * self.cell_size,
            bg="white"
        )
        self.canvas.pack(pady=5)

        # Сетка пикселей
        for i in range(self.canvas_size):
            for j in range(self.canvas_size):
                x0 = i * self.cell_size
                y0 = j * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                self.canvas.create_rectangle(x0, y0, x1, y1, fill="white", outline="gray")

        # Текстовое окно справа
        text_frame = tk.Frame(main_frame)
        text_frame.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH, expand=True)

        self.text_widget = tk.Text(
            text_frame,
            width=30,
            height=15,
            wrap=tk.WORD,
            font=("Arial", 10)
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True)

        # Добавим начальный текст
        self.show_predict()

        # Панель управления
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        tk.Button(
            control_frame,
            text="Clear Canvas",
            command=self.clear_canvas
        ).pack(side=tk.LEFT, padx=10, ipadx=10)

    def show_predict(self):
        self.nn.forward_propagation()
        output = self.nn.get_output_values()
        self.update_text("\n".join([f"{index}: {round(value, 5)}" for index, value in enumerate(output)]))

    def update_text(self, message):
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(tk.END, message)

    def draw(self, event):
        x = event.x // self.cell_size
        y = event.y // self.cell_size

        if 0 <= x < self.canvas_size and 0 <= y < self.canvas_size:
            self.canvas.create_rectangle(
                x * self.cell_size,
                y * self.cell_size,
                (x + 1) * self.cell_size,
                (y + 1) * self.cell_size,
                fill="black",
                outline="black"
            )

            self.pixels[y][x] = 1.0

            self.nn.set_input_value(self.pixels.flatten())
            self.show_predict()

            self.prev_x = x
            self.prev_y = y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.pixels = np.zeros((self.canvas_size, self.canvas_size))
        for i in range(self.canvas_size):
            for j in range(self.canvas_size):
                x0 = i * self.cell_size
                y0 = j * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                self.canvas.create_rectangle(x0, y0, x1, y1, fill="white", outline="gray")
        self.show_predict()


def main():
    root = tk.Tk()
    app = PixelDrawer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
