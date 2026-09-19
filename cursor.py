import subprocess
import wayland_automation as wa

def get_screen_size():
    output = subprocess.check_output(["xrandr"], text=True)

    for line in output.splitlines():
        if "*" in line:
            width, height = map(int, line.split()[0].split("x"))
            return width, height

    raise RuntimeError("Could not determine screen size")


def apply_dead_zone(hx, hy, prev_hx, prev_hy, threshold):
    if prev_hx is not None:
        if abs(hx - prev_hx) < threshold and abs(hy - prev_hy) < threshold: return prev_hx, prev_hy

    return hx, hy


def map_to_screen(hx, hy, cfg, screen_w, screen_h):
    def map_range(value, in_min, in_max, out_min, out_max):
        value = max(in_min, min(value, in_max))
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    x = map_range(hx, -cfg.head_x_range, cfg.head_x_range, 0, screen_w)
    y = map_range(hy,  cfg.head_y_min,   cfg.head_y_max,   0, screen_h)
    return x, y


def apply_smooth(screen_x, screen_y, smooth_x, smooth_y, alpha):
    if smooth_x is None:
        return screen_x, screen_y

    return (
        alpha * screen_x + (1 - alpha) * smooth_x,
        alpha * screen_y + (1 - alpha) * smooth_y,
    )


def move(mouse, x, y):
    mouse.click(int(x), int(y), "nothing")


def make_mouse():
    return wa.Mouse()
