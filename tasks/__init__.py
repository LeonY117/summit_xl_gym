from tasks.summit_01 import Summit
from tasks.summit_01_noBoxes import Summit as SummitNoBoxes
from tasks.summit_01_noBoxes_xy import Summit as SummitNoBoxesXY
from tasks.summit_02 import Summit as Summit_02
from tasks.summit_03 import Summit as Summit_03
from tasks.summit_04 import Summit as Summit_04
from tasks.summit_05 import Summit as Summit_05


isaacgym_task_map = {
    # "Summit": Summit,
    "Summit": SummitNoBoxes,
    # "Summit": SummitNoBoxesXY,
    "Summit_02": Summit_02,
    "Summit_03": Summit_03,
    "Summit_04": Summit_04,
    "Summit_05": Summit_05,
}
