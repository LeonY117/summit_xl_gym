from tasks.summit_01 import Summit
from tasks.summit_01_noBoxes import Summit as SummitNoBoxes
from tasks.summit_01_noBoxes_xy import Summit as SummitNoBoxesXY


isaacgym_task_map = {
    # "Summit": Summit,
    # "Summit": Summit,
    "Summit": SummitNoBoxes
    # "Summit": SummitNoBoxesXY
}
